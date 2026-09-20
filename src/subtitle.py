"""
SRT 字幕生成核心：音频 → 带时间轴的 SRT 字幕（引擎子进程 + 纯 Python 排版）

两条 ASR 路径（cfg.srt_asr / kwargs["asr_backend"]，默认 config.SRT_ASR）：

- qwen3_asr（默认）：Qwen3-ASR 转写（`--text-out`，保留标点）+ Qwen3-ForcedAligner
  强制对齐（`--words-out` 词级时间戳，须同时给 `--session-option
  qwen3_asr.forced_aligner_model_path=`）→ 词级时间轴 + 带标点转写文本。
- sensevoice：silero VAD 分段（`--task vad --family silero_vad`，输出语音段
  边界）→ 逐段切成独立 wav → sense_asr 批量转写（`--batch-audio-dir`，模型
  只加载一次）→ 段级时间轴。复用已缓存的 SenseVoice 权重，无额外下载，
  时间轴精度低于词级。

两条路径都归到同一组 Cue（起止秒 + 文本），由本模块统一排版成 SRT：

- 断条只在标点处：以标点为界把文本切成"从句"（标点之间的整段），从句是断条
  的原子单位，整体成条、不再从中间切开。标点不可用（未开启或对齐失败）时
  退回 停顿 / 单条最长秒数 / 每屏容量 三个约束——词级路径用词间时距，段级
  路径按字符宽度比例拆分该段时长。
- 从句贪心累积到每屏容量（每行宽度 × 每屏行数）；句末标点处成句即断、句间
  停顿超过阈值即断、累积超过单条最长秒数即断。只有单个从句本身就超过一屏
  容量时，才在该从句内部按条目边界硬切（标点之间无处可退时的最后手段），
  切出的后半段继续与后续从句一起累积。
- 断行按显示宽度（CJK=2、其余=1），折行点同样优先标点（但以不撑宽行为前提）；
  中文之间不插空格，拉丁词之间保留空格。

推理（VAD / ASR / 对齐）全部在 audiocpp C++ 引擎侧完成，Python 只做分段落盘、
结果汇总与文本排版（与 pipeline.py 的输出命名同性质，不做模型推理）。

模型权重：SenseVoice 走 ASR 现有路径；Qwen3-ASR / Qwen3-ForcedAligner 本地
手放优先，缺失经 HF 下载并留在 HF 默认缓存（工程 models/ 仅支持手工放置）。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field

from . import segment_llm
from .audiocpp import (
    _backend_flag,
    _ensure_binary,
    _src_dir,
    ensure_tmp_dir,
    run_cli,
)
from .config import (
    Config,
    MODELS_DIR,
    SRT_ASR,
    SRT_BREAK_ON_COMMA,
    SRT_CPS_MAX,
    SRT_ENUM_COMMA_AS_SPACE,
    SRT_HOTWORDS,
    SRT_HOTWORDS_FILE,
    SRT_HOTWORDS_PROMPT,
    SRT_ITN,
    SRT_LINE_BALANCE,
    SRT_MAX_BLOCK_SECONDS,
    SRT_MAX_GAP_SECONDS,
    SRT_MAX_LINES,
    SRT_MAX_LINE_WIDTH,
    SRT_MIN_BLOCK_SECONDS,
    SRT_MIN_CUE_WIDTH,
    SRT_MIN_LINE_WIDTH,
    SRT_PUNCTUATION,
    SRT_QWEN3_ALIGNER_FILE,
    SRT_QWEN3_ALIGNER_LOCAL,
    SRT_QWEN3_ALIGNER_REPO,
    SRT_QWEN3_ASR_FILE,
    SRT_QWEN3_ASR_LOCAL,
    SRT_QWEN3_ASR_REPO,
    SRT_QWEN3_PUNCTUATION,
    SRT_VAD_MERGE_GAP,
    SRT_VAD_MIN_SPEECH,
    SRT_VAD_MODEL,
    TMP_DIR,
)
from .hf import _ensure_gguf_file
from .sensevoice import _ensure_model as _ensure_sensevoice
from .sensevoice import _to_16k_mono

# ASR 输出统一以 16 kHz mono 送入引擎，词/段时间戳的样本率即 16000
_SR = 16000

# VAD 族与引擎自带资源相对 audiocpp 源码目录的路径
_VAD_FAMILY = "silero_vad"
_VAD_MODEL_REL = os.path.join("assets", "framework", "models", "silero_vad")


@dataclass
class Cue:
    """一条字幕（起止秒 + 文本）。"""

    start: float
    end: float
    text: str


@dataclass
class SrtResult:
    """一次字幕生成的产物。"""

    srt_path: str                 # 已写入的 SRT 文件绝对路径
    srt_text: str                 # SRT 全文（供界面预览）
    cues: list[Cue] = field(default_factory=list)
    duration_sec: float = 0.0     # 音频时长（16 kHz 时长）
    asr_backend: str = ""         # 实际使用的 ASR 路径


# ── 排版参数 ────────────────────────────────────────────────

@dataclass
class SrtOptions:
    """字幕排版与时间轴约束（常量在 config.py 顶部，UI 可运行期覆盖）。"""

    max_width: int
    max_lines: int
    max_block_seconds: float
    max_gap_seconds: float
    min_block_seconds: float
    break_on_comma: bool       # 逗号（，）处是否即断条
    cps_max: float             # 单条字幕语速上限（CJK 字数/秒，0 = 不检查）
    min_line_width: int        # 孤行阈值（宽度），行宽低于它算孤行
    line_balance: float        # 折行均衡权重（0 = 行内尽量填满）
    min_cue_width: int         # 碎条阈值（宽度），低于此宽度尽量并入相邻条
    punctuation: bool          # 字幕文本是否输出标点
    enum_comma_space: bool     # 顿号（、）是否输出为空格


def _pick(kwargs: dict, key: str, default):
    """kwargs 覆盖优先；None / 空串回落到 config 常量。"""
    v = kwargs.get(key)
    return default if v is None or v == "" else v


def _pick_bool(kwargs: dict, key: str, default: bool) -> bool:
    """布尔开关：kwargs 里明确给了就用它，否则用 config 常量。"""
    v = kwargs.get(key)
    if v is None or v == "":
        return bool(default)
    return bool(v)


def _hotwords_text(hotwords, cfg: Config) -> str:
    """ASR 热词/上下文文本（仅 qwen3_asr 路径使用）。

    取值顺序：调用方 kwargs → cfg.srt_hotwords → config.SRT_HOTWORDS →
    工程根目录 hotword.txt（web 每次提交字幕任务时写回的文件）。
    """
    text = str(hotwords if hotwords is not None
               else cfg.srt_hotwords or SRT_HOTWORDS).strip()
    if text:
        return text
    if SRT_HOTWORDS_FILE and os.path.isfile(SRT_HOTWORDS_FILE):
        try:
            with open(SRT_HOTWORDS_FILE, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""
    return ""


def _hotwords_prompt(hotwords: str) -> str:
    """把热词拼成 Qwen3-ASR 的系统提示词（引擎 `--text`）。

    模板 SRT_HOTWORDS_PROMPT 含 {hotwords} 时替换，不含时拼在模板末尾；
    模板留空 = 直接把热词文本当系统提示词。
    """
    tpl = str(SRT_HOTWORDS_PROMPT or "").strip()
    if not tpl:
        return hotwords
    if "{hotwords}" in tpl:
        return tpl.replace("{hotwords}", hotwords)
    return tpl + hotwords


def _options(kwargs: dict) -> SrtOptions:
    return SrtOptions(
        max_width=int(_pick(kwargs, "max_line_width", SRT_MAX_LINE_WIDTH)),
        max_lines=int(_pick(kwargs, "max_lines", SRT_MAX_LINES)),
        max_block_seconds=float(
            _pick(kwargs, "max_block_seconds", SRT_MAX_BLOCK_SECONDS)),
        max_gap_seconds=float(
            _pick(kwargs, "max_gap_seconds", SRT_MAX_GAP_SECONDS)),
        min_block_seconds=float(
            _pick(kwargs, "min_block_seconds", SRT_MIN_BLOCK_SECONDS)),
        cps_max=float(_pick(kwargs, "cps_max", SRT_CPS_MAX)),
        break_on_comma=_pick_bool(kwargs, "break_on_comma", SRT_BREAK_ON_COMMA),
        min_line_width=int(_pick(kwargs, "min_line_width", SRT_MIN_LINE_WIDTH)),
        line_balance=float(_pick(kwargs, "line_balance", SRT_LINE_BALANCE)),
        min_cue_width=int(_pick(kwargs, "min_cue_width", SRT_MIN_CUE_WIDTH)),
        punctuation=_pick_bool(kwargs, "punctuation", SRT_PUNCTUATION),
        enum_comma_space=_pick_bool(kwargs, "enum_comma_space",
                                    SRT_ENUM_COMMA_AS_SPACE),
    )


# ── 文本宽度、标点与折行（CJK 按 2 计；中文之间不插空格）────

_CJK_RANGES = (
    (0x1100, 0x115F), (0x2E80, 0xA4CF), (0xAC00, 0xD7A3), (0xF900, 0xFAFF),
    (0xFE30, 0xFE6F), (0xFF00, 0xFF60), (0xFFE0, 0xFFE6), (0x20000, 0x3FFFD),
)

# 从句末尾的断条级别（见 _break_kind）
_BREAK_NONE = ""              # 不是断条点：只作从句边界
_BREAK_COMMA = "comma"        # 逗号：SRT_BREAK_ON_COMMA 打开时无条件断条
_BREAK_SENTENCE = "sentence"  # 句末标点：宽度够即断条

# 断句标点：句末（成句即可断）与句内（超限时优先在此断）；数字里的 "."
# 由 _punct_kind 另行排除（4.5 / 1,000 / twenty-two 不算断句）
_SENTENCE_PUNCT = "。！？!?…."
# 逗号（SRT_BREAK_ON_COMMA 开关作用的对象）与其余句内标点（只作从句边界）
_COMMA_PUNCT = "，,"
_CLAUSE_PUNCT = "，、；：,;:-—"
_BREAK_PUNCT = _SENTENCE_PUNCT + _CLAUSE_PUNCT


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _is_break_punct(ch: str) -> bool:
    return ch in _BREAK_PUNCT


def _norm_char(ch: str) -> str:
    """归一化用于对齐定位：只保留字母/数字（含 CJK 文字与全角字母数字）。

    不能按"CJK 区块"判断正文：，。！？：；等全角标点本身就落在全角/CJK 兼容
    区块内（U+FF00–FF60、U+2E80–A4CF），按区块判断会把标点当成正文；这里用
    isalnum()（标点类别 Po/Ps/Pe 均为 False）区分，再排除断句标点与空白。
    """
    if ch.isspace() or _is_break_punct(ch):
        return ""
    return ch.lower() if ch.isalnum() else ""


def _char_width(ch: str) -> int:
    """显示宽度：CJK 与全角字符按 2 计，其余按 1。"""
    return 2 if _is_cjk(ch) else 1


def _text_width(text: str) -> int:
    return sum(_char_width(ch) for ch in text)


def _needs_space(left: str, right: str) -> bool:
    """原子之间是否插空格：两侧都是非 CJK（拉丁词）才插。"""
    return bool(left) and not _is_cjk(left[-1]) and not _is_cjk(right[0])


def _join_atoms(atoms: list[str]) -> str:
    """把排版原子拼回文本：拉丁词之间补空格，含 CJK 处不补。"""
    out = ""
    for a in atoms:
        if out and _needs_space(out, a):
            out += " "
        out += a
    return out


def _punct_kind(text: str, i: int) -> str:
    """位置 i 的标点类别："sentence"（句末）/ "clause"（句内）/ ""（不是断句标点）。

    数字内部的 "." "," "-"（如 4.5、1,000）与词内连字符（twenty-two）不算断句：
    前者至少一侧是数字，后者两侧是字母/数字。
    """
    ch = text[i]
    if ch in (".", ",", "-"):
        prev = text[i - 1] if i > 0 else " "
        nxt = text[i + 1] if i + 1 < len(text) else " "
        if ch == "-":
            if prev.isalnum() or nxt.isalnum():
                return ""
        elif prev.isdigit() or nxt.isdigit():
            return ""
    if ch in _SENTENCE_PUNCT:
        return "sentence"
    if ch in _BREAK_PUNCT:
        return "clause"
    return ""


def _text_items(text: str) -> list[tuple[str, float, float, bool, bool]]:
    """把文本切成排版条目：(原子, 0.0, 0.0, 句内标点在前, 句末标点在前)。

    与词级条目同构（时间列留 0），使断条与折行共用同一套"标点优先"逻辑。
    """
    raw: list[list] = []            # [原子, 其后是句内标点, 其后是句末标点]
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            raw.append([buf, False, False])
            buf = ""

    for i, ch in enumerate(text):
        kind = _punct_kind(text, i)
        if ch.isspace():
            flush()
        elif kind:
            # 标点并入前一个原子：既保留到输出文本，又作为断句标记
            flush()
            if raw:
                raw[-1][0] += ch
                raw[-1][1] = True
                if kind == "sentence":
                    raw[-1][2] = True
            else:                        # 文本以标点开头（罕见）：单独成原子
                raw.append([ch, False, False])
        elif _is_cjk(ch):
            flush()
            raw.append([ch, False, False])
        else:
            buf += ch
    flush()

    items: list[tuple[str, float, float, bool, bool]] = []
    prev_clause = False
    prev_sentence = False
    for atom, clause_after, sentence_after in raw:
        items.append((atom, 0.0, 0.0, prev_clause, prev_sentence))
        prev_clause, prev_sentence = clause_after, sentence_after
    return items


def _align_tokens(text: str, tokens: list[str]
                  ) -> list[tuple[bool, bool, str]]:
    """把词条对齐到带标点转写，返回每个词条的 (句内标点在前, 句末标点在前, 文本片段)。

    词级路径的词条本身不含标点、也不保证覆盖全部字符（对齐器可能漏词），
    因此字幕文本不直接用词条拼接，而是取转写文本的**逐词片段**：第 i 个词条
    的片段 = 从它的首字符到下一个词条首字符之间的原文（去首尾空白）。这样
    标点与漏掉的字符都会随片段保留下来，输出与转写文本一致、不丢字；
    片段同时用作"第 i 个词之前可断句"的定位依据。
    """
    chars: list[str] = []
    flags: list[tuple[bool, bool]] = []
    char_pos: list[int] = []             # 归一化字符 → 原文下标
    pending_clause = False
    pending_sentence = False
    for i, ch in enumerate(text):
        norm = _norm_char(ch)
        if norm:
            chars.append(norm)
            flags.append((pending_clause, pending_sentence))
            char_pos.append(i)
            pending_clause = pending_sentence = False
        elif not ch.isspace():
            kind = _punct_kind(text, i)
            if kind == "sentence":
                pending_clause = pending_sentence = True
            elif kind == "clause":
                pending_clause = True
    stream = "".join(chars)

    # 第一遍：定位每个词条在原文中的字符区间
    starts: list[int | None] = []
    matched: list[tuple[bool, bool]] = []
    cursor = 0
    for tok in tokens:
        norm_tok = "".join(_norm_char(c) for c in tok)
        pos = stream.find(norm_tok, cursor) if norm_tok else -1
        if pos < 0:                      # 对不上的词条：不给断句偏好，也不占片段
            starts.append(None)
            matched.append((False, False))
            continue
        end = pos + len(norm_tok)
        starts.append(char_pos[pos])
        matched.append(flags[pos])
        cursor = end

    # 第二遍：片段 = 本词条首字符 → 下一个已匹配词条首字符（末尾到文末），
    # 去掉首尾空白；未匹配的词条其文本会被前一个片段带上，不会丢字
    out: list[tuple[bool, bool, str]] = []
    for idx, start in enumerate(starts):
        if start is None:
            out.append((*matched[idx], ""))
            continue
        nxt = len(text)
        for later in starts[idx + 1:]:
            if later is not None and later > start:
                nxt = later
                break
        out.append((*matched[idx], text[start:nxt].strip()))
    return out


def _prefix_widths(items: list) -> list[int]:
    """各前缀的显示宽度（含原子间空格），返回长度 len(items)+1 的列表。

    空格规则与 _join_atoms 一致：仅当相邻原子首尾都不是 CJK 时插一个空格。
    """
    widths = [0]
    for j, it in enumerate(items):
        add = _text_width(it[0])
        if j and _needs_space(items[j - 1][0], it[0]):
            add += 1
        widths.append(widths[-1] + add)
    return widths


def _can_join(a: str, b: str, opts: SrtOptions) -> bool:
    """两段文字合并后是否仍在每屏容量内（按装箱后的行数判断）。"""
    text = _join_atoms([a, b])
    return bool(text) and len(_layout(_text_items(text), opts)) <= opts.max_lines


# ── 断条：标点边界切从句，从句累积成条 ──────────────────────
#
# 条目统一为 (文本片段, 起秒, 止秒, 句内标点在前, 句末标点在前)：
# 词级条目来自词条 + 转写片段对齐（_items_from_words），文本条目来自
# _text_items（时间列留 0，段级路径另按宽度比例分配时长）。两条路径共用
# 下面这套从句切分与累积逻辑，断点只落在标点上。

def _items_from_words(words: list[tuple[str, float, float]],
                      breaks: list[tuple[bool, bool, str]]) -> list[tuple]:
    """词级条目：文本取转写片段（带标点），标志位 = 本词条之前有标点。"""
    items: list[tuple] = []
    for idx, (atom, start, end) in enumerate(words):
        clause, sentence, piece = (breaks[idx] if idx < len(breaks)
                                   else (False, False, ""))
        items.append((piece or atom, start, end, clause, sentence))
    return items


# 末尾成对的闭合符号（引号 / 右括号）：判断"末尾是不是标点"时先剥掉
_CLOSING_MARKS = "”\"’'）)】」』》〉>"


def _strip_tail(text: str) -> str:
    """去掉末尾空白与闭合符号，便于判断这条文字的收尾标点。"""
    return text.rstrip().rstrip(_CLOSING_MARKS).rstrip()


def _break_kind(text: str, opts: SrtOptions) -> str:
    """该从句末尾的断条级别（从句能否直接收条）：

    - "sentence"：句末标点（。！？），倾向在此收条（见 _break_cost）；
    - "comma"：逗号（SRT_BREAK_ON_COMMA 打开时），无条件收条；
    - ""：其余句内标点（顿号、分号、冒号），只作从句边界，容量不够时才在这里断。
    """
    stripped = _strip_tail(text)
    if not stripped:
        return _BREAK_NONE
    if stripped[-1] in _SENTENCE_PUNCT:
        return _BREAK_SENTENCE
    if opts.break_on_comma and stripped[-1] in _COMMA_PUNCT:
        return _BREAK_COMMA
    return _BREAK_NONE


def _units(items: list[tuple],
           opts: SrtOptions) -> list[tuple[list[tuple], str]]:
    """按标点边界把条目流切成从句单元：[(条目列表, 断条级别)]。

    从句单元是断条的原子单位：单元内部没有标点（标点都紧跟在它前面的那个
    条目里），因此只能整体成条；单元之间可以断条，级别决定能否直接收条
    （见 _break_kind）。标点不可用（未开启或对齐失败）时全部条目属同一个
    单元，退化为纯容量切分。
    """
    units: list[tuple[list[tuple], str]] = []
    cur: list[tuple] = []
    kind = _BREAK_NONE
    for it in items:
        if cur and (it[3] or it[4]):          # 本条目之前有标点：从句边界
            units.append((cur, kind or _break_kind(cur[-1][0], opts)))
            cur, kind = [], _BREAK_NONE
        cur.append(it)
        if any(c in _SENTENCE_PUNCT for c in it[0]):
            kind = _BREAK_SENTENCE
    if cur:
        units.append((cur, kind or _break_kind(cur[-1][0], opts)))
    return units


def _append_width(cur: list, width: int, items: list) -> int:
    """把 items 接到 cur 之后的行宽度（含必要的空格）。"""
    if not items:
        return width
    add = _prefix_widths(items)[-1]
    if cur and _needs_space(cur[-1][0], items[0][0]):
        add += 1
    return width + add


# ── 折行：从句整体成行，从句自己过宽时在从句内均衡拆行 ──────
#
# 从句边界之外的拆行点（只在从句自己宽过一行时才允许）按"词边界是否安全"
# 分档：前一个字是虚词或标点、且后一个字不是虚词时算安全（"…改造是│新与旧"
# 这类）；否则算硬拆（可能拆开词，如 "废│弃"）。代价按 max_width² 归一，
# 换行宽度上限不同时行为基本一致。
# 断在其后通常不拆词的单字虚词（"…文化与│商业"、"…完成了│从…"）
_LINE_SAFE_END = "的了是和与在把被对为及或则"
# 断在其前不宜（行首出现虚词）；与下一个字常组成双字词的字（更加/而且/就是…）
_LINE_BAD_NEXT = "且是有加多种么些个什"
_LINE_NO_END = "“‘「『（《\"'("                              # 行尾不宜为开引号/开括号
_LINE_ORPHAN_FACTOR = 3.0      # 孤行（宽度 < SRT_MIN_LINE_WIDTH）代价系数
_LINE_UNSAFE_FACTOR = 1.5      # 从句内硬拆（可能拆词）代价系数
_LINE_QUOTE_FACTOR = 0.03      # 行尾开引号代价系数


def _cut_safe(prev_text: str, next_text: str) -> bool:
    """从句内部拆行是否落在"词边界安全"处。

    前一个条目以虚词或断句标点收尾、且下一个条目不以虚词起首时算安全；
    其余位置可能把词拆开（对齐器给的是逐字词条，没有词边界信息）。
    """
    prev = _strip_tail(prev_text)
    if not prev:
        return False
    if prev[-1] not in _LINE_SAFE_END and prev[-1] not in _BREAK_PUNCT:
        return False
    return next_text[:1] not in _LINE_BAD_NEXT


def _line_levels(items: list, opts: SrtOptions) -> dict[int, int]:
    """每个可断位置 → 覆盖其后条目所需的最少行数（n → 0）。

    与 _layout 用同一套可断位置与行宽约束，因此 _fits 判定（levels[0] <=
    max_lines）与真实渲染行数一致。
    """
    n = len(items)
    if n == 0:
        return {0: 0}
    prefix = _prefix_widths(items)
    cuts = [0] + _line_cut_positions(items, opts)
    inf = float("inf")
    levels: dict[int, int] = {n: 0}
    for k in range(len(cuts) - 2, -1, -1):
        i = cuts[k]
        best = inf
        for m in range(k + 1, len(cuts)):
            j = cuts[m]
            width = prefix[j] - prefix[i]
            if width > opts.max_width and m > k + 1:
                break                     # 除单个超宽条目外，行不得超宽
            cand = 1 + levels.get(j, inf)
            if cand < best:
                best = cand
            if width > opts.max_width:
                break
        levels[i] = best
    return levels


def _min_lines(items: list, opts: SrtOptions) -> int:
    """装箱后的最少行数：换行只落在从句边界（超长从句内部也允许）。"""
    if not items:
        return 0
    return int(_line_levels(items, opts).get(0, 1))


def _line_cut_positions(items: list, opts: SrtOptions) -> list[int]:
    """可作为行结束的条目下标（升序，不含 0）：从句边界，加上超长从句内部。

    从句（标点之间的整段）是折行的原子单位：装不下就整段挪到下一行；只有
    从句自己就宽过一行（标点之间无处可断）时，才允许在该从句内部拆分。
    """
    positions: list[int] = []
    acc = 0
    for unit_items, _kind in _units(items, opts):
        start = acc
        acc += len(unit_items)
        if _prefix_widths(unit_items)[-1] > opts.max_width:
            positions.extend(range(start + 1, acc))
        positions.append(acc)
    return positions


def _line_cost(line: list, width: int, ideal: float, opts: SrtOptions,
               next_text: str | None) -> float:
    """一行的代价：偏离理想行宽（均衡）+ 孤行 + 从句内硬拆 + 行尾开引号。

    优先级由系数决定（按 max_width² 归一）：先避免孤行，再避免把词拆开，
    最后才追求两行等宽。next_text = 下一行首条目文本；None 表示本行是末行。
    """
    scale = max(opts.line_balance, 1e-6) * float(opts.max_width) ** 2
    cost = opts.line_balance * (width - ideal) ** 2
    floor = opts.min_line_width
    if floor > 0 and width < floor:
        cost += _LINE_ORPHAN_FACTOR * scale * (floor - width) / floor
    if line and line[-1][0][-1:] in _LINE_NO_END:
        cost += _LINE_QUOTE_FACTOR * scale
    if next_text is not None and line and not _cut_safe(line[-1][0], next_text):
        cost += _LINE_UNSAFE_FACTOR * scale
    return cost


def _greedy_layout(items: list, opts: SrtOptions) -> list[list[tuple]]:
    """行内尽量填满（不均衡、不避孤行）：SRT_LINE_BALANCE = 0 时的旧行为。"""
    max_width = opts.max_width
    lines: list[list[tuple]] = []
    cur: list[tuple] = []
    width = 0
    for unit_items, _kind in _units(items, opts):
        if _prefix_widths(unit_items)[-1] <= max_width:
            need = _append_width(cur, width, unit_items)
            if cur and need > max_width:
                lines.append(cur)
                cur, width = [], 0
                need = _append_width(cur, width, unit_items)
            cur, width = cur + unit_items, need
            continue
        for it in unit_items:                 # 超长从句：在当前行内逐条目填满
            need = _append_width(cur, width, [it])
            if cur and need > max_width:
                lines.append(cur)
                cur, width = [], 0
                need = _append_width(cur, width, [it])
            cur, width = cur + [it], need
    if cur:
        lines.append(cur)
    return lines


def _layout(items: list, opts: SrtOptions) -> list[list[tuple]]:
    """一条字幕的行装箱：从句整体成行，从句自己过宽时在从句内均衡拆行。

    行数取最少行数，理想行宽 = 总宽 / 行数，再按"偏离理想行宽² + 孤行罚 +
    行首行尾禁则"在所有可断位置里取全局最优。因此长从句会被拆成两行大致
    等宽（"…废弃的国际家具博览中心" → "…废弃的国际家具 / 博览中心"），而不是
    填满首行、末行只剩一两个字（旧的贪心会给出 "…国际家 / 具博览中心"）。
    """
    n = len(items)
    if n == 0:
        return []
    if opts.line_balance <= 0:
        return _greedy_layout(items, opts)
    prefix = _prefix_widths(items)
    cuts = [0] + _line_cut_positions(items, opts)
    levels = _line_levels(items, opts)
    limit = max(1, int(levels.get(0, 1)))
    ideal = prefix[-1] / limit
    index = {p: k for k, p in enumerate(cuts)}
    memo: dict[int, tuple[float, list[int]]] = {}

    def rec(i: int) -> tuple[float, list[int]]:
        if i == n:
            return 0.0, []
        if i in memo:
            return memo[i]
        best: tuple[float, list[int]] | None = None
        for m in range(index[i] + 1, len(cuts)):
            j = cuts[m]
            if levels.get(j, -1) != levels.get(i, 0) - 1:
                continue                      # 只走最少行数路径（与 _fits 一致）
            width = prefix[j] - prefix[i]
            if width > opts.max_width and m > index[i] + 1:
                break
            tail = rec(j)
            nxt = items[j][0] if j < n else None
            cost = _line_cost(items[i:j], width, ideal, opts, nxt) + tail[0]
            if best is None or cost < best[0]:
                best = (cost, [j] + tail[1])
            if width > opts.max_width:
                break
        if best is None:                      # 理论不可达：整条一行，避免死循环
            best = (0.0, [n])
        memo[i] = best
        return best

    ends = rec(0)[1]
    lines: list[list[tuple]] = []
    prev = 0
    for end in ends:
        lines.append(items[prev:end])
        prev = end
    if prev < n:                              # 兜底：不允许丢条目
        lines.append(items[prev:])
    return lines


def _fits(items: list, opts: SrtOptions) -> bool:
    """该条装箱后是否在每屏行数内（与 _layout 同约束的最少行数判定）。"""
    if not items:
        return True
    return _min_lines(items, opts) <= opts.max_lines


def _flush(cues: list[Cue], items: list) -> None:
    """把 items 收成一条字幕（时间取首尾条目）。"""
    text = _join_atoms([it[0] for it in items])
    if text:
        cues.append(Cue(start=items[0][1], end=items[-1][2], text=text))


# ── 断条：整篇取代价最优（A3）────────────────────────────────
#
# 断条点只能在从句边界（标点之间的整段；从句自己超过一屏时可在其内部硬拆），
# 因此任何断法都不切词组。在此前提下全篇取代价最小的断法：
#   每条字幕：过短罚（宽度 < SRT_MIN_CUE_WIDTH）、语速过快罚（> SRT_CPS_MAX）
#   每个断点：句末标点/逗号 = 负代价（倾向在此断）；顿号/分号/冒号 = 正代价；
#             从句内硬拆（可能拆词）= 更大正代价；明显停顿（> max_gap/2）= 小幅负代价
# 停顿超过 SRT_MAX_GAP_SECONDS 的位置必须断（硬约束）。代价按
# SRT_LINE_BALANCE × max_width² 归一，换行宽度不同也能保持同样的取舍倾向。
_CUE_SHORT_FACTOR = 2.0      # 过短（宽度 < min_cue_width）代价系数
_CUE_CPS_FACTOR = 1.0        # 语速过快代价系数
_BREAK_REWARD_FACTOR = 0.25  # 句末标点/逗号处断条的负代价（倾向断）
_BREAK_PAUSE_FACTOR = 0.12   # 明显停顿处断条的负代价
_BREAK_CLAUSE_FACTOR = 0.6   # 顿号/分号/冒号处断条的代价
_BREAK_HARD_FACTOR = 1.2     # 从句内硬拆（可能拆词）的代价
_BREAK_LLM_FACTOR = 0.35     # 小模型指定的断点：更强的正向倾向
_BREAK_PRIOR_FACTOR = 0.08   # 有小模型分组时，标点/停顿自身的倾向降到先验级


def _cue_cost(text_items: list, duration: float,
              opts: SrtOptions, scale: float) -> float:
    """一条字幕的代价：过短 + 语速过快（都是负向的观感指标）。

    text_items 取"该条渲染文本的条目"（与折行、渲染同一份解析），宽度因此与
    实际排版一致。
    """
    width = _prefix_widths(text_items)[-1] if text_items else 0
    cost = 0.0
    floor = opts.min_cue_width
    if floor > 0 and width < floor:
        cost += _CUE_SHORT_FACTOR * scale * (floor - width) / floor
    if opts.cps_max > 0 and duration > 0:
        cps = (width / 2.0) / duration          # CJK 字数 / 秒
        if cps > opts.cps_max:
            cost += (_CUE_CPS_FACTOR * scale
                     * (cps - opts.cps_max) / opts.cps_max)
    return cost


def _break_cost(items: list, p: int, opts: SrtOptions, scale: float,
                preferred: frozenset = frozenset()) -> float:
    """在 items[p] 之前断条的代价（负 = 倾向在此断）。

    preferred = 小模型（SRT_LLM breaks）指定的断点：命中则给更强的正向倾向；
    其余位置的标点/停顿倾向降为先验级，让小模型的语义判断占主导（硬约束
    容量/最长秒数/停顿不受影响）。
    """
    if p <= 0 or p >= len(items):
        return 0.0
    if preferred and p in preferred:
        return -_BREAK_LLM_FACTOR * scale
    prior = _BREAK_PRIOR_FACTOR if preferred else _BREAK_REWARD_FACTOR
    pause = _BREAK_PRIOR_FACTOR if preferred else _BREAK_PAUSE_FACTOR
    kind = _break_kind(items[p - 1][0], opts)
    if kind:
        return -prior * scale                   # 句末标点 / 逗号
    tail = _strip_tail(items[p - 1][0])
    if tail and tail[-1] in _CLAUSE_PUNCT:
        return _BREAK_CLAUSE_FACTOR * scale     # 顿号 / 分号 / 冒号
    if items[p][1] - items[p - 1][2] > opts.max_gap_seconds / 2:
        return -pause * scale                   # 明显停顿：小幅倾向
    return _BREAK_HARD_FACTOR * scale           # 从句内硬拆（可能拆词）


def _cue_cut_positions(items: list, opts: SrtOptions) -> list[int]:
    """可作条末的条目下标（升序，含 n）：从句边界 + 超容量从句内部的硬拆点。"""
    positions: list[int] = []
    acc = 0
    for unit_items, _kind in _units(items, opts):
        start = acc
        acc += len(unit_items)
        if not _fits(unit_items, opts):         # 从句自己超过一屏：内部可硬拆
            positions.extend(range(start + 1, acc))
        positions.append(acc)
    return positions


def _fallback_cues(items: list, opts: SrtOptions) -> list[list[tuple]]:
    """异常兜底：逐条目贪心累积，直到再装一个就超出每屏容量。"""
    out: list[list[tuple]] = []
    cur: list[tuple] = []
    for it in items:
        cand = cur + [it]
        if cur and not _fits(_text_items(_join_atoms([x[0] for x in cand])),
                             opts):
            out.append(cur)
            cur = [it]
            continue
        cur = cand
    if cur:
        out.append(cur)
    return out


def _preferred_cuts(items: list, opts: SrtOptions,
                    hint: list[int] | None) -> frozenset:
    """把"每组含几个从句"（小模型输出）换算成"倾向在此断条"的条目下标。"""
    if not hint:
        return frozenset()
    ends: list[int] = []
    acc = 0
    for unit_items, _kind in _units(items, opts):
        acc += len(unit_items)
        ends.append(acc)
    out: set[int] = set()
    k = 0
    for size in hint:
        k += max(int(size), 1)
        if k < len(ends):
            out.add(ends[k - 1])
    return frozenset(out)


def _dp_cues(items: list, opts: SrtOptions,
             hint: list[int] | None = None) -> list[list[tuple]]:
    """整篇代价最小的断条：返回每条的条目列表（动态规划）。

    候选断点见 _cue_cut_positions，容量与最长秒数是硬约束（不可行即跳过），
    其余（过短、语速、断点位置）按代价取舍。复杂度 O(n × 每条最多条目数)。
    """
    n = len(items)
    if n == 0:
        return []
    scale = max(opts.line_balance, 1e-6) * float(opts.max_width) ** 2
    preferred = _preferred_cuts(items, opts, hint)
    forced = {p for p in range(1, n)
              if items[p][1] - items[p - 1][2] > opts.max_gap_seconds}
    forced.add(n)                              # 末尾必断（收尾）
    positions = sorted(set(_cue_cut_positions(items, opts)) | forced)
    positions = [0] + [p for p in positions if 0 < p <= n]
    # nxt[k] = k 之后第一个"必须断"的位置：断条不得跨过它
    nxt = [n] * (n + 1)
    stop = n
    for k in range(n - 1, -1, -1):
        nxt[k] = stop
        if k in forced:
            stop = k

    prefix = _prefix_widths(items)
    capacity = opts.max_width * opts.max_lines
    best: dict[int, float] = {0: 0.0}
    prev: dict[int, int] = {}
    for p in positions:
        if p == 0:
            continue
        for a in reversed(positions):
            if a >= p:
                continue
            if a not in best:
                continue
            if nxt[a] < p:
                break                          # 跨过必须断的位置：更早的也不行
            if prefix[p] - prefix[a] > capacity:
                break                          # 宽度单调增长：更早的只会更宽
            duration = items[p - 1][2] - items[a][1]
            if duration > opts.max_block_seconds:
                break
            cue = items[a:p]
            # 容量按"该条渲染文本"的解析判定（与折行、_can_join 同一份口径）
            parsed = _text_items(_join_atoms([it[0] for it in cue]))
            if not _fits(parsed, opts):
                continue
            cost = (best[a] + _cue_cost(parsed, duration, opts, scale)
                    + _break_cost(items, p, opts, scale, preferred))
            if p not in best or cost < best[p]:
                best[p] = cost
                prev[p] = a
    if n not in best:                             # 容量设置异常：保守按容量切
        return _fallback_cues(items, opts)
    out: list[list[tuple]] = []
    cur = n
    while cur > 0:
        a = prev[cur]
        out.append(items[a:cur])
        cur = a
    out.reverse()
    return out


def _timed_items(text: str, start: float, end: float) -> list[tuple]:
    """段级路径：文本条目按显示宽度比例切分该段时间（无词级时间可用）。"""
    items = _text_items(text)
    if not items:
        return []
    widths = [max(_text_width(it[0]), 1) for it in items]
    total = sum(widths)
    span = max(end - start, 0.0)
    out: list[tuple] = []
    cursor = start
    for it, width in zip(items, widths):
        step = span * width / total
        out.append((it[0], cursor, min(cursor + step, end), it[3], it[4]))
        cursor += step
    if out:
        last = out[-1]
        out[-1] = (last[0], last[1], max(last[2], end), last[3], last[4])
    return out


def _render_lines(text: str, opts: SrtOptions) -> list[str]:
    """把一条字幕折成若干行：换行点落在从句（标点）边界上。

    折行与断条共用同一套装箱逻辑（_layout）：从句整体装进一行，装不下才
    换行；单从句比一行还长时在该从句内部按条目边界硬切。不做行数截断：
    上游（_dp_cues）已按每屏行数切好，这里若仍多出
    行数（单个词条本身就超宽等），一并输出而不是裁掉末行——裁掉末行等于
    丢字幕文本。
    """
    return [_join_atoms([it[0] for it in line])
            for line in _layout(_text_items(text), opts)]


# ── 时间轴 → Cue ────────────────────────────────────────────

def _cues_from_segments(segments: list[tuple[float, float]],
                        texts: list[str],
                        opts: SrtOptions) -> list[Cue]:
    """VAD 段级时间轴：每段条目按宽度比例分配时长，再走与词级同一套断条。"""
    cues: list[Cue] = []
    for (start, end), text in zip(segments, texts):
        text = (text or "").strip()
        if not text or end <= start:
            continue
        for group in _dp_cues(_timed_items(text, start, end), opts):
            _flush(cues, group)
    return cues


def _cues_from_words(words: list[tuple[str, float, float]],
                    breaks: list[tuple[bool, bool, str]],
                    opts: SrtOptions,
                    hint: list[int] | None = None) -> list[Cue]:
    """词级时间轴：整篇取代价最小的断条（断点落在标点上，见 _dp_cues）。

    breaks[i] = (句内标点在前, 句末标点在前, 该词条的转写片段)，来自带标点转写
    文本与词条序列的对齐（见 _align_tokens）；标点不可用时标志全为 False、
    片段为空串，全篇退化为一个从句，只按容量、停顿与最长秒数切分。
    hint = 小模型给的"每组含几个从句"（可选，见 _preferred_cuts）。
    """
    cues: list[Cue] = []
    for group in _dp_cues(_items_from_words(words, breaks), opts, hint):
        _flush(cues, group)
    return cues


# ── 小 LLM 辅助（可选；见 src/segment_llm.py）────────────────

def _has_punct(text: str) -> bool:
    """该段文本是否已有断句标点。"""
    return any(_is_break_punct(ch) for ch in (text or ""))


def _llm_mode(kwargs: dict) -> str:
    """本次运行的断句辅助形态（kwargs → SRT_LLM_MODE；"" 表示按 config）。"""
    value = kwargs.pop("llm_mode", None)
    return str(value).strip() if value not in (None, "") else ""


def _llm_punct_text(text: str, logger: logging.Logger, mode: str) -> str:
    """B1（词级路径）：转写完全没有标点时用模型补标点，再交给规则断条。"""
    if not segment_llm.enabled("punct", mode) or _has_punct(text):
        return text
    return segment_llm.restore_punctuation(text, logger, multi_line=False)


def _llm_punct_texts(texts: list[str], logger: logging.Logger,
                     mode: str) -> list[str]:
    """B1（段级路径）：整批文本都没有标点时补标点（多段一次请求，行数须一致）。

    SenseVoice 关掉 ITN（或转写本身无标点）时文本是一整串，规则断条只能按
    容量硬切；补上标点后与词级路径同规则。只有一段时退回单行模式。
    """
    body = [t for t in texts if (t or "").strip()]
    if not segment_llm.enabled("punct", mode) or not body:
        return texts
    if any(_has_punct(t) for t in body):
        return texts
    if len(body) == 1:
        fixed = segment_llm.restore_punctuation(body[0], logger,
                                                multi_line=False)
        if not fixed or fixed == body[0]:
            return texts
        first = next(i for i, t in enumerate(texts) if (t or "").strip())
        return [fixed if i == first else t for i, t in enumerate(texts)]
    joined = "\n".join(t.strip() for t in texts)
    fixed = segment_llm.restore_punctuation(joined, logger, multi_line=True)
    lines = [ln.strip() for ln in fixed.splitlines() if ln.strip()]
    return lines if len(lines) == len(texts) else texts


def _llm_group_hint(items: list, opts: SrtOptions, logger: logging.Logger,
                    mode: str) -> list[int] | None:
    """B2（词级路径）：让小模型判断每条字幕该含哪几个从句（失败返回 None）。"""
    if not segment_llm.enabled("breaks", mode):
        return None
    clauses = [_join_atoms([it[0] for it in unit_items])
               for unit_items, _kind in _units(items, opts)]
    if len(clauses) < 2:
        return None
    groups = segment_llm.choose_groups(clauses, logger)
    if groups:
        logger.info("断句辅助：%d 个从句 → %d 条字幕分组", len(clauses), len(groups))
    return groups


def _ends_sentence(text: str) -> bool:
    """该条文字是否停在句末标点上。"""
    stripped = _strip_tail(text)
    return bool(stripped) and stripped[-1] in _SENTENCE_PUNCT


def _merge_short_cues(cues: list[Cue], opts: SrtOptions) -> list[Cue]:
    """把过短的碎条尽量并入相邻条目（优先并入上一条，放不下再并入下一条）。

    断条已按从句（标点之间的整段）成条，碎条只会因句间长停顿或每屏容量这
    两个原因偶尔出现；这里统一收口：合并后仍要满足每屏容量（按装箱行数），
    且两条之间的停顿不超过 max_gap_seconds（长停顿处的断条不硬并）。并入后
    时间取两者的并集，交给 _fix_times 收尾。

    上一条已停在句末标点时改优先并入下一条：把 "后来，" 这类引出下句的短条
    挂到它引出的那句上，比接在上一句末尾更自然。
    """
    if opts.min_cue_width <= 0:
        return cues
    work = [Cue(start=c.start, end=c.end, text=c.text) for c in cues]
    i = 0
    while i < len(work):
        cue = work[i]
        if _text_width(cue.text) >= opts.min_cue_width:
            i += 1
            continue
        prev = work[i - 1] if i > 0 else None
        nxt = work[i + 1] if i + 1 < len(work) else None
        order = ("nxt", "prev")
        if prev is None or not _ends_sentence(prev.text):
            order = ("prev", "nxt")
        merged = False
        for side in order:
            other = prev if side == "prev" else nxt
            if other is None:
                continue
            if side == "prev":
                if (cue.start - other.end) > opts.max_gap_seconds:
                    continue
                if not _can_join(other.text, cue.text, opts):
                    continue
                other.text = _join_atoms([other.text, cue.text])
                other.start = min(other.start, cue.start)
                other.end = max(other.end, cue.end)
            else:
                if (other.start - cue.end) > opts.max_gap_seconds:
                    continue
                if not _can_join(cue.text, other.text, opts):
                    continue
                other.text = _join_atoms([cue.text, other.text])
                other.start = min(other.start, cue.start)
                other.end = max(other.end, cue.end)
            work.pop(i)
            merged = True
            break
        if merged:
            i = max(i - 1, 0)       # 合并后前一条可能仍偏短，回头再看
            continue
        i += 1
    return work


def _fix_times(cues: list[Cue], opts: SrtOptions,
               duration: float = 0.0) -> list[Cue]:
    """时间轴收尾：排序、去掉空条、补足最短显示时长、避免相邻重叠。"""
    ordered = sorted((c for c in cues if (c.text or "").strip()),
                     key=lambda c: c.start)
    fixed: list[Cue] = []
    for i, cue in enumerate(ordered):
        start = max(cue.start, 0.0)
        end = max(cue.end, start + 0.2)
        if opts.min_block_seconds > 0:
            end = max(end, start + opts.min_block_seconds)
        if i + 1 < len(ordered) and ordered[i + 1].start > start:
            end = min(end, ordered[i + 1].start)
        if duration > 0:
            end = min(end, max(duration, start + 0.2))
        if end <= start:
            end = start + 0.05
        fixed.append(Cue(start=start, end=end, text=cue.text.strip()))
    return fixed


# ── SRT 渲染 ────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _output_text(text: str, opts: SrtOptions) -> str:
    """按输出开关整理字幕文本：顿号转空格；关闭标点输出时删掉其余标点。

    标点只影响输出文本，断条/断行仍按原始标点判断；空白统一压缩为单个空格
    （顿号转空格后可能产生连续空格）。关标点时会保留词内符号（I've、4.5、
    twenty-two），只去掉断句标点与独立标点。
    """
    import unicodedata

    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == "、":
            if opts.enum_comma_space:
                out.append(" ")
            elif opts.punctuation:
                out.append(ch)
            continue
        if not opts.punctuation and unicodedata.category(ch).startswith("P"):
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if _is_break_punct(ch) or not (prev.isalnum() and nxt.isalnum()):
                continue
        out.append(ch)
    return " ".join("".join(out).split())


def _render_srt(cues: list[Cue], opts: SrtOptions) -> str:
    """渲染 SRT 文本：序号 + 时间轴 + 折行后的字幕文本。

    标点处理放在折行之后：顿号转成的空格要保留（若先处理再折行，原子拼接
    会把中文之间的空格吃掉），关标点则只影响输出文本。
    """
    blocks: list[str] = []
    index = 0
    for cue in cues:
        lines = [_output_text(ln, opts)
                 for ln in _render_lines(cue.text, opts)]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        index += 1
        blocks.append("%d\n%s --> %s\n%s"
                      % (index, _fmt_time(cue.start), _fmt_time(cue.end),
                         "\n".join(lines)))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _unique_srt_path(out_dir: str, out_name: str) -> str:
    """<out_dir>/<out_name>.<unix秒>.srt；同秒冲突则秒数递增（与 pipeline 同规则）。"""
    ts = int(time.time())
    while True:
        path = os.path.join(out_dir, f"{out_name}.{ts}.srt")
        if not os.path.exists(path):
            return path
        ts += 1


# ── 引擎调用：VAD 分段 / 批量转写 / 词级对齐 ────────────────

def _vad_segments(binary: str, wav16: str, cfg: Config, logger: logging.Logger,
                  merge_gap: float, min_speech: float) -> list[tuple[float, float]]:
    """silero VAD 分段：返回 [start_sec, end_sec]（已按间隙合并、丢弃过短段）。"""
    model = SRT_VAD_MODEL or os.path.join(_src_dir(), _VAD_MODEL_REL)
    ensure_tmp_dir()
    fd, seg_json = tempfile.mkstemp(suffix=".json", prefix="vad-seg-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "vad", "--family", _VAD_FAMILY,
               "--model", model, "--backend", _backend_flag(cfg.device),
               "--audio", wav16, "--segments-out", seg_json]
        logger.info("VAD 分段中（silero_vad）…")
        run_cli(cmd, cfg.device, logger, cwd=_src_dir())
        raw = []
        # 全程静音时引擎可能写出空文件（甚至不写），按"无语音段"处理
        if os.path.exists(seg_json) and os.path.getsize(seg_json) > 0:
            with open(seg_json, encoding="utf-8") as f:
                raw = json.load(f) or []
    finally:
        if os.path.exists(seg_json):
            try:
                os.remove(seg_json)
            except OSError:
                pass

    spans = sorted((float(s["start_sample"]) / _SR,
                    float(s["end_sample"]) / _SR) for s in raw)
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    kept = [(s, e) for s, e in merged if (e - s) >= min_speech]
    logger.info("VAD 分段: %d 段（原始 %d 段，合并间隙 %.2fs，最短 %.2fs）",
                len(kept), len(spans), merge_gap, min_speech)
    return kept


def _parse_batch_texts(stdout: str) -> dict[str, str]:
    """解析批量 ASR stdout 的 `request_id=` / `text_output=` 配对。"""
    mapping: dict[str, str] = {}
    current: str | None = None
    for line in stdout.splitlines():
        if line.startswith("request_id="):
            current = line[len("request_id="):].strip()
        elif line.startswith("text_output=") and current is not None:
            mapping[current] = line[len("text_output="):].strip()
    return mapping


def _sensevoice_texts(binary: str, wav16: str, spans: list[tuple[float, float]],
                      cfg: Config, logger: logging.Logger,
                      itn: bool | None) -> list[str]:
    """逐段切 wav → sense_asr 批量转写（模型只加载一次），返回与 spans 等长的文本。"""
    import shutil

    import soundfile as sf

    ensure_tmp_dir()
    seg_dir = tempfile.mkdtemp(prefix="srt-seg-", dir=TMP_DIR)
    try:
        data, sr = sf.read(wav16, dtype="float32", always_2d=False)
        ids: list[str | None] = []
        for i, (start, end) in enumerate(spans):
            chunk = data[int(start * sr):int(end * sr)]
            if len(chunk) == 0:
                ids.append(None)
                continue
            name = f"seg{i:05d}"
            sf.write(os.path.join(seg_dir, name + ".wav"), chunk, sr)
            ids.append(name)

        model = _ensure_sensevoice(logger)
        if cfg.asr_model and os.path.isfile(cfg.asr_model):
            model = os.path.abspath(cfg.asr_model)
        cmd = [binary, "--task", "asr", "--family", "sense_asr",
               "--model", model, "--backend", _backend_flag(cfg.device),
               "--batch-audio-dir", seg_dir]
        if itn is not None:
            cmd += ["--request-option", "enable_itn=%s"
                    % ("true" if itn else "false")]
        if cfg.language:
            cmd += ["--language", str(cfg.language)]
        logger.info("转写 %d 个语音段（SenseVoice 批量）…",
                    sum(1 for x in ids if x))
        out = run_cli(cmd, cfg.device, logger, cwd=_src_dir())
        mapping = _parse_batch_texts(out)
        return [mapping.get(name, "") if name else "" for name in ids]
    finally:
        shutil.rmtree(seg_dir, ignore_errors=True)


def _local_for(file_in_repo: str) -> str:
    """工程 models/ 下与 HF 仓库同构的手工放置路径。"""
    return os.path.join(MODELS_DIR, os.path.basename(os.path.dirname(file_in_repo)),
                        os.path.basename(file_in_repo))


def _ensure_qwen3_model(repo: str, file_in_repo: str, local_override: str,
                        logger: logging.Logger, what: str) -> str:
    """Qwen3 权重定位：手工放置优先，否则经 HF 下载并留在默认缓存。"""
    local = local_override or _local_for(file_in_repo)
    if os.path.isfile(local):
        return local
    logger.info("定位 %s 权重: %s（缺失自动经 HF 下载到默认缓存）",
                what, file_in_repo)
    return _ensure_gguf_file(repo, file_in_repo, logger)


def _read_text_file(path: str) -> str:
    """读取引擎写出的文本文件（不存在或为空时返回空串）。"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def _qwen3_words(binary: str, wav16: str, cfg: Config, logger: logging.Logger,
                 hotwords: str = ""
                 ) -> tuple[list[tuple[str, float, float]], str]:
    """Qwen3-ASR + ForcedAligner：返回 (词级 (词, 起秒, 止秒), 带标点转写文本)。

    `--words-out` 的词条不含标点，标点只来自 `--text-out`（由
    SRT_QWEN3_PUNCTUATION 控制引擎 request-option preserve_punctuation）。

    hotwords 非空时经引擎 `--text` 作为 Qwen3-ASR 的系统提示词注入（引擎侧
    src/models/qwen3_asr/session.cpp 把它当 request.context 拼进 chat 模板），
    用于专有名词/术语纠偏；引擎不做词表强制匹配，成句的提示更稳。
    """
    asr = _ensure_qwen3_model(SRT_QWEN3_ASR_REPO, SRT_QWEN3_ASR_FILE,
                              SRT_QWEN3_ASR_LOCAL, logger, "Qwen3-ASR")
    aligner = _ensure_qwen3_model(SRT_QWEN3_ALIGNER_REPO, SRT_QWEN3_ALIGNER_FILE,
                                  SRT_QWEN3_ALIGNER_LOCAL, logger,
                                  "Qwen3-ForcedAligner")
    ensure_tmp_dir()
    fd, words_json = tempfile.mkstemp(suffix=".json", prefix="words-", dir=TMP_DIR)
    os.close(fd)
    fd, text_out = tempfile.mkstemp(suffix=".txt", prefix="asr-text-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "asr", "--family", "qwen3_asr",
               "--model", asr, "--backend", _backend_flag(cfg.device),
               "--audio", wav16, "--words-out", words_json,
               "--text-out", text_out,
               "--session-option",
               f"qwen3_asr.forced_aligner_model_path={aligner}"]
        if SRT_QWEN3_PUNCTUATION:
            cmd += ["--request-option", "qwen3_asr.preserve_punctuation=true"]
        if cfg.language:
            cmd += ["--language", str(cfg.language)]
        if hotwords:
            # 引擎把 --text 当 Qwen3-ASR 的系统提示词（request.context）
            cmd += ["--text", _hotwords_prompt(hotwords)]
            logger.info("ASR 热词/上下文已注入（%d 字）：%s",
                        len(hotwords), hotwords)
        logger.info("Qwen3-ASR 转写 + 强制对齐（词级时间戳）…")
        run_cli(cmd, cfg.device, logger, cwd=_src_dir())
        raw = []
        # 全程静音等情况下引擎可能写出空文件（甚至不写），按"无词条"处理
        if os.path.exists(words_json) and os.path.getsize(words_json) > 0:
            with open(words_json, encoding="utf-8") as f:
                raw = json.load(f) or []
        transcript = _read_text_file(text_out)
    finally:
        for path in (words_json, text_out):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    words = [(str(w.get("word", "")),
              float(w.get("start_sample", 0)) / _SR,
              float(w.get("end_sample", 0)) / _SR) for w in raw]
    logger.info("对齐得到 %d 个词", len(words))
    return words, transcript


# ── 对外入口 ────────────────────────────────────────────────

def subtitles(cfg: Config, logger: logging.Logger, **kwargs) -> SrtResult:
    """音频 → SRT 字幕。

    kwargs：
      audio / out_dir / out_name（数据参数）；
      asr_backend（qwen3_asr | sensevoice；留空用 cfg.srt_asr / SRT_ASR）；
      language（留空用 cfg.language，再留空 = 模型自动）；
      itn（SenseVoice 反向文本规范化；留空用 SRT_ITN）；
      hotwords（ASR 热词/上下文，**仅 qwen3_asr 生效**：经引擎 `--text` 作为
        系统提示词注入；留空用 cfg.srt_hotwords / SRT_HOTWORDS / hotword.txt）；
      punctuation（字幕文本是否输出标点；留空用 SRT_PUNCTUATION）、
      enum_comma_space（顿号转空格；留空用 SRT_ENUM_COMMA_AS_SPACE）、
      break_on_comma（逗号处即收条；留空用 SRT_BREAK_ON_COMMA）、
      llm_mode（小 LLM 辅助断句形态 punct | breaks | both | off；留空用
        SRT_LLM / SRT_LLM_MODE，默认关闭；模型不产生时间轴，输出校验不过即
        退回规则结果）、
      min_cue_width（碎条阈值宽度；留空用 SRT_MIN_CUE_WIDTH，0 = 关闭）、
      max_line_width / max_lines / max_block_seconds / max_gap_seconds /
      min_block_seconds / cps_max / min_line_width / line_balance
      （排版与时间轴；留空用 config.py 顶部常量）。

    引擎调用与排版分离：本函数只负责分段、汇总与写盘，推理在引擎子进程内。
    """
    audio = kwargs.pop("audio", None) or cfg.srt_audio
    out_dir = kwargs.pop("out_dir", None) or TMP_DIR
    out_name = kwargs.pop("out_name", None) or "subtitle"
    backend = str(kwargs.pop("asr_backend", None) or cfg.srt_asr
                  or SRT_ASR).strip().lower()
    language = kwargs.pop("language", None) or cfg.language or None
    itn = kwargs.pop("itn", None)
    itn = SRT_ITN if itn is None or itn == "" else bool(itn)
    # 热词/上下文：kwargs → cfg.srt_hotwords → SRT_HOTWORDS → hotword.txt
    hotwords = _hotwords_text(kwargs.pop("hotwords", None), cfg)
    llm_mode = _llm_mode(kwargs)
    opts = _options(kwargs)

    if not audio or not os.path.isfile(audio):
        raise ValueError("请提供有效的音频文件（wav）")

    binary = _ensure_binary(logger)
    ensure_tmp_dir()
    fd, wav16 = tempfile.mkstemp(suffix=".wav", prefix="srt-16k-", dir=TMP_DIR)
    os.close(fd)
    try:
        logger.info("音频重采样 16 kHz mono …")
        _to_16k_mono(audio, wav16)
        import soundfile as sf
        duration = sf.info(wav16).frames / float(_SR)

        if backend in ("qwen3", "qwen3_asr"):
            words, transcript = _qwen3_words(binary, wav16, cfg, logger,
                                             hotwords)
            if transcript:
                logger.info("转写文本: %s", transcript)
            transcript = _llm_punct_text(transcript, logger, llm_mode)
            breaks = _align_tokens(transcript, [w[0] for w in words])
            items = _items_from_words(words, breaks)
            hint = _llm_group_hint(items, opts, logger, llm_mode)
            cues = _cues_from_words(words, breaks, opts, hint)
            backend = "qwen3_asr"
        else:
            if hotwords:
                logger.warning(
                    "SenseVoice（sense_asr 族）没有上下文接口，热词已被忽略"
                    "（热词仅 qwen3_asr 路径生效）")
            spans = _vad_segments(binary, wav16, cfg, logger,
                                  float(_pick(kwargs, "vad_merge_gap",
                                              SRT_VAD_MERGE_GAP)),
                                  float(_pick(kwargs, "vad_min_speech",
                                              SRT_VAD_MIN_SPEECH)))
            if not spans:
                raise RuntimeError(
                    "未检测到语音段（音频可能为静音，或最短语音段阈值过大）")
            texts = _sensevoice_texts(binary, wav16, spans, cfg, logger, itn)
            texts = _llm_punct_texts(texts, logger, llm_mode)
            cues = _cues_from_segments(spans, texts, opts)
            backend = "sensevoice"

        cues = _fix_times(_merge_short_cues(cues, opts), opts, duration)
        srt_text = _render_srt(cues, opts)
        if not cues:
            raise RuntimeError("未识别到语音内容（音频可能为静音）")

        os.makedirs(out_dir, exist_ok=True)
        srt_path = _unique_srt_path(out_dir, out_name)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        logger.info("字幕完成：%d 条（%.1fs 音频，ASR=%s）→ %s",
                    len(cues), duration, backend, srt_path)
        return SrtResult(srt_path=srt_path, srt_text=srt_text, cues=cues,
                         duration_sec=duration, asr_backend=backend)
    finally:
        if os.path.exists(wav16):
            try:
                os.remove(wav16)
            except OSError:
                pass
