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

- 断条优先标点：句末标点（。！？）处成句即断；句内标点（，、；：）作为超限时的
  回退断点，避免把词组从中间切开。标点不可用（未开启或对齐失败）时退回
  停顿 / 单条最长秒数 / 每屏行数 三个约束——词级路径用词间时距，段级路径
  按字符宽度比例拆分该段时长。
- 断行按显示宽度（CJK=2、其余=1），折行点同样优先标点；中文之间不插空格，
  拉丁词之间保留空格。

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
    SRT_BLOCK_EXTEND_SECONDS,
    SRT_ENUM_COMMA_AS_SPACE,
    SRT_ITN,
    SRT_MAX_BLOCK_SECONDS,
    SRT_MAX_GAP_SECONDS,
    SRT_MAX_LINES,
    SRT_MAX_LINE_WIDTH,
    SRT_MIN_BLOCK_SECONDS,
    SRT_MIN_CUE_WIDTH,
    SRT_PUNCTUATION,
    SRT_QWEN3_ALIGNER_FILE,
    SRT_QWEN3_ALIGNER_LOCAL,
    SRT_QWEN3_ALIGNER_REPO,
    SRT_QWEN3_ASR_FILE,
    SRT_QWEN3_ASR_LOCAL,
    SRT_QWEN3_ASR_REPO,
    SRT_QWEN3_PUNCTUATION,
    SRT_SENTENCE_BREAK_RATIO,
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
    sentence_break_ratio: float
    block_extend_seconds: float
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
        sentence_break_ratio=float(
            _pick(kwargs, "sentence_break_ratio", SRT_SENTENCE_BREAK_RATIO)),
        block_extend_seconds=float(
            _pick(kwargs, "block_extend_seconds", SRT_BLOCK_EXTEND_SECONDS)),
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

# 断句标点：句末（成句即可断）与句内（超限时优先在此断）；数字里的 "."
# 由 _punct_kind 另行排除（4.5 / 1,000 / twenty-two 不算断句）
_SENTENCE_PUNCT = "。！？!?…."
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


def _wrap(atoms: list[str], max_width: int) -> list[str]:
    """按宽度贪心折行（不超 max_width），返回行列表。"""
    lines: list[str] = []
    line = ""
    width = 0
    for a in atoms:
        aw = _text_width(a)
        sep = 1 if (line and _needs_space(line, a)) else 0
        if line and width + sep + aw > max_width:
            lines.append(line)
            line, width = a, aw
            continue
        line = f"{line} {a}" if sep else f"{line}{a}"
        width += sep + aw
    if line:
        lines.append(line)
    return lines


def _preferred_cut(items: list) -> int:
    """超限时的断点下标：优先最近的句末标点，其次最近的句内标点，否则到末尾。

    items 元素取第 4/5 位作为 "该元素之前有句内/句末标点" 的布尔标记
    （词级条目 = (词, 起, 止, 句内, 句末)，文本条目 = (原子, 0, 0, 句内, 句末)）。
    返回值为下一段的起点：items[:cut] 归上一条，items[cut:] 归下一条。
    """
    for want_sentence in (True, False):
        for j in range(len(items) - 1, 0, -1):
            marked = items[j][4] if want_sentence else items[j][3]
            if marked:
                return j
    return len(items)


def _text_of(items: list) -> str:
    """条目序列的文字（原子按需插空格）。"""
    return _join_atoms([it[0] for it in items])


def _is_sliver(items: list, opts: SrtOptions) -> bool:
    """该段文字是否过短（碎条），不宜单独成为一条字幕。"""
    if opts.min_cue_width <= 0 or not items:
        return False
    return _text_width(_text_of(items)) < opts.min_cue_width


def _can_join(a: str, b: str, opts: SrtOptions) -> bool:
    """两段文字合并后是否仍在每屏容量内（宽度近似判断）。"""
    return (_text_width(_join_atoms([a, b]))
            <= opts.max_width * opts.max_lines)


def _over_limit(items: list, opts: SrtOptions) -> bool:
    """该条是否已超限：单条最长秒数，或折行后超过每屏行数。"""
    if len(items) < 2:
        return False
    if (items[-1][2] - items[0][1]) > opts.max_block_seconds:
        return True
    return len(_wrap([it[0] for it in items], opts.max_width)) > opts.max_lines


def _extend_to_punct(words: list[tuple[str, float, float]],
                     breaks: list, idx: int, buf: list,
                     opts: SrtOptions) -> bool:
    """超限但句内无标点可退时：若最近的标点边界就在预算内（时间 + 宽度），
    再多收几个词，让字幕停在自然句读处，而不是切出 "表达。" 这类尾巴。

    边界有两种位置：下一个词之前有标点（flag[0]/flag[1]，本词归下一条），
    或本词自己带尾随标点（flag[2]，本词仍归本条）——后者对应 "…表达。" 这种
    半句结尾，正是要避免切掉的情形。
    """
    if opts.block_extend_seconds <= 0:
        return False
    deadline = buf[-1][2] + opts.block_extend_seconds
    width = _text_width(_join_atoms([b[0] for b in buf]))
    for j in range(idx + 1, len(words)):
        atom, start, end = words[j]
        flag = breaks[j] if j < len(breaks) else (False, False, "")
        piece = flag[2] or atom
        tail_punct = any(_is_break_punct(c) for c in piece)
        boundary = end if tail_punct else start
        if boundary > deadline:
            return False
        width += _text_width(piece)
        if width > opts.max_width * opts.max_lines:   # 会撑破每屏行数预算
            return False
        if flag[0] or flag[1] or tail_punct:          # 预算内遇到标点边界
            return True
    return False


def _flush(cues: list[Cue], items: list, cut: int) -> None:
    """把 items[:cut] 收成一条字幕（时间取首尾元素）。"""
    part = items[:cut]
    if not part:
        return
    text = _join_atoms([p[0] for p in part])
    if not text:
        return
    cues.append(Cue(start=part[0][1], end=part[-1][2], text=text))


def _split_text(text: str, max_width: int, max_lines: int) -> list[str]:
    """把文本切成若干块：每块折行后不超过 max_lines 行（供拆分时间轴用）。

    切点优先句内/句末标点，避免把词组从中间切开；以折行结果判定容量，
    避免词较长时折行数超出上限、末行被裁掉。
    """
    items = _text_items(text)
    blocks: list[str] = []
    cur: list = []
    for it in items:
        cur.append(it)
        if (len(cur) > 1
                and len(_wrap([c[0] for c in cur], max_width)) > max_lines):
            cut = _preferred_cut(cur)
            if cut < 1 or cut >= len(cur):
                cut = len(cur) - 1
            blocks.append(_join_atoms([c[0] for c in cur[:cut]]))
            cur = cur[cut:]
    if cur:
        blocks.append(_join_atoms([c[0] for c in cur]))
    return [b for b in blocks if b]


def _render_lines(text: str, max_width: int, max_lines: int) -> list[str]:
    """把一条字幕折成若干行；折行点优先标点。

    不做行数截断：上游（_split_text / _cues_from_words）已按每屏行数切好，
    这里若仍多出行数（单个词条本身就超宽等），一并输出而不是裁掉末行——
    裁掉末行等于丢字幕文本。max_lines 仅用于说明期望值。
    """
    items = _text_items(text)
    lines: list[str] = []
    cur: list = []
    for it in items:
        cur.append(it)
        if len(cur) > 1 and len(_wrap([c[0] for c in cur], max_width)) > 1:
            cut = _preferred_cut(cur)
            if cut < 1 or cut >= len(cur):
                cut = len(cur) - 1
            lines.append(_join_atoms([c[0] for c in cur[:cut]]))
            cur = cur[cut:]
    if cur:
        lines.append(_join_atoms([c[0] for c in cur]))
    return [ln for ln in lines if ln]


# ── 时间轴 → Cue ────────────────────────────────────────────

def _cues_from_segments(segments: list[tuple[float, float]],
                        texts: list[str],
                        opts: SrtOptions) -> list[Cue]:
    """VAD 段级时间轴：每段文本超出排版容量时按字符宽度比例拆分该段时长。"""
    cues: list[Cue] = []
    for (start, end), text in zip(segments, texts):
        text = (text or "").strip()
        duration = max(end - start, 0.0)
        if not text or duration <= 0:
            continue
        blocks = _split_text(text, opts.max_width, opts.max_lines)
        if not blocks:
            continue
        weights = [max(_text_width(b), 1) for b in blocks]
        total = sum(weights)
        cursor = start
        for block, weight in zip(blocks, weights):
            block_end = cursor + duration * (weight / total)
            cues.append(Cue(start=cursor, end=block_end, text=block))
            cursor = block_end
        cues[-1].end = end
    return cues


def _cues_from_words(words: list[tuple[str, float, float]],
                    breaks: list[tuple[bool, bool]],
                    opts: SrtOptions) -> list[Cue]:
    """词级时间轴：标点优先断句，其次按停顿/单条最长秒数/每屏行数断条。

    breaks[i] = (句内标点在前, 句末标点在前, 该词条的转写片段)，来自带标点转写
    文本与词条序列的对齐（见 _align_tokens）；标点不可用时标志全为 False、
    片段为空串，退化为纯时距/长度断条。
    """
    cues: list[Cue] = []
    buf: list[tuple] = []
    min_sentence_width = max(int(opts.max_width * opts.sentence_break_ratio), 1)
    for idx, (atom, start, end) in enumerate(words):
        if idx < len(breaks):
            clause, sentence, piece = breaks[idx]
        else:
            clause, sentence, piece = False, False, ""
        if buf:
            if (start - buf[-1][2]) > opts.max_gap_seconds:
                _flush(cues, buf, len(buf))          # 自然停顿：直接断
                buf = []
            elif (sentence and _text_width(_join_atoms([b[0] for b in buf]))
                    >= min_sentence_width):
                _flush(cues, buf, len(buf))          # 句末标点：成句即断
                buf = []
        buf.append((piece or atom, start, end, clause, sentence))
        if _over_limit(buf, opts):
            cut = _preferred_cut(buf)
            if cut == len(buf) and _extend_to_punct(words, breaks, idx, buf,
                                                    opts):
                continue          # 句内无标点：顺延到最近标点处再断
            # 回退点要同时满足两点，否则继续往前退：
            # 1) 上一条不超行数/秒数预算；2) 不给下一条留一两字的碎尾
            while cut > 1 and (_over_limit(buf[:cut], opts)
                               or _is_sliver(buf[cut:], opts)):
                inner = _preferred_cut(buf[:cut])
                cut = inner if inner < cut else cut - 1
            _flush(cues, buf, cut)
            buf = buf[cut:]
    _flush(cues, buf, len(buf))
    return cues


def _merge_short_cues(cues: list[Cue], opts: SrtOptions) -> list[Cue]:
    """把过短的碎条尽量并入相邻条目（优先并入上一条，放不下再并入下一条）。

    词级切条时已经尽量不留碎尾，但自然停顿/超时断条仍可能切出 "间"、"轮"
    这类一两字的孤条；这里统一收口：合并后仍要满足每屏容量（宽度），且两条
    之间的停顿不超过 max_gap_seconds（长停顿处的断条不硬并）。并入后时间取
    两者的并集，交给 _fix_times 收尾。
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
        if (prev is not None
                and (cue.start - prev.end) <= opts.max_gap_seconds
                and _can_join(prev.text, cue.text, opts)):
            prev.text = _join_atoms([prev.text, cue.text])
            prev.start = min(prev.start, cue.start)
            prev.end = max(prev.end, cue.end)
            work.pop(i)
            i = max(i - 1, 0)       # 合并后前一条可能仍偏短，回头再看
            continue
        if (nxt is not None
                and (nxt.start - cue.end) <= opts.max_gap_seconds
                and _can_join(cue.text, nxt.text, opts)):
            nxt.text = _join_atoms([cue.text, nxt.text])
            nxt.start = min(nxt.start, cue.start)
            nxt.end = max(nxt.end, cue.end)
            work.pop(i)
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
                 for ln in _render_lines(cue.text, opts.max_width,
                                         opts.max_lines)]
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


def _qwen3_words(binary: str, wav16: str, cfg: Config, logger: logging.Logger
                 ) -> tuple[list[tuple[str, float, float]], str]:
    """Qwen3-ASR + ForcedAligner：返回 (词级 (词, 起秒, 止秒), 带标点转写文本)。

    `--words-out` 的词条不含标点，标点只来自 `--text-out`（由
    SRT_QWEN3_PUNCTUATION 控制引擎 request-option preserve_punctuation）。
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
      punctuation（字幕文本是否输出标点；留空用 SRT_PUNCTUATION）、
      enum_comma_space（顿号转空格；留空用 SRT_ENUM_COMMA_AS_SPACE）、
      min_cue_width（碎条阈值宽度；留空用 SRT_MIN_CUE_WIDTH，0 = 关闭）、
      max_line_width / max_lines / max_block_seconds / max_gap_seconds /
      min_block_seconds / sentence_break_ratio / block_extend_seconds
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
            words, transcript = _qwen3_words(binary, wav16, cfg, logger)
            if transcript:
                logger.info("转写文本: %s", transcript)
            breaks = _align_tokens(transcript, [w[0] for w in words])
            cues = _cues_from_words(words, breaks, opts)
            backend = "qwen3_asr"
        else:
            spans = _vad_segments(binary, wav16, cfg, logger,
                                  float(_pick(kwargs, "vad_merge_gap",
                                              SRT_VAD_MERGE_GAP)),
                                  float(_pick(kwargs, "vad_min_speech",
                                              SRT_VAD_MIN_SPEECH)))
            if not spans:
                raise RuntimeError(
                    "未检测到语音段（音频可能为静音，或最短语音段阈值过大）")
            texts = _sensevoice_texts(binary, wav16, spans, cfg, logger, itn)
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
