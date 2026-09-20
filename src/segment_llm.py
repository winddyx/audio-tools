"""
audio-tools — 小 LLM 辅助断句（llama.cpp 子进程，可选）

规则断条（src/subtitle.py 的代价最优断条）保证"断点落在标点上、不切词组、不超
每屏容量、不超最长秒数"，但它只看得见标点、停顿时长和宽度，判断不了语义。本
模块用本机小模型补上语义那一层，两种形态：

- B1 补停顿标记 restore_punctuation：转写没有标点（或标点丢失）时，让模型在
  停顿处插入分隔符 ｜（再映射为逗号），交给规则断条——规则完全依赖标点，
  这一层收益最直接，SenseVoice 关 ITN 时最有用；
- B2 断条分组 choose_groups：规则先按标点切出从句，模型只决定"每条字幕包含哪
  几个连续的从句"（输出每组从句数），规则再据此调整断点倾向。

三条底线（LLM 只加不减）：

- 模型不产生时间轴：时间永远来自 Qwen3-ForcedAligner / silero VAD；
- 输出严格校验，不通过就整段丢弃、退回规则结果（B1 校验"去掉标点与分隔符后
  逐字一致"，B2 校验"每组 ≥ 1 且总和等于从句数"）；
- 采样固定（温度 0、固定 seed），结果按内容哈希缓存，重跑同一素材不再调用。

引擎/模型定位与子进程调用复用 src/llamarun.py；开关与参数在 src/config.py
顶部常量（SRT_LLM*），默认关闭。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata

from .config import (
    SRT_LLM,
    SRT_LLM_BATCH_CLAUSES,
    SRT_LLM_CACHE,
    SRT_LLM_CACHE_DIR,
    SRT_LLM_DEVICE,
    SRT_LLM_FILE,
    SRT_LLM_LOCAL,
    SRT_LLM_MAX_TOKENS,
    SRT_LLM_MODE,
    SRT_LLM_PUNCT_CHARS,
    SRT_LLM_REPO,
    SRT_LLM_SEED,
    SRT_LLM_TEMPERATURE,
    SRT_LLM_TIMEOUT,
    SRT_LLM_TOP_K,
    SRT_LLM_TOP_P,
)
from .llamarun import LlmParams, model_path, run_once

# ── 提示词（小模型只做"补标点 / 分组"这两件确定性的事）────────

_PUNCT_PROMPT = """在下面的中文里插入分隔符 ｜，标在说话停顿处（哪里该换一条字幕就标哪里）。
只输出加好分隔符的那一行原文，一个字都不要改，不要解释。

原文：改是从旧变新造是从无到有十八年来峻佳设计的改造轨迹横跨了地产
结果：改是从旧变新｜造是从无到有｜十八年来｜峻佳设计的改造轨迹横跨了地产

原文：{text}
/no_think
结果："""

_PUNCT_MULTILINE_PROMPT = """在下面几行中文里各插入分隔符 ｜，标在说话停顿处（哪里该换一条
字幕就标哪里）。逐行处理：一个字都不要改，行数与原文一致，每行只输出加好分隔符
的那一行，不要解释、不要编号、不要引号。

原文：
{text}
/no_think
结果："""

# 分隔符（模型可能输出全角或半角）：映射为逗号——规则只关心"这里有断点"，
# 标点类型不影响断条与折行。
_SEPARATORS = ("｜", "|", "︱", "丨")


_GROUP_PROMPT = """你在给短视频做字幕断条。下面是口播稿按标点切成的从句，编号从 0 开始，
请判断每条字幕应该包含哪几个连续的从句：一条字幕尽量是一个完整的语意单元
（一个完整短句，或关系紧密的短句），不要把两个互不相干的句子挤在一条里，
也不要把一个完整句子拆成两条。

逐条给出从句个数，用逗号分隔，总和必须等于 {total}。

示例：
从句：[0]改是从旧变新 [1]造是从无到有 [2]十八年来 [3]峻佳设计的改造轨迹横跨了地产
分组：1,2,1

从句：
{clauses}
/no_think
分组："""

_MODE_PUNCT = "punct"
_MODE_BREAKS = "breaks"


def enabled(want: str, mode: str = "") -> bool:
    """该形态是否启用：总开关打开且有效形态包含它。

    mode 为空 → 用 SRT_LLM_MODE（config）；mode 显式给了 off/none → 关闭
    （web 的「断句辅助」下拉即用这条路径在运行期覆盖）。
    """
    if not SRT_LLM:
        return False
    active = (mode or SRT_LLM_MODE or "both").strip().lower()
    if active in ("off", "none", "0", "false"):
        return False
    if active in ("both", "all", "1", "true"):
        return True
    return want in active


def _params() -> LlmParams:
    return LlmParams(
        max_tokens=SRT_LLM_MAX_TOKENS,
        temperature=SRT_LLM_TEMPERATURE,
        top_p=SRT_LLM_TOP_P,
        top_k=SRT_LLM_TOP_K,
        repeat_penalty="1.0",
        timeout=SRT_LLM_TIMEOUT,
        device=SRT_LLM_DEVICE,
        seed=SRT_LLM_SEED,
        label="SRT 断句辅助",
    )


def _model(logger: logging.Logger) -> str:
    return model_path(SRT_LLM_REPO, SRT_LLM_FILE, SRT_LLM_LOCAL, logger,
                      desc="断句辅助")


# ── 缓存：同一素材重跑不再调用模型 ──────────────────────────

def _cache_path(kind: str, payload: str) -> str:
    key = hashlib.sha1(
        (kind + "\x1f" + SRT_LLM_REPO + "\x1f" + SRT_LLM_FILE + "\x1f"
         + payload).encode("utf-8")).hexdigest()
    return os.path.join(SRT_LLM_CACHE_DIR, key + ".json")


def _cache_get(kind: str, payload: str):
    if not SRT_LLM_CACHE:
        return None
    path = _cache_path(kind, payload)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _cache_put(kind: str, payload: str, value) -> None:
    if not SRT_LLM_CACHE or value is None:
        return
    path = _cache_path(kind, payload)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except OSError:
        pass


# ── B1 标点恢复 ────────────────────────────────────────────

def _plain(text: str) -> str:
    """去掉标点、分隔符与空白，只留正文（用于校验模型没有增删改文字）。"""
    out: list[str] = []
    for ch in text or "":
        if ch in _SEPARATORS or ch.isspace():
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        out.append(ch)
    return "".join(out)


def _with_commas(text: str) -> str:
    """把模型输出的停顿分隔符换成逗号（分隔符只用来定位断点）。"""
    out = text
    for sep in _SEPARATORS:
        out = out.replace(sep, "，")
    return out


_THINK_RE = re.compile(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|\Z)", re.S)


def _strip_think(text: str) -> str:
    """去掉思考模型的思维链块（Qwen3 等会先输出 think 段）。"""
    return _THINK_RE.sub("", text or "")


_MARKS = ("结果：", "结果:", "分组：", "分组:", "补标点：", "答案：",
          "补标点结果：", "分组结果：")


def _clean_line(line: str) -> str:
    """去掉答案行上的提示前缀、引号，以及模型常一起吐出的末尾标记。"""
    out = (line or "").strip()
    for prefix in _MARKS:
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
            break
    out = out.strip().strip('"').strip("“”").strip()
    for mark in _MARKS:
        while out.endswith(mark):
            out = out[:-len(mark)].rstrip()
    return out


def _candidates(text: str) -> list[str]:
    """模型输出里可能的答案（已去思考块/前缀/标记）：先逐行、再整体。

    多行答案（补标点按行处理）要整体作为一个候选，单行答案则是某一行的
    内容——两种都给出，由调用方按校验挑选。
    """
    body = _strip_think(text)
    lines = [ln.strip() for ln in body.splitlines()]
    per_line = [_clean_line(ln) for ln in lines if _clean_line(ln)]
    per_line.reverse()                           # 末行最可能是答案
    kept = [ln for ln in lines if ln and _clean_line(ln) not in _MARKS]
    while kept and (_clean_line(kept[-1]) in _MARKS or not _clean_line(kept[-1])):
        kept.pop()
    whole = "\n".join(_clean_line(ln) for ln in kept if _clean_line(ln))
    return per_line + ([whole] if whole else [])


def _answer(text: str) -> str:
    """取模型输出里最可能的"答案行"（末行优先）。"""
    got = _candidates(text)
    return got[0] if got else ""


def restore_punctuation(text: str, logger: logging.Logger,
                        multi_line: bool = False) -> str:
    """B1：给没有标点的转写补断点标记（模型输出 ｜ → 映射为逗号）。

    校验不过返回原文（等于没做）。
    """
    src = (text or "").strip()
    if not src:
        return text
    cached = _cache_get("punct-multi" if multi_line else "punct", src)
    if isinstance(cached, str) and cached:
        return cached
    if multi_line or len(src) <= max(int(SRT_LLM_PUNCT_CHARS), 0):
        fixed = _punct_once(src, logger, multi_line)
    else:                                       # 长串拆短再补，标点更密
        parts = _chunk(src, int(SRT_LLM_PUNCT_CHARS))
        fixed = "".join(_punct_once(p, logger, False) for p in parts)
    if not fixed or _plain(fixed) != _plain(src):
        logger.warning("断句辅助（补标点）校验不过，按原文本继续")
        return text
    _cache_put("punct-multi" if multi_line else "punct", src, fixed)
    logger.info("断句辅助：补标点完成（%d 字）", len(fixed))
    return fixed


def _chunk(text: str, limit: int) -> list[str]:
    """按字数把长串切成小块（补标点用；块边界落在标点上更好，这里只求均分）。"""
    limit = max(limit, 8)
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def _punct_once(text: str, logger: logging.Logger, multi_line: bool) -> str:
    """单次补标点请求；失败或校验不过时返回原文（等于这一段没补）。"""
    try:
        model = _model(logger)
        template = _PUNCT_MULTILINE_PROMPT if multi_line else _PUNCT_PROMPT
        out = run_once(model, template.format(text=text), logger, _params())
    except Exception as e:                      # 模型/引擎问题都不该打断出字幕
        logger.warning("断句辅助（补标点）失败，按原文本继续: %s", e)
        return text
    fixed = _accept_punct(text, out, multi_line)
    if fixed is None:
        logger.warning("断句辅助（补标点）输出校验不过，按原文本继续")
        return text
    return fixed


def _accept_punct(src: str, out: str, multi_line: bool) -> str | None:
    """校验补标点结果：去标点后逐字一致（多行还要行数一致）。

    逐候选行搜索：模型可能带解释或思考块，只有"逐字一致"的那一行才算数。
    """
    if not multi_line:
        want = _plain(src)
        for cand in _candidates(out):
            if _plain(cand) == want:
                return _with_commas(cand)
        return None
    src_lines = [ln for ln in (src or "").splitlines() if ln.strip()]
    for cand in _candidates(out):
        out_lines = [ln for ln in cand.splitlines() if ln.strip()]
        if len(out_lines) != len(src_lines):
            continue
        if all(_plain(a) == _plain(b) for a, b in zip(src_lines, out_lines)):
            return "\n".join(_with_commas(ln) for ln in out_lines)
    return None


# ── B2 断条分组 ────────────────────────────────────────────

def choose_groups(clauses: list[str], logger: logging.Logger) -> list[int] | None:
    """B2：判断每条字幕包含几个连续从句，返回与 clauses 等长的分组列表。

    失败返回 None（调用方退回规则断条）。分批请求，每批失败只影响该批
    （退回"每个从句一条"，即规则默认）。
    """
    if len(clauses) <= 1:
        return [len(clauses)] if clauses else []
    groups: list[int] = []
    batch = max(int(SRT_LLM_BATCH_CLAUSES), 1)
    for start in range(0, len(clauses), batch):
        part = clauses[start:start + batch]
        got = _group_batch(part, logger)
        groups.extend(got if got else [1] * len(part))
    return groups if sum(groups) == len(clauses) else None


def _group_batch(part: list[str], logger: logging.Logger) -> list[int] | None:
    """一批从句的分组：调用 + 校验，失败返回 None。"""
    body = "\n".join("[%d]%s" % (i, c) for i, c in enumerate(part))
    cached = _cache_get("groups", body)
    if isinstance(cached, list) and sum(cached) == len(part):
        return [int(x) for x in cached]
    try:
        model = _model(logger)
        prompt = _GROUP_PROMPT.format(total=len(part), clauses=body)
        out = run_once(model, prompt, logger, _params())
    except Exception as e:
        logger.warning("断句辅助（分组）失败，该批用规则断条: %s", e)
        return None
    sizes = None
    for cand in _candidates(out):
        sizes = _parse_groups(cand, len(part))
        if sizes is not None:
            break
    if sizes is None:
        logger.warning("断句辅助（分组）输出校验不过，该批用规则断条")
        return None
    _cache_put("groups", body, sizes)
    return sizes


def _parse_groups(answer: str, total: int) -> list[int] | None:
    """解析分组输出：接受 `2,1,3`（每组从句数）或 `1,2,5`（断点下标）。

    校验：每组 ≥ 1、总和等于 total。解析不出来返回 None。
    """
    if not answer:
        return None
    text = answer.replace("，", ",").replace("、", ",").strip()
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if not nums:
        return None
    if sum(nums) == total and all(n >= 1 for n in nums):
        return nums
    # 兼容"断点下标"形式：0 起、升序、末位不含 total
    if (all(0 <= n < total for n in nums)
            and nums == sorted(set(nums)) and len(nums) < total):
        sizes, prev = [], 0
        for idx in nums + [total]:
            sizes.append(idx - prev)
            prev = idx
        if all(s >= 1 for s in sizes):
            return sizes
    return None
