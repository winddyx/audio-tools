"""
audio-tools — 粤语文案翻译（Hy-MT2-1.8B，llama.cpp 子进程）

普通话/中文文案 → 粤语口播文案：为语音克隆提供粤语 TTS 输入。Hy-MT2 是
腾讯开源的多语言翻译模型（官方支持粤语 yue 双向互译），本模块用本机
llama-completion（llama.cpp 单次补全 CLI）子进程做推理，Python 只做编排：

- 引擎：新版 llama.cpp（ggml >= 0.10）把单次补全拆到 llama-completion。
  LLAMA_CLI 留空时自动 clone + cmake 编译 vendor/llama.cpp（与 audiocpp
  同模式，不依赖本机 brew 安装）；引擎定位/构建、模型定位与子进程调用
  统一在 src/llamarun.py（与 src/segment_llm.py 共用）；
- 模型经 HuggingFace 下载，只留在 HF 默认缓存并生成 .gguf 硬链接别名
  （src/hf._ensure_gguf_file），不落工程目录；HYMT2_LOCAL 可手工放置；
- 提示词模板：默认读取 HYMT2_PROMPT_FILE 指向的纯文本（含 {text} 占位，
  运行时被源文案替换；不含时直接拼末尾）；web 页提示词框可编辑；
- 长文案按段落分块（HYMT2_CHUNK_CHARS）逐段翻译后拼接，避免长文漏翻；
- 推理失败抛 RuntimeError（带 stderr 尾部诊断），不在入口裸奔。

web.py「粤语翻译」页调用 translate()。采样/设备等可调参数统一在
src/config.py 顶部常量（同名 env 可覆盖），本文件不再散落默认值。
"""

from __future__ import annotations

import logging
import os

from .config import (
    HYMT2_CHUNK_CHARS,
    HYMT2_DEVICE,
    HYMT2_FILE,
    HYMT2_LOCAL,
    HYMT2_MAX_TOKENS,
    HYMT2_PROMPT_FILE,
    HYMT2_REPETITION_PENALTY,
    HYMT2_REPO,
    HYMT2_TEMPERATURE,
    HYMT2_TIMEOUT,
    HYMT2_TOP_K,
    HYMT2_TOP_P,
    HYMT2_TRAD,
)
from .llamarun import LlmParams, model_path, run_once

# 粤语功能字标志集：译文若完全不含这些字，基本可判定模型没执行粤语转换
# （回显原文/繁体书面化），translate() 会自动重跑一次。
_YUE_MARKERS = "嘅咗喺哋啲係唔睇畀諗乜嘢仲同埋呢啲嗰啲呢度嗰度"


def _looks_cantonese(text: str) -> bool:
    """粗判输出是否做了粤语转换（含任意粤语功能字即认为已转换）。"""
    return any(ch in _YUE_MARKERS for ch in text)


def _to_hk_trad(text: str) -> str:
    """译文统一转香港繁体（zhconv zh-hk + zh-hant 兜底）。

    zhconv 的 zh-hk 表有个别简体漏转（如"户"），zh-hant 只处理剩余简体
    字、不会改动已转好的繁体与粤语字，故链式转换补漏。未安装/失败原样返回。
    """
    if not text or not HYMT2_TRAD:
        return text
    try:
        from zhconv import convert
        return convert(convert(text, "zh-hk"), "zh-hant")
    except Exception:
        return text


def _ensure_model(logger: logging.Logger) -> str:
    """返回可直接喂给 llama-completion 的模型 .gguf 路径（本地优先）。"""
    return model_path(HYMT2_REPO, HYMT2_FILE, HYMT2_LOCAL, logger,
                      desc="翻译")


def _run_once(model: str, prompt: str, logger: logging.Logger) -> str:
    """调用 llama-completion 完成一次翻译，返回模型输出文本。

    采样/设备参数用 config 顶部常量（默认腾讯官方推荐）；引擎定位与自动
    构建、模型定位、子进程调用都在 src/llamarun.py；失败抛 RuntimeError
    （含 stderr 尾部）。
    """
    return run_once(model, prompt, logger, LlmParams(
        max_tokens=HYMT2_MAX_TOKENS,
        temperature=HYMT2_TEMPERATURE,
        top_p=HYMT2_TOP_P,
        top_k=HYMT2_TOP_K,
        repeat_penalty=HYMT2_REPETITION_PENALTY,
        timeout=HYMT2_TIMEOUT,
        device=HYMT2_DEVICE,
        label="llama-cli 翻译",
    ))


def _chunk_text(text: str, limit: int) -> list[str]:
    """把源文案切成 ≤ limit 字的段落块（0 = 不分块）。

    优先按换行分段；单段仍超限时按句号/问号/叹号/分号断句，最后按字数硬切。
    """
    if limit <= 0 or len(text) <= limit:
        return [text]
    blocks: list[str] = []
    cur = ""
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            if cur:
                blocks.append(cur)
                cur = ""
            continue
        while len(para) > limit:
            if cur:
                blocks.append(cur)
                cur = ""
            cut = -1
            for i in range(limit - 1, max(limit - 120, 0) - 1, -1):
                if para[i] in "。！？；":
                    cut = i + 1
                    break
            if cut < 0:
                cut = limit
            blocks.append(para[:cut])
            para = para[cut:].lstrip("，、,. ")
        if cur and len(cur) + len(para) + 1 > limit:
            blocks.append(cur)
            cur = para
        else:
            cur = (cur + " " + para) if cur else para
    if cur:
        blocks.append(cur)
    return [b for b in blocks if b.strip()] or [text]


def _build_prompt(template: str, text: str) -> str:
    """把源文案填入提示词模板：模板含 {text} 则替换，否则拼在末尾。"""
    if "{text}" in template:
        return template.replace("{text}", text)
    return template.rstrip() + "\n\n" + text


# 提示词模板文件缺失/为空时的兜底（正常情况下文件始终随仓库存在）
_FALLBACK_PROMPT = (
    "请将下面的普通话文案翻译成地道的香港粤语，逐句翻译，唔好照抄原文，"
    "全文用香港繁体字输出，只输出译文，不加解释。\n\n{text}"
)


def default_prompt() -> str:
    """读取默认提示词模板（HYMT2_PROMPT_FILE 指向的纯文本，可直接手改）。

    文件缺失或内容为空时返回内置兜底模板，不中断使用。
    """
    try:
        with open(HYMT2_PROMPT_FILE, encoding="utf-8") as f:
            content = f.read().strip("\n")
    except OSError:
        return _FALLBACK_PROMPT
    return content.strip() or _FALLBACK_PROMPT


def translate(text: str, prompt: str | None = None,
              logger: logging.Logger | None = None) -> str:
    """普通话/中文文案 → 粤语文案。

    text：源文案；prompt：提示词模板（默认读取 HYMT2_PROMPT_FILE 指向的
    纯文本，web 页右侧可编辑框的值直接传入）。长文案自动分块逐段翻译，
    段间以空行连接。
    """
    if logger is None:
        logger = logging.getLogger("omni")
    text = (text or "").strip()
    if not text:
        raise RuntimeError("待翻译文案为空。")
    template = (prompt or "").strip() or default_prompt()

    model = _ensure_model(logger)
    chunks = _chunk_text(text, HYMT2_CHUNK_CHARS)
    logger.info("粤语翻译: 源文案 %d 字，分 %d 段（模型 %s）",
                len(text), len(chunks), os.path.basename(model))
    parts: list[str] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        logger.info("[%d/%d] 翻译中 …", i, total)
        out = _run_once(model, _build_prompt(template, chunk), logger)
        # 输出偶发带"译文："前缀/引号包裹，做轻量清理
        cleaned = out
        for prefix in ("译文：", "译文:", "粤语译文：", "粤语：", '"'):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip()
                break
        # 1.8B 偶发不执行粤语转换（回显原文或转繁体书面语），
        # 输出不含粤语功能字时自动重跑一次再取用
        if not cleaned or not _looks_cantonese(cleaned):
            logger.warning(
                "第 %d/%d 段输出未见粤语用字，自动重跑一次 …", i, total)
            out2 = _run_once(model, _build_prompt(template, chunk), logger)
            if _looks_cantonese(out2):
                cleaned = out2
        parts.append(cleaned)
    result = "\n\n".join(p for p in parts if p)
    # 简体字会让 audiocpp 粤语前端按普通话读音处理（如"消费者"读错），
    # 统一转香港繁体后再返回
    return _to_hk_trad(result)
