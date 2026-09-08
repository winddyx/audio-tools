"""
audio-tools — 粤语文案翻译（Hy-MT2-1.8B，llama.cpp 子进程）

普通话/中文文案 → 粤语口播文案：为语音克隆提供粤语 TTS 输入。Hy-MT2 是
腾讯开源的多语言翻译模型（官方支持粤语 yue 双向互译），本模块用本机
llama-completion（llama.cpp 单次补全 CLI）子进程做推理，Python 只做编排：

- 模型经 HuggingFace 下载，只留在 HF 默认缓存并生成 .gguf 硬链接别名
  （src/hf._ensure_gguf_file），不落工程目录；HYMT2_LOCAL 可手工放置；
- 提示词模板（config.HYMT2_PROMPT）含 {text} 占位，运行时被源文案替换；
- 长文案按段落分块（HYMT2_CHUNK_CHARS）逐段翻译后拼接，避免长文漏翻；
- llama-cli 失败抛 RuntimeError（带 stderr 尾部诊断），不在入口裸奔。

web.py「粤语翻译」页调用 translate()。采样/设备等可调参数统一在
src/config.py 顶部常量（同名 env 可覆盖），本文件不再散落默认值。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

from .config import (
    HYMT2_CHUNK_CHARS,
    HYMT2_DEVICE,
    HYMT2_FILE,
    HYMT2_LOCAL,
    HYMT2_MAX_TOKENS,
    HYMT2_PROMPT,
    HYMT2_REPETITION_PENALTY,
    HYMT2_REPO,
    HYMT2_TEMPERATURE,
    HYMT2_TIMEOUT,
    HYMT2_TOP_K,
    HYMT2_TOP_P,
    LLAMA_CLI,
)
from .hf import _ensure_gguf_file

# 粤语功能字标志集：译文若完全不含这些字，基本可判定模型没执行粤语转换
# （回显原文/繁体书面化），translate() 会自动重跑一次。
_YUE_MARKERS = "嘅咗喺哋啲係唔睇畀諗乜嘢仲同埋呢啲嗰啲呢度嗰度"


def _looks_cantonese(text: str) -> bool:
    """粗判输出是否做了粤语转换（含任意粤语功能字即认为已转换）。"""
    return any(ch in _YUE_MARKERS for ch in text)


def _binary(logger: logging.Logger) -> str:
    """定位 llama-cli：LLAMA_CLI 绝对路径优先，否则 PATH 查找。"""
    if os.path.sep in LLAMA_CLI and os.path.isfile(LLAMA_CLI):
        return LLAMA_CLI
    found = shutil.which(LLAMA_CLI)
    if found:
        return found
    raise RuntimeError(
        "未找到翻译可执行文件（当前设置 LLAMA_CLI = %s），请先安装 "
        "llama.cpp（brew install llama.cpp），或在 src/config.py 设置 "
        "LLAMA_CLI 指向可执行文件。" % LLAMA_CLI
    )


def _ensure_model(logger: logging.Logger) -> str:
    """返回可直接喂给 llama-cli 的模型 .gguf 路径（本地手工放置优先）。"""
    if HYMT2_LOCAL:
        path = os.path.abspath(HYMT2_LOCAL)
        if not os.path.isfile(path):
            raise RuntimeError(f"HYMT2_LOCAL 指向的文件不存在: {path}")
        return path
    logger.info("翻译模型: %s/%s（Hy-MT2-1.8B，llama.cpp）",
                HYMT2_REPO, HYMT2_FILE)
    return _ensure_gguf_file(HYMT2_REPO, HYMT2_FILE, logger)


def _run_once(model: str, prompt: str, logger: logging.Logger) -> str:
    """调用 llama-cli 完成一次翻译，返回模型输出文本。

    采样参数用 config 顶部常量（默认腾讯官方推荐）；失败抛 RuntimeError
    （含 stderr 尾部）。HYMT2_DEVICE=cpu 时加 --device none 强制 CPU。
    """
    cmd = [
        _binary(logger),
        "-m", model,
        "-p", prompt,
        "-n", str(HYMT2_MAX_TOKENS),
        "--temp", HYMT2_TEMPERATURE,
        "--top-p", HYMT2_TOP_P,
        "--top-k", str(HYMT2_TOP_K),
        "--repeat-penalty", HYMT2_REPETITION_PENALTY,
        "--no-display-prompt",
        "--jinja",
        "-st",
    ]
    if HYMT2_DEVICE == "cpu":
        cmd += ["--device", "none"]
    elif HYMT2_DEVICE:
        cmd += ["--device", HYMT2_DEVICE]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=HYMT2_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"llama-cli 翻译超时（>{HYMT2_TIMEOUT}s）；可调大 "
            "HYMT2_TIMEOUT 或改小 HYMT2_MAX_TOKENS / HYMT2_CHUNK_CHARS。"
        )
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-600:]
        raise RuntimeError(
            f"llama-completion 翻译失败（rc={r.returncode}，{dt:.0f}s）: {tail}")
    logger.info("llama-completion 完成: %.1fs（输出 %d 字符）", dt, len(r.stdout or ""))
    out = (r.stdout or "").strip()
    # llama-completion 会把结束标记（[end of text] 等）也打到 stdout，清掉
    for marker in ("[end of text]", "[end of turn]", "[EOT]"):
        while out.endswith(marker):
            out = out[:-len(marker)].rstrip()
    return out


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


def translate(text: str, prompt: str | None = None,
              logger: logging.Logger | None = None) -> str:
    """普通话/中文文案 → 粤语文案。

    text：源文案；prompt：提示词模板（默认 config.HYMT2_PROMPT，web 页
    右侧可编辑框的值直接传入）。长文案自动分块逐段翻译，段间以空行连接。
    """
    if logger is None:
        logger = logging.getLogger("omni")
    text = (text or "").strip()
    if not text:
        raise RuntimeError("待翻译文案为空。")
    template = (prompt or "").strip() or HYMT2_PROMPT

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
    return "\n\n".join(p for p in parts if p)
