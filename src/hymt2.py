"""
audio-tools — 粤语文案翻译（Hy-MT2-1.8B，llama.cpp 子进程）

普通话/中文文案 → 粤语口播文案：为语音克隆提供粤语 TTS 输入。Hy-MT2 是
腾讯开源的多语言翻译模型（官方支持粤语 yue 双向互译），本模块用本机
llama-completion（llama.cpp 单次补全 CLI）子进程做推理，Python 只做编排：

- 引擎：新版 llama.cpp（ggml >= 0.10）把单次补全拆到 llama-completion。
  LLAMA_CLI 留空时自动 clone + cmake 编译 vendor/llama.cpp（与 audiocpp
  同模式，不依赖本机 brew 安装），产物
  vendor/llama.cpp/build/bin/llama-completion（vendor 已 gitignore）；
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
import shutil
import subprocess
import time

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
    LLAMA_BUILD_ARGS,
    LLAMA_CLI,
    LLAMA_DEBUG,
    LLAMA_REF,
    LLAMA_REPO,
    VENDOR_DIR,
)
from .hf import _ensure_gguf_file

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


_LLAMA_BIN_NAME = "llama-completion"


def _binary(logger: logging.Logger) -> str:
    """定位 llama-completion：LLAMA_CLI 显式设置优先，否则自动构建 vendor。

    - LLAMA_CLI 非空：当作可执行名（PATH 查找）或绝对路径，找不到即报错；
    - LLAMA_CLI 留空（默认）：复用 vendor/llama.cpp/build/bin 下已有产物，
      缺失则自动 clone + cmake 编译（同 audiocpp 引擎模式，不依赖本机安装）。
    """
    if LLAMA_CLI:
        if os.path.sep in LLAMA_CLI and os.path.isfile(LLAMA_CLI):
            return LLAMA_CLI
        found = shutil.which(LLAMA_CLI)
        if found:
            return found
        raise RuntimeError(f"LLAMA_CLI 指向的可执行未找到: {LLAMA_CLI}")
    built = os.path.join(VENDOR_DIR, "llama.cpp", "build", "bin", _LLAMA_BIN_NAME)
    if os.path.isfile(built):
        return built
    return _ensure_built(logger)


def _run_step(cmd: list[str], logger: logging.Logger, desc: str) -> None:
    """执行 clone/编译等一次性步骤；失败抛 RuntimeError（带输出尾部）。"""
    t0 = time.time()
    try:
        if LLAMA_DEBUG:
            r = subprocess.run(cmd)          # 透传原始输出（调试用）
        else:
            r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{desc}失败：缺少命令 {e.filename}（请检查 git/cmake 是否安装）")
    if r.returncode != 0:
        tail = "" if LLAMA_DEBUG else (r.stderr or r.stdout or "").strip()[-800:]
        raise RuntimeError(f"{desc}失败（rc={r.returncode}）: {tail}")
    logger.info("%s完成（%.0fs）", desc, time.time() - t0)


def _ensure_built(logger: logging.Logger) -> str:
    """自动 clone + 编译 llama.cpp，返回 llama-completion 绝对路径。"""
    src = os.path.join(VENDOR_DIR, "llama.cpp")
    build = os.path.join(src, "build")
    binary = os.path.join(build, "bin", _LLAMA_BIN_NAME)
    if shutil.which("cmake") is None:
        raise RuntimeError("未找到 cmake，请先安装（brew install cmake）后重试。")
    if not os.path.isfile(os.path.join(src, "CMakeLists.txt")):
        logger.info("未找到 llama.cpp 源码，clone %s（分支 %s）…",
                    LLAMA_REPO, LLAMA_REF)
        os.makedirs(VENDOR_DIR, exist_ok=True)
        _run_step([
            "git", "clone", "--depth", "1", "--branch", LLAMA_REF,
            "--single-branch", "--recurse-submodules",
            LLAMA_REPO, src,
        ], logger, "git clone llama.cpp")
    logger.info("cmake 配置 vendor/llama.cpp（首次较久）…")
    cfg = ["cmake", "-S", src, "-B", build, "-DCMAKE_BUILD_TYPE=Release"]
    cfg += (LLAMA_BUILD_ARGS or "").split()
    _run_step(cfg, logger, "cmake 配置")
    logger.info("cmake 编译 %s（多核，请耐心等待）…", _LLAMA_BIN_NAME)
    _run_step([
        "cmake", "--build", build, "--config", "Release",
        "--target", _LLAMA_BIN_NAME, "--parallel",
    ], logger, "cmake 编译")
    if not os.path.isfile(binary):
        raise RuntimeError(f"编译完成但未找到产物: {binary}")
    logger.info("%s 就绪: %s", _LLAMA_BIN_NAME, binary)
    return binary


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
