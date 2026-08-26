"""
OmniVoice 配音工具 — 全局设置（所有可调变量统一在这里）

规则：
- 直接改本文件里的变量即可生效；
- 同名环境变量在运行时覆盖文件默认值（如 `OMNIVOICE_GGUF_BASE=...`）；
- cli.py / web.py 与核心模块（src/backends/gguf.py、src/asr.py 等）都从这里
  取值，不再各自散落默认值；改权重 / 二进制 / 端口等只需动这一处。
"""

from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    """环境变量取值：未设置或为空时用文件默认值。"""
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ── 推理后端 ──────────────────────────────────────────────
# 只保留 GGUF 后端（C++/GGML 推理：Serveurperso/OmniVoice-GGUF，输出 24 kHz；
# 首次运行自动 clone+编译 omnivoice.cpp 到项目内 vendor/ 并下载权重到 HF 缓存）。
# cli.py / web.py 直接 from src.backends.gguf import generate, _load_model。

# ── GGUF 后端：权重（HuggingFace 仓库 Serveurperso/OmniVoice-GGUF）──
GGUF_REPO = _env("OMNIVOICE_GGUF_REPO", "Serveurperso/OmniVoice-GGUF")
# 变体：BF16（默认，精度最高）/ Q8_0（更小更快）/ F32 / Q4_K_M，base 与 codec 需配套
GGUF_BASE = _env("OMNIVOICE_GGUF_BASE", "omnivoice-base-BF16.gguf")
GGUF_CODEC = _env("OMNIVOICE_GGUF_CODEC", "omnivoice-tokenizer-BF16.gguf")

# ── GGUF 后端：omnivoice.cpp 二进制（留空 = 自动 clone+编译到 vendor/）──
CPP_BIN = _env("OMNIVOICE_CPP_BIN", "")             # 已编译的 omnivoice-tts 绝对路径
CPP_SRC = _env("OMNIVOICE_CPP_SRC", "")             # 已有源码目录（默认 vendor/omnivoice.cpp）
CPP_BUILD_ARGS = _env("OMNIVOICE_CPP_BUILD_ARGS", "")  # 追加 cmake 参数，如 "-DGGML_CUDA=ON"
# True = 透传 omnivoice-tts 的全部 stderr（ggml 内核编译 / MaskGIT 步进等，调试用）
GGUF_DEBUG = _env_bool("OMNIVOICE_GGUF_DEBUG", False)

# ── ASR（参考音频转写）：SenseVoiceSmall-GGUF（FunASR llama.cpp runtime）──
# 权重经 HF 下载（本地优先 + 镜像兜底），二进制 llama-funasr-sensevoice
# 留空时自动从 GitHub Releases 下载预编译包到项目内 vendor/funasr-llamacpp/。
ASR_GGUF_REPO = _env("ASR_GGUF_REPO", "FunAudioLLM/SenseVoiceSmall-GGUF")
ASR_GGUF_BASE = _env("ASR_GGUF_BASE", "sensevoice-small-q8.gguf")  # q8 / f16 / 原版
ASR_VAD_REPO = _env("ASR_VAD_REPO", "FunAudioLLM/fsmn-vad-GGUF")
ASR_VAD_BASE = _env("ASR_VAD_BASE", "fsmn-vad.gguf")
FUNASR_LLAMACPP_BIN = _env("FUNASR_LLAMACPP_BIN", "")  # 已下载的二进制路径（留空自动获取）

# ── Web 界面 ──────────────────────────────────────────────
WEB_IP = _env("OMNIVOICE_WEB_IP", "0.0.0.0")
WEB_PORT = _env_int("OMNI_PORT", 38001)
# True = 启动后自动用默认浏览器打开界面
WEB_AUTO_OPEN_BROWSER = _env_bool("OMNIVOICE_WEB_OPEN_BROWSER", False)
