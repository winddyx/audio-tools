"""
OmniVoice 配音工具 — 全局设置（所有可调变量统一在这里）

规则：
- 直接改本文件里的变量即可生效；
- 同名环境变量在运行时覆盖文件默认值（如 `OMNIVOICE_BACKEND=transformers`
  临时切换后端，无需改文件）；
- cli.py / web.py 与核心模块（src/backends/gguf.py 等）都从这里取值，
  不再各自散落默认值；改后端 / 权重 / 端口等只需动这一处。
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
# "gguf"（默认）：C++/GGML 推理（Serveurperso/OmniVoice-GGUF，输出 24 kHz；
#   首次运行自动 clone+编译 omnivoice.cpp 到项目内 vendor/ 并下载权重到 HF 缓存）
# "transformers"：原 k2-fsa/OmniVoice transformers 实现
BACKEND = _env("OMNIVOICE_BACKEND", "gguf")

# ── GGUF 后端：权重（HuggingFace 仓库 Serveurperso/OmniVoice-GGUF）──
GGUF_REPO = _env("OMNIVOICE_GGUF_REPO", "Serveurperso/OmniVoice-GGUF")
# 变体：BF16（默认，精度最高）/ Q8_0（更小更快）/ F32 / Q4_K_M，base 与 codec 需配套
GGUF_BASE = _env("OMNIVOICE_GGUF_BASE", "omnivoice-base-BF16.gguf")
GGUF_CODEC = _env("OMNIVOICE_GGUF_CODEC", "omnivoice-tokenizer-BF16.gguf")

# ── GGUF 后端：omnivoice.cpp 二进制（留空 = 自动 clone+编译到 vendor/）──
CPP_BIN = _env("OMNIVOICE_CPP_BIN", "")             # 已编译的 omnivoice-tts 绝对路径
CPP_SRC = _env("OMNIVOICE_CPP_SRC", "")             # 已有源码目录（默认 vendor/omnivoice.cpp）
CPP_BUILD_ARGS = _env("OMNIVOICE_CPP_BUILD_ARGS", "")  # 追加 cmake 参数，如 "-DGGML_CUDA=ON"

# ── Web 界面 ──────────────────────────────────────────────
WEB_IP = _env("OMNIVOICE_WEB_IP", "0.0.0.0")
WEB_PORT = _env_int("OMNI_PORT", 38001)
# True = 启动后自动用默认浏览器打开界面
WEB_AUTO_OPEN_BROWSER = _env_bool("OMNIVOICE_WEB_OPEN_BROWSER", False)


# ── 后端模块选择（按 BACKEND 懒加载，供 cli.py / web.py 直接使用）──
# 注意：本段必须放在所有变量定义之后（gguf.py 等会在 import 时读上面的常量）。
def get_backend():
    from src.backends import get_backend as _get_backend
    return _get_backend(BACKEND)


_backend = get_backend()
generate = _backend.generate
_load_model = _backend._load_model
