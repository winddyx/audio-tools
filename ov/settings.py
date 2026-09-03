"""唯一设置文件：一切可调设置 = 本文件顶部变量。

规则（项目通用约定）：
- 直接改本文件里的变量默认值即可生效；
- 同名环境变量在运行时覆盖（如 OMNIVOICE_GGUF_BASE=...）；
- CLI / Web / 各模型引擎都从这里取值，不各自散落默认值。

设备策略（v2，不再依赖 torch）：
- DEVICE 留空时由引擎自选：darwin 优先试 Metal(MTL0)，其余交给运行时默认；
  后端初始化失败自动回退 CPU 重试一次。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ── 环境变量解析助手 ──────────────────────────────────────


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_opt_int(name: str) -> Optional[int]:
    """可选整数：未设置/为空/非法 → None（交给引擎默认值）。"""
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _env_opt_float(name: str) -> Optional[float]:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _env_opt_bool(name: str) -> Optional[bool]:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ── 通用运行设置 ──────────────────────────────────────────
LOG_LEVEL = _env("OV_LOG_LEVEL", "INFO")
OUTPUT_DIR = _env("OUTPUT_DIR", "")          # 留空：CLI 输出到文本文件目录 / Web 输出到项目 out/
RUNTIME_DIR = _env("OV_RUNTIME_DIR", "")     # 引擎二进制与构建产物目录（默认 <项目根>/runtime/）
DEVICE = _env("OV_DEVICE", "")               # cuda/mps/cpu/留空=自选（见文件头策略）

# ── 生成参数默认覆盖（None = 用引擎自身默认值；同名 env 可覆盖）──
NUM_STEP = _env_opt_int("NUM_STEP")
DENOISE = _env_opt_bool("DENOISE")
AUDIO_CHUNK_DURATION = _env_opt_float("AUDIO_CHUNK_DURATION")
AUDIO_CHUNK_THRESHOLD = _env_opt_float("AUDIO_CHUNK_THRESHOLD")
DURATION = _env_opt_float("DURATION")

# ── CLI 默认行为（不设启动参数）───────────────────────────
DRAW_COUNT = _env_int("DRAW_COUNT", 2)       # 每轮抽卡次数
DEFAULT_LANGUAGE = _env("LANGUAGE", "")      # 合成语言；留空 = 引擎自动判断
DEFAULT_INSTRUCT = _env("INSTRUCT", "")      # 非空时 CLI 走"声音设计"模式
REF_TEXT_FILE = _env("REF_TEXT_FILE", "")    # 克隆模式参考文本文件（留空 = 自动 ASR 转写）

# ── 模型资产：OmniVoice TTS（GGUF，HF 下载）───────────────
OV_GGUF_REPO = _env("OMNIVOICE_GGUF_REPO", "Serveurperso/OmniVoice-GGUF")
OV_GGUF_BASE = _env("OMNIVOICE_GGUF_BASE", "omnivoice-base-BF16.gguf")
OV_GGUF_CODEC = _env("OMNIVOICE_GGUF_CODEC", "omnivoice-tokenizer-BF16.gguf")
OV_CPP_BIN = _env("OMNIVOICE_CPP_BIN", "")       # 已编译 omnivoice-tts 绝对路径
OV_CPP_SRC = _env("OMNIVOICE_CPP_SRC", "")       # 已有源码目录（默认 runtime/omnivoice.cpp）
OV_CPP_BUILD_ARGS = _env("OMNIVOICE_CPP_BUILD_ARGS", "")  # 追加 cmake 参数
OV_DEBUG = _env_bool("OMNIVOICE_GGUF_DEBUG", False)        # 透传引擎 stderr

# ── 模型资产：SenseVoice ASR（GGUF，HF 下载）──────────────
ASR_ENGINE_ID = _env("ASR_ENGINE_ID", "sensevoice")   # 克隆缺 ref_text 时用于自动转写
ASR_GGUF_REPO = _env("ASR_GGUF_REPO", "FunAudioLLM/SenseVoiceSmall-GGUF")
ASR_GGUF_BASE = _env("ASR_GGUF_BASE", "sensevoice-small-q8.gguf")
ASR_VAD_REPO = _env("ASR_VAD_REPO", "FunAudioLLM/fsmn-vad-GGUF")
ASR_VAD_BASE = _env("ASR_VAD_BASE", "fsmn-vad.gguf")
ASR_BIN = _env("FUNASR_LLAMACPP_BIN", "")          # 已下载二进制；留空自动获取

# ── Web 界面 ──────────────────────────────────────────────
WEB_IP = _env("OMNIVOICE_WEB_IP", "0.0.0.0")
WEB_PORT = _env_int("OMNI_PORT", 38001)
WEB_AUTO_OPEN_BROWSER = _env_bool("OMNIVOICE_WEB_OPEN_BROWSER", False)


# ── 派生路径（不需要用户改，除非要换位置）─────────────────
def project_root() -> Path:
    """项目根：<repo>/ov/settings.py → ov → 项目根。"""
    return Path(__file__).resolve().parent.parent


def runtime_dir() -> Path:
    return Path(RUNTIME_DIR) if RUNTIME_DIR else project_root() / "runtime"
