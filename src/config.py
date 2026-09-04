"""
audio-tools — 核心配置（唯一设置源）

包含：Config 数据类 + 全局可调设置（推理引擎 audiocpp / TTS 与 ASR 模型 /
Web 选项）+ 设备检测。原 config.py 的 torch 依赖逻辑（transformers 后端、
MPS 内存设置、PyTorch 线程池）已随 torch 后端移除而删除——推理全部在
audio.cpp C++ 子进程完成，Python 侧不再 import torch。

规则：
- 直接改本文件里的顶部变量即可生效；
- 同名环境变量在运行时覆盖文件默认值（如 `TTS_MODEL=indextts2`）；
- vc.py / web.py 与核心模块都从这里取值，不再各自散落默认值。

目录规划（业务入口在根，核心在 src/）：
- 根目录：vc.py（CLI 入口）、web.py（Gradio 入口）
- src/：config.py（本文件，全局设置）、audiocpp.py（推理引擎运行器）、
  omnivoice.py / indextts2.py（TTS 模型核心）、sensevoice.py（ASR 核心）、
  hf.py（HuggingFace 下载）、pipeline.py（统一编排）
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    """环境变量取值：未设置或为空时用文件默认值。"""
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    """环境变量整数取值：未设置、为空或解析失败时用文件默认值。"""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _to_bool(v: str) -> bool:
    """把字符串解析为布尔（1/true/yes/on → True，其余 → False）。"""
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool) -> bool:
    """环境变量布尔取值：未设置或为空时用文件默认值。"""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return _to_bool(v)


@dataclass
class Config:
    """全局默认配置。可通过环境变量覆盖。"""

    # ── 模型（本地优先；缺失时自动从 HuggingFace 下载到 HF 缓存）──
    tts_model: str = ""       # "omnivoice" | "indextts2"；留空用 TTS_MODEL
    device: str = ""          # 留空则自动检测（cuda > xpu > mps > cpu）

    # ── 生成模式（本期只做语音克隆）──
    language: str = ""        # 语言代码（如 en / zh / yue）；留空 = 自动判断
    ref_audio: str = ""
    ref_text: str = ""        # 参考音频转写文本；留空则用 SenseVoice 转写

    # ── 长文本配音 ──
    text_path: str = ""
    draw_count: int = 2       # 抽卡次数
    output_dir: str = ""      # 留空则输出到文本文件所在目录

    # ── ASR 子命令（可选，SenseVoice；用于校对/数据集/验证）──
    transcribe: bool = False  # --transcribe：转写 ref_audio 并打印文本
    asr_model: str = ""       # 本地 SenseVoice GGUF 文件路径（默认用 ASR_GGUF_*）


# ── 项目内固定路径（模型/引擎可换，目录本身不可调）────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = os.path.join(_PROJECT_ROOT, "vendor")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")   # 本地模型目录（gitignore）
TMP_DIR = os.path.join(_PROJECT_ROOT, ".tmp")        # 运行期临时目录


# ── 推理引擎：audio.cpp（audiocpp_cli，ggml 框架）──────────
# 引擎源仓库固定（自动 clone 时使用；已有源码目录可用 AUDIOCPP_SRC 指向）
AUDIOCPP_REPO = "https://github.com/0xShug0/audio.cpp.git"
AUDIOCPP_BIN = _env("AUDIOCPP_BIN", "")     # 已编译二进制绝对路径（留空自动定位/构建）
AUDIOCPP_SRC = _env("AUDIOCPP_SRC", "")     # 已有源码目录（默认 vendor/audiocpp）
# 追加 cmake 参数（如 "-DGGML_CUDA=ON"）；构建默认参数见 src/audiocpp.py
AUDIOCPP_BUILD_ARGS = _env("AUDIOCPP_BUILD_ARGS", "")
# True = 透传子进程原始输出（构建/推理全部直通终端，调试用）
AUDIOCPP_DEBUG = _env_bool("AUDIOCPP_DEBUG", False)


# ── TTS 模型（audiocpp 族）────────────────────────────────
# TTS_MODEL 切换模型（弱化单一模型绑定）：omnivoice / indextts2。
# 各模型的 GGUF 文件与 HF 兜底仓库定义在对应模型核心
# （src/omnivoice.py、src/indextts2.py），本文件只放默认选择与本地目录。
TTS_MODEL = _env("TTS_MODEL", "omnivoice")

# ── 生成参数（各模型核心在拼 CLI 时消费；空值 = 不传 flag = 用引擎默认）──
# 每个模型族只消费自己支持的参数，其余仍按核心内"忽略并提示"处理：
# - OmniVoice：去噪步数 / CFG 引导尺度（引擎默认 32 步 / 2.0）+ 随机种子
# - IndexTTS-2.5：gpt 层 top-k / top-p / temperature（引擎默认 30 / 0.8 / 0.8）+ 随机种子
OMNI_INFERENCE_STEPS = _env_int("OMNI_INFERENCE_STEPS", 0)   # 0 = 引擎默认
OMNI_GUIDANCE_SCALE = _env("OMNI_GUIDANCE_SCALE", "")        # 空 = 引擎默认
INDEXTTS_TOP_K = _env_int("INDEXTTS_TOP_K", 0)               # 0 = 引擎默认
INDEXTTS_TOP_P = _env("INDEXTTS_TOP_P", "")                  # 空 = 引擎默认
INDEXTTS_TEMPERATURE = _env("INDEXTTS_TEMPERATURE", "")      # 空 = 引擎默认
GEN_SEED = _env_int("GEN_SEED", -1)                          # -1 = 随机（不传 seed）

# ── ASR（参考音频转写）：SenseVoice-Small（audiocpp sense_asr 族）──
# 权重经 HF 下载（本地优先 + 镜像兜底）；注意该 GGUF 是 audiocpp 专用包
# （FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp），与旧 llama-funasr 包不同。
ASR_GGUF_REPO = _env("ASR_GGUF_REPO", "FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp")
ASR_GGUF_BASE = _env("ASR_GGUF_BASE", "sensevoice-small-q8-audiocpp-v1.gguf")


# ── Web 界面 ──────────────────────────────────────────────
WEB_IP = _env("AUDIOTOOLS_WEB_IP", "0.0.0.0")
WEB_PORT = _env_int("AUDIOTOOLS_WEB_PORT", 38001)
# True = 启动后自动用默认浏览器打开界面
WEB_AUTO_OPEN_BROWSER = _env_bool("AUDIOTOOLS_WEB_OPEN_BROWSER", False)


# ── 设备检测（无 torch：纯平台探测 + 引擎能力）─────────────
def get_best_device() -> str:
    """自动检测最佳可用设备（cuda > xpu > mps > cpu）。

    audio.cpp 是独立 C++ 引擎，Python 侧不做 torch 探测；这里按平台给
    默认后端，用户可在 DEVICE / 各入口顶部变量显式覆盖（cuda / cpu）。
    - darwin + Apple Silicon → mps（audiocpp 映射 --backend metal）
    - 其余平台 → cpu（NVIDIA GPU 用户请显式设 DEVICE=cuda）
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    if os.environ.get("DEVICE") in ("cuda", "xpu"):
        return os.environ["DEVICE"]
    return "cpu"


def _should_fallback_to_cpu(e: Exception, device: str) -> bool:
    """设备（GPU 后端）初始化失败是否应自动回退 CPU。"""
    if not device:
        return False
    if device in ("cuda", "mps", "xpu"):
        return isinstance(e, RuntimeError)
    return False


# ── 日志噪音控制 ──────────────────────────────────────────
# 把 HuggingFace 相关第三方库（httpx / huggingface_hub / urllib3 等）的
# INFO 日志压到 WARNING，保留 WARNING 及以上提示（如缺 HF_TOKEN）与业务日志。
# CLI / web 入口在 logging.basicConfig 之后调用 _quiet_hf_logs()。


def _quiet_hf_logs() -> None:
    """把 HuggingFace 相关第三方库的 INFO 日志压到 WARNING。"""
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                 "filelock", "fsspec"):
        logging.getLogger(name).setLevel(logging.WARNING)
