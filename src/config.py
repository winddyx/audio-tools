"""
OmniVoice 配音工具 — 核心配置（唯一设置源）

包含：Config 数据类 + 全局可调设置（GGUF 权重 / C++ 二进制 / ASR / Web 选项）
+ 设备检测与容错 + 日志噪音控制。原 config.py / device.py / logs.py 合并于此，
所有可调项统一在本文件维护，cli / web 与核心模块从这里取值。

规则：
- 直接改本文件里的顶部变量即可生效；
- 同名环境变量在运行时覆盖文件默认值（如 `OMNIVOICE_GGUF_BASE=...`）；
- cli.py / web.py 与核心模块（src/gguf.py、src/funasr.py 等）都从这里取值，
  不再各自散落默认值。
"""

from __future__ import annotations

import logging
import os
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
    """全局默认配置。可通过环境变量或 CLI 参数覆盖。"""

    # ── 模型（统一由 HuggingFace 管理；路径取自下载接口返回值）──
    model_id: str = "k2-fsa/OmniVoice"  # transformers 后端模型 ID（GGUF 后端忽略）
    model_path: str = ""      # 非空时优先于 model_id（本地 snapshot 目录）
    device: str = ""          # 留空则自动检测（CUDA > XPU > MPS > CPU）
    dtype: str = ""           # 仅 transformers 后端生效；GGUF 量化随文件而定（默认 BF16）

    # ── 生成模式 ──
    language: str = ""        # 语言代码/名称（如 en / zh / English）；留空 = 自动判断
    ref_audio: str = ""
    ref_text: str = ""        # 参考音频转写文本；留空则用 SenseVoiceSmall-GGUF 转写
    instruct: str = ""        # 声音设计指令（如 "female, low pitch, british accent"）

    # ── 长文本配音 ──
    text_path: str = ""
    draw_count: int = 2       # 抽卡次数
    output_dir: str = ""      # 留空则输出到文本文件所在目录

    # ── ASR 子命令（可选，SenseVoiceSmall-GGUF；用于校对/数据集/验证）──
    transcribe: bool = False  # --transcribe：转写 ref_audio 并打印文本
    asr_model: str = ""       # 本地 SenseVoice GGUF 文件路径（默认经 HF 下载 q8，见下方 ASR 设置）
    asr_hub: str = ""         # 保留字段（GGUF 经 huggingface_hub 下载，无 hub 切换）
    asr_vad: str = ""         # 保留字段（VAD 由 fsmn-vad.gguf 承担，见下方 ASR 设置）
    asr_lang_sym: str = ""    # 保留字段（SenseVoice 自动检测语言）
    asr_region_sym: str = ""  # 保留字段（SenseVoice 不支持地区强制，已废弃）

    # 生成参数不在此配置：全部交由后端自身默认值，如需覆盖用环境变量
    # （见 src/omni.py 的 _GEN_PARAM_ENVS / _gen_kwargs）。


# ── 推理后端 ──────────────────────────────────────────────
# 只保留 GGUF 后端（C++/GGML 推理：Serveurperso/OmniVoice-GGUF，输出 24 kHz；
# 首次运行自动 clone+编译 omnivoice.cpp 到项目内 vendor/ 并下载权重到 HF 缓存）。
# cli.py / web.py 直接 from src.gguf import generate, _load_model。

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


# ── 设备检测与容错 ────────────────────────────────────────
# 设备加速优先级（全局规则）：CUDA > XPU(Intel oneAPI/Arc GPU) > MPS > CPU。
# - cuda / xpu 推理默认 bfloat16，mps 用 fp32（mps 不支持 bf16 且 torch 2.13
#   在 MPS 上加载 fp16 权重会 SIGTRAP 崩溃，已实测复现）。
# - mps / xpu 加载或推理失败 → 警告并自动回退 CPU 重试一次。
# - MPS 设备下解除 PyTorch MPS 分配器内存上限（避免系统其它进程占用高时
#   连几十 MiB 都申请不到而报 "MPS backend out of memory"）。


def get_best_device() -> str:
    """自动检测最佳可用设备：CUDA > XPU(Intel oneAPI/Arc GPU) > MPS > CPU。

    - XPU 需安装 PyTorch 官方 xpu 构建（pip install torch --index-url
      https://download.pytorch.org/whl/xpu），普通构建没有 torch.xpu，
      用 getattr 防御式探测，缺失时静默跳过。
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        _apply_mps_memory_settings("mps", logging.getLogger("omni"))
        return "mps"
    return "cpu"


def _is_mps_oom(e: Exception) -> bool:
    """MPS 后端内存不足（"MPS backend out of memory (…)" 报错）？"""
    return (isinstance(e, RuntimeError)
            and "MPS" in str(e)
            and "out of memory" in str(e).lower())


def _should_fallback_to_cpu(e: Exception, device: str) -> bool:
    """设备（mps/xpu）加载/推理失败是否应自动回退 CPU。

    - mps：仅限明确的内存不足（"MPS backend out of memory"）；其它错误属
      代码/输入问题，回退 CPU 无意义，直接抛出。
    - xpu：Intel GPU 运行时/驱动偶发不稳定，任意 RuntimeError 都回退 CPU
      重试一次更稳（符合设备加速优先级规则：mps/xpu 推理失败则警告回退 cpu）。
    """
    if device == "mps":
        return _is_mps_oom(e)
    if device == "xpu":
        return isinstance(e, RuntimeError)
    return False


def _apply_mps_memory_settings(device: str, logger: logging.Logger) -> None:
    """MPS 设备下解除 PyTorch MPS 分配器的内存上限（幂等，可重复调用）。

    MPS 默认上限接近系统总内存，且受系统其它进程占用影响——其它占用高时
    连几十 MiB 都申请不到（报 "MPS backend out of memory (other allocations: …)"）。
    设 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 解除上限，按需向系统申请（可能
    挤占系统内存，故 MPS OOM 时另有自动回退 CPU 兜底）。须在 MPS 分配器首次
    初始化前设置；用户已显式设置该变量时尊重用户值。
    """
    if device != "mps":
        return
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        return
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    logger.info("MPS 内存上限已解除（PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0；"
                "MPS OOM 时自动回退 CPU）")


_THREAD_CONFIGURED = False  # CPU 线程池已配置（幂等标志，避免重复设置/重复日志）


def _configure_cpu_threads(cfg, logger: logging.Logger) -> None:
    """CPU 设备下显式配置 PyTorch 线程池，用满所有可用核心（仅执行一次）。

    - 线程数取值优先级：THREADS 环境变量 > OMP_NUM_THREADS > os.cpu_count()。
      torch 默认只按物理核心建线程池，Windows 超线程机器上逻辑核心会闲置；
      显式设置后全部逻辑核心参与计算，对长文本配音的 batch 算子有明显增益。
    - 只调整 torch 的线程池（torch.set_num_threads），不改环境变量、不影响
      numpy 等其他 OpenMP 库；CUDA/MPS 设备跳过（GPU 推理与线程数无关）。
    - 在模型加载前调用（加载/推理是首次算子执行点），CLI 与 web 共用
      （web 复用 src 的 _load_model / ASR 加载，自动生效）。
    """
    global _THREAD_CONFIGURED
    if _THREAD_CONFIGURED:
        return
    if (cfg.device or "cpu") != "cpu":
        _THREAD_CONFIGURED = True
        return

    import torch
    n = os.cpu_count() or 1
    for env in ("THREADS", "OMP_NUM_THREADS"):
        v = (os.environ.get(env) or "").strip()
        if v.isdigit() and int(v) >= 1:
            n = int(v)
            break
    torch.set_num_threads(n)
    _THREAD_CONFIGURED = True
    logger.info("CPU 线程池: %d 线程（可用核心 %d；THREADS 环境变量可覆盖）",
                n, os.cpu_count() or 1)


# ── 日志噪音控制 ──────────────────────────────────────────
# 把 HuggingFace 相关第三方库（httpx / huggingface_hub / urllib3 等）的
# INFO 日志压到 WARNING，保留 WARNING 及以上提示（如缺 HF_TOKEN）与业务日志。
# CLI / web 入口在 logging.basicConfig 之后调用 _quiet_hf_logs()。


def _quiet_hf_logs() -> None:
    """把 HuggingFace 相关第三方库的 INFO 日志压到 WARNING。

    符合模型管理规则：HF 相关 INFO 日志（httpx/huggingface_hub/urllib3 等）
    压到 WARNING，保留 WARNING 及以上提示（如缺 HF_TOKEN）与业务日志。
    """
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                 "filelock", "fsspec"):
        logging.getLogger(name).setLevel(logging.WARNING)