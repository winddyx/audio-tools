"""
OmniVoice 配音工具 — 设备检测与容错

设备加速优先级（全局规则）：CUDA > XPU(Intel oneAPI/Arc GPU) > MPS > CPU。
- cuda / xpu 推理默认 bfloat16，mps 用 fp32（mps 不支持 bf16 且 torch 2.13
  在 MPS 上加载 fp16 权重会 SIGTRAP 崩溃，已实测复现）。
- mps / xpu 加载或推理失败 → 警告并自动回退 CPU 重试一次。
- MPS 设备下解除 PyTorch MPS 分配器内存上限（避免系统其它进程占用高时
  连几十 MiB 都申请不到而报 "MPS backend out of memory"）。
"""

from __future__ import annotations

import logging
import os


# ── 设备检测 ──────────────────────────────────────────────


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
    logger.info("⚙️  MPS 内存上限已解除（PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0；"
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
    logger.info("🧵 CPU 线程池: %d 线程（可用核心 %d；THREADS 环境变量可覆盖）",
                n, os.cpu_count() or 1)
