"""
transformers 推理后端：k2-fsa/OmniVoice（原 omni.py 的 TTS 实现）

经 `OmniVoice.from_pretrained()` 加载官方 HuggingFace 权重（transformers
PreTrainedModel，自带 config.json / tokenizer / audio_tokenizer），在进程内
生成。与 GGUF 后端（src/backends/gguf.py）接口完全一致：

    _load_model(cfg, logger) → 模型对象（.sampling_rate 等）
    generate(cfg, logger, text=…, language=…, ref_audio=…, ref_text=…,
             instruct=…, **gen_kwargs) → [音频数组]

由 settings.BACKEND 选择（默认 "gguf"，本后端保留为
"transformers" 分支）。模型管理规则见 src/hf.py。
"""

from __future__ import annotations

import logging
import time

from ..config import Config
from ..device import (
    _apply_mps_memory_settings,
    _configure_cpu_threads,
    _should_fallback_to_cpu,
)
from ..hf import resolve_path


_OMNIVOICE_MODEL = None   # 全局 OmniVoice 模型缓存（单例）


# ── 模型加载 ──────────────────────────────────────────────
#
# OmniVoice 是 transformers PreTrainedModel，自带 config.json / tokenizer /
# audio_tokenizer（均在 k2-fsa/OmniVoice 快照内），用 from_pretrained 直接加载，
# 无需生成任何临时 config；快照内没有的辅助资源由模型内部按需经 huggingface_hub
# 下载，同样受 HF_ENDPOINT 镜像约束（参考文本转写已改由 FunASR/SenseVoiceSmall
# 承担，不走 Whisper）。


# DTYPE 环境变量允许值（白名单校验，避免 getattr(torch, x) 抛原始 AttributeError）
_DTYPE_ALLOWED = {"float16", "float32", "float64", "bfloat16"}


def _default_dtype(device: str) -> str:
    """默认精度：CUDA/XPU 用 bfloat16（GPU 原生加速、指数范围大不易溢出），
    MPS/CPU 用 float32。

    torch 2.13 在 MPS 上加载 float16 权重会 SIGTRAP 崩溃（已实测复现），
    MPS 上必须用 float32；CPU 同理。CUDA/XPU 的 bfloat16 由 GPU 原生加速、
    精度比 float16 稳（指数范围大）；老架构 GPU 若不支持 bf16 可用
    DTYPE=float16 显式覆盖。可用 DTYPE 环境变量覆盖。
    """
    if device in ("cuda", "xpu"):
        return "bfloat16"
    return "float32"


def _load_model(cfg: Config, logger: logging.Logger):
    """加载 OmniVoice 模型（带全局缓存，避免重复加载）。"""
    global _OMNIVOICE_MODEL
    if _OMNIVOICE_MODEL is not None:
        return _OMNIVOICE_MODEL

    # MPS 设备先解除分配器内存上限（幂等；显式 --device mps 路径也在此生效）
    _apply_mps_memory_settings(cfg.device, logger)

    # CPU 设备下先把线程池配满再加载模型（幂等；web 复用此路径，自动生效）
    _configure_cpu_threads(cfg, logger)

    import torch
    from omnivoice import OmniVoice

    # 先经 resolve_path 下载/定位快照，再交给 from_pretrained：
    # 模型内部对本地目录直接复用，全程不触发额外网络请求
    resolved = resolve_path(cfg.model_id, cfg.model_path)
    logger.info("📦 模型目录: %s", resolved)

    # 默认 dtype：CUDA/XPU 用 bfloat16（GPU 原生加速），MPS/CPU 用 float32——
    # torch 2.13 在 MPS 上加载 float16 权重会 SIGTRAP 崩溃（已验证），且
    # HiggAudio tokenizer 不支持 MPS 会自动落 CPU，float32 更稳。可用 DTYPE
    # 环境变量显式覆盖（如 float16/float32/bfloat16）。
    if cfg.dtype and cfg.dtype not in _DTYPE_ALLOWED:
        raise ValueError(
            f"无效 DTYPE: {cfg.dtype}（可选: {', '.join(sorted(_DTYPE_ALLOWED))}）")
    dtype = getattr(torch, cfg.dtype) if cfg.dtype else getattr(
        torch, _default_dtype(cfg.device)
    )

    logger.info("⏳ 加载 OmniVoice 模型（%s, %s）…", cfg.device, dtype)
    t0 = time.time()
    try:
        _OMNIVOICE_MODEL = OmniVoice.from_pretrained(
            resolved, device_map=cfg.device, dtype=dtype,
        )
    except RuntimeError as e:
        if not _should_fallback_to_cpu(e, cfg.device):
            raise
        # MPS 内存不足（其它进程占用高时连小分配都失败）/ XPU 运行时失败：
        # 自动改用 CPU 重试
        failed_device = cfg.device.upper()
        cfg.device = "cpu"
        logger.warning("⚠️ %s 加载失败（%s），自动改用 CPU 加载 …",
                       failed_device, e)
        dtype = getattr(torch, cfg.dtype) if cfg.dtype else getattr(
            torch, _default_dtype("cpu")
        )
        _OMNIVOICE_MODEL = OmniVoice.from_pretrained(
            resolved, device_map="cpu", dtype=dtype,
        )
    logger.info("✓ 模型加载: %.1fs (%s)", time.time() - t0, cfg.device)
    return _OMNIVOICE_MODEL


def generate(cfg: Config, logger: logging.Logger, **kwargs):
    """model.generate() 包装：MPS/XPU 推理失败时自动改用 CPU 重载模型并重试一次。

    生成阶段若触发 MPS OOM（长文本峰值内存）或 XPU 运行时失败，清除全局
    模型缓存、以 CPU 重新加载（重载只发生一次，后续走缓存）后重试；CLI 与
    web 共用本函数。
    """
    model = _load_model(cfg, logger)
    try:
        return model.generate(**kwargs)
    except RuntimeError as e:
        if not _should_fallback_to_cpu(e, cfg.device):
            raise
        global _OMNIVOICE_MODEL
        failed_device = cfg.device.upper()
        _OMNIVOICE_MODEL = None  # 强制按 CPU 重载
        cfg.device = "cpu"
        logger.warning("⚠️ 生成时 %s 推理失败（%s），自动改用 CPU 重试 …",
                       failed_device, e)
        model = _load_model(cfg, logger)
        return model.generate(**kwargs)
