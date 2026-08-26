"""
OmniVoice 配音工具 — 兼容层（共享核心 + GGUF 后端聚合导出）

核心逻辑在 src/ 包（config / logs / device / hf / params / asr / backends）。
推理后端只保留 GGUF（src/backends/gguf.py：omnivoice.cpp + Serveurperso/
OmniVoice-GGUF）；ASR 用 SenseVoiceSmall-GGUF（src/asr.py，FunASR llama.cpp
runtime）。

本文件保留仅为兼容历史 `from omni import ...` 的调用方，等价于旧单文件
omni.py（GGUF 后端）。新代码请从 src 导入；可调设置统一在 settings.py。
"""

from __future__ import annotations

# 共享核心
from src.config import Config, _to_bool
from src.device import (
    _apply_mps_memory_settings,
    _configure_cpu_threads,
    _is_mps_oom,
    _should_fallback_to_cpu,
    get_best_device,
)
from src.hf import _HF_MIRROR, _hf_download, _switch_hf_endpoint, resolve_path
from src.logs import _quiet_hf_logs
from src.params import _GEN_PARAM_ENVS, _gen_kwargs
from src.asr import _transcribe_ref

# GGUF 推理后端（项目唯一后端）
from src.backends.gguf import _load_model, generate

__all__ = [
    "Config", "_to_bool", "get_best_device",
    "_is_mps_oom", "_should_fallback_to_cpu",
    "_apply_mps_memory_settings", "_configure_cpu_threads",
    "_HF_MIRROR", "_hf_download", "_switch_hf_endpoint", "resolve_path",
    "_quiet_hf_logs", "_GEN_PARAM_ENVS", "_gen_kwargs",
    "_transcribe_ref", "_load_model", "generate",
]
