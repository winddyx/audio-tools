"""
OmniVoice 配音工具 — 兼容层（transformers 后端聚合导出）

核心逻辑已拆分到 src/ 包（config / logs / device / hf / params / asr /
backends）。本文件保留仅为兼容历史 `from omni import ...` 的调用方，
等价于旧单文件 omni.py（transformers 后端）。

新代码请从 src 导入；后端切换统一在 settings.py（BACKEND 变量 / OMNIVOICE_BACKEND 环境变量）：
- "gguf"（默认）：C++/GGML 推理（Serveurperso/OmniVoice-GGUF BF16）
- "transformers"：本文件背后的 k2-fsa/OmniVoice 实现
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
from src.asr import (
    _asr_hub,
    _asr_language,
    _asr_model,
    _asr_model_id,
    _asr_vad_id,
    _clean_asr_text,
    _transcribe_ref,
)

# transformers 后端（等价于旧 omni.py 的 TTS 实现）
from src.backends.transformers import (
    _DTYPE_ALLOWED,
    _default_dtype,
    _load_model,
    generate,
)

__all__ = [
    "Config", "_to_bool", "get_best_device",
    "_is_mps_oom", "_should_fallback_to_cpu",
    "_apply_mps_memory_settings", "_configure_cpu_threads",
    "_HF_MIRROR", "_hf_download", "_switch_hf_endpoint", "resolve_path",
    "_quiet_hf_logs", "_GEN_PARAM_ENVS", "_gen_kwargs",
    "_asr_hub", "_asr_language", "_asr_model", "_asr_model_id",
    "_asr_vad_id", "_clean_asr_text", "_transcribe_ref",
    "_DTYPE_ALLOWED", "_default_dtype", "_load_model", "generate",
]
