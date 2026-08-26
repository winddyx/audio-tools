"""
OmniVoice 配音工具 — 核心包（src/）

共享核心（与推理后端无关）：
- config：Config 数据类 + _to_bool
- logs：HF 相关第三方库日志降噪（_quiet_hf_logs）
- device：设备检测与容错（get_best_device / mps、xpu 回退 CPU）
- hf：HuggingFace 下载与缓存管理（本地优先 + hf-mirror 兜底）
- params：生成参数环境变量映射（_GEN_PARAM_ENVS / _gen_kwargs）
- asr：参考音频转写（FunASR/SenseVoiceSmall）

推理后端（backends/，接口一致，由 settings.BACKEND 选择）：
- backends.gguf：C++/GGML（omnivoice.cpp，Serveurperso/OmniVoice-GGUF BF16）
- backends.transformers：k2-fsa/OmniVoice（transformers）

旧入口 omni.py 现为兼容 shim，聚合导出 transformers 后端与共享核心。
"""

from .backends import BACKEND_IDS, get_backend
from .config import Config, _to_bool
from .device import get_best_device
from .logs import _quiet_hf_logs
from .params import _GEN_PARAM_ENVS, _gen_kwargs
from .asr import _transcribe_ref

__all__ = [
    "Config", "_to_bool", "_quiet_hf_logs", "get_best_device",
    "_GEN_PARAM_ENVS", "_gen_kwargs", "_transcribe_ref",
    "get_backend", "BACKEND_IDS",
]
