"""
OmniVoice 配音工具 — 核心包（src/）

平铺模块布局（无子包、无后端注册机制）：
- config：唯一设置源（Config 数据类 + 顶部可调变量 + 设备检测 + 日志压噪）
- hf：HuggingFace 下载与缓存管理（本地优先 + hf-mirror 兜底）
- funasr：参考音频转写（SenseVoiceSmall-GGUF，FunASR llama.cpp runtime）
- omni：生成参数环境变量映射（_GEN_PARAM_ENVS / _gen_kwargs）+ Web 语言表
  （LANG_NAME_TO_ID）+ 旧 `from omni import ...` 兼容导出面
- pipeline：统一编排 synthesize()/draw()（ASR → gguf.generate → 命名 → 写盘）
- gguf：唯一推理后端（C++/GGML，omnivoice.cpp + Serveurperso/OmniVoice-GGUF）

入口：vc.py（CLI）、web.py（Gradio Web），共用 src/。
"""

from .config import (
    Config,
    _quiet_hf_logs,
    _to_bool,
    get_best_device,
)
from .funasr import _transcribe_ref
from .gguf import generate
from .gguf import _load_model as _load_model
from .hf import _HF_MIRROR, _hf_download, _switch_hf_endpoint, resolve_path
from .omni import (
    LANG_NAME_TO_ID,
    _GEN_PARAM_ENVS,
    _gen_kwargs,
    lang_display_name,
)

__all__ = [
    "Config", "_to_bool", "_quiet_hf_logs", "get_best_device",
    "_transcribe_ref", "generate", "_load_model",
    "_HF_MIRROR", "_hf_download", "_switch_hf_endpoint", "resolve_path",
    "_GEN_PARAM_ENVS", "_gen_kwargs",
    "LANG_NAME_TO_ID", "lang_display_name",
]