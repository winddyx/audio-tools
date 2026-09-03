"""
audio-tools — 核心包（src/）

平铺模块布局（无子包、无后端注册机制）：
- config：唯一设置源（Config 数据类 + 顶部可调变量 + 设备检测 + 日志压噪）
- audiocpp：推理引擎运行器（audio.cpp / audiocpp_cli，模型无关）
- omnivoice / indextts2：TTS 模型核心（语音克隆；TTS_MODEL 切换）
- sensevoice：ASR 核心（SenseVoice-Small，audiocpp sense_asr 族）
- hf：HuggingFace 下载与缓存管理（本地优先 + hf-mirror 兜底）
- pipeline：统一编排 synthesize()/draw()（ASR → TTS → 命名 → 写盘）

入口：vc.py（CLI）、web.py（Gradio Web），共用 src/。
"""

from .config import (
    Config,
    _quiet_hf_logs,
    _to_bool,
    get_best_device,
)
from .sensevoice import _transcribe_ref
from .audiocpp import AudioResult, ChunkInfo
from .hf import _HF_MIRROR, _hf_download, _switch_hf_endpoint, resolve_path
from .pipeline import SynthesisResult, draw, synthesize

__all__ = [
    "Config", "_to_bool", "_quiet_hf_logs", "get_best_device",
    "_transcribe_ref",
    "AudioResult", "ChunkInfo",
    "_HF_MIRROR", "_hf_download", "_switch_hf_endpoint", "resolve_path",
    "SynthesisResult", "synthesize", "draw",
]
