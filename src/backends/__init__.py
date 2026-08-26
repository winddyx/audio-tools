"""
推理后端注册与选择。

只保留 GGUF 后端（C++/GGML：omnivoice.cpp + Serveurperso/OmniVoice-GGUF）。
接口：
    _load_model(cfg, logger) → 模型句柄（.sampling_rate = 24000）
    generate(cfg, logger, **kwargs) → [音频数组]
"""

from __future__ import annotations

BACKEND_IDS = ("gguf",)


def get_backend(name: str):
    """按名称返回后端模块（懒加载）。"""
    if name == "gguf":
        from . import gguf
        return gguf
    raise ValueError(f"未知推理后端: {name}（可选: {' / '.join(BACKEND_IDS)}）")
