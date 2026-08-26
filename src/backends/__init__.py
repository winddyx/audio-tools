"""
推理后端注册与选择。

每个后端模块（transformers / gguf）提供一致的接口：
    _load_model(cfg, logger) → 模型对象/句柄（.sampling_rate）
    generate(cfg, logger, **kwargs) → [音频数组]
"""

from __future__ import annotations

BACKEND_IDS = ("gguf", "transformers")


def get_backend(name: str):
    """按名称返回后端模块（懒加载，避免未选中的后端被 import 拉入重依赖）。"""
    if name == "gguf":
        from . import gguf
        return gguf
    if name == "transformers":
        from . import transformers
        return transformers
    raise ValueError(f"未知推理后端: {name}（可选: {' / '.join(BACKEND_IDS)}）")
