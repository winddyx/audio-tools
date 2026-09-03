"""模型注册：ModelSpec + 注册表。

核心只认识 ModelSpec 与 Engine 协议，不认识任何具体模型。
具体模型在 ov/models/<name>/ 声明 spec 并 register()；ov.models 包被
导入一次即完成注册（入口经 ov.api 引用，天然触发）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet

from .types import Engine


@dataclass(frozen=True)
class ModelSpec:
    """一个生成模型的完整描述（自包含，未来模型照此新增）。"""

    id: str
    kind: str                          # "tts" / "asr" / 未来扩展
    description: str
    capabilities: FrozenSet[str]       # 见 ov/models/*/spec.py 注释
    supported_params: FrozenSet[str]   # GenParams 中本引擎支持的字段
    engine: Engine                     # 已实例化的引擎适配器（可重入、无状态）
    # 长文本兜底分块阈值：无 native_longform 能力时超过即逐段合成
    fallback_chunk_chars: int = 200


_MODELS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.id in _MODELS:
        raise ValueError(f"模型重复注册: {spec.id}")
    _MODELS[spec.id] = spec
    return spec


def get_model(model_id: str) -> ModelSpec:
    try:
        return _MODELS[model_id]
    except KeyError:
        raise ValueError(
            f"未知模型: {model_id}（已注册: {', '.join(sorted(_MODELS))}）"
        ) from None


def models() -> list[ModelSpec]:
    return list(_MODELS.values())
