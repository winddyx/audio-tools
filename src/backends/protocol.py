"""
推理后端公共契约：结果结构与接口形态（类型层，无实现）。

后端模块（当前唯一实现 src/backends/gguf.py）需满足的模块级接口：

    _load_model(cfg, logger) -> ModelHandle   # 句柄至少带 .sampling_rate = 24000
    generate(cfg, logger, **kwargs) -> AudioResult

generate() 的 kwargs：text / language / ref_audio / ref_text / instruct，
以及 src/params.py 中该后端支持的生成参数子集。返回 AudioResult（含音频与
分块元数据），不再返回裸 [音频数组]。cli.py / web.py 一律经 src/pipeline.py
调用，不直接触碰后端。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ChunkInfo:
    """一段合成文本。

    由 C++ 分块器（text-chunker.h）给出的分段结果：句末标点 + 换行等规则
    切出的每一块。single-shot 路径下为整篇文本单块。按段重生成时直接取
    text 重新调用即可。
    """

    text: str


@dataclass
class AudioResult:
    """一次合成的完整结果（音频 + 元数据）。"""

    audio: np.ndarray  # mono float32 PCM
    sampling_rate: int
    chunks: list[ChunkInfo] = field(default_factory=list)


class ModelHandle(Protocol):
    """模型句柄的最小契约（各后端实现可附带额外字段）。"""

    sampling_rate: int


class Backend(Protocol):
    """后端模块的接口形态（供类型标注与测试替身参考）。"""

    def _load_model(self, cfg: Any, logger: Any) -> ModelHandle: ...

    def generate(self, cfg: Any, logger: Any, **kwargs: Any) -> AudioResult: ...
