"""数据契约：核心层与引擎/界面之间的类型层（无实现逻辑）。

约定：
- 一切穿过编排层的数据都走这里定义的类型；界面/引擎不允许发明新字段。
- 音频一律 mono float32 PCM，采样率随引擎（OmniVoice 输出 24 kHz）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class Segment:
    """一段生成文本（长文本分块结果 / single-shot 整篇单块）。

    按段重生成时直接取 text 重新调用即可；文本与音频对齐由上游引擎负责。
    """

    text: str


@dataclass(frozen=True)
class GenParams:
    """通用生成参数（采样/时长等）。引擎只取自己支持的白名单子集。

    各字段 None 表示"用引擎默认值"；具体支持哪些由引擎声明。
    """

    num_step: Optional[int] = None
    denoise: Optional[bool] = None
    audio_chunk_duration: Optional[float] = None
    audio_chunk_threshold: Optional[float] = None
    duration: Optional[float] = None
    # 供未来模型扩展：引擎不认识的 key 会被忽略并记录
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynthesizeRequest:
    """一次 TTS 合成请求（内容数据；运行设置走 settings 顶部变量）。"""

    text: str
    language: str = ""                       # 留空 = 引擎自动判断
    ref_audio: str = ""                      # 语音克隆参考音频路径
    ref_text: str = ""                       # 参考音频转写文本；留空且给 ref_audio 时自动 ASR
    instruct: str = ""                       # 声音设计指令（非空且无 ref_audio 时生效）
    params: GenParams = field(default_factory=GenParams)


@dataclass(frozen=True)
class TranscribeRequest:
    """一次 ASR 转写请求。"""

    audio: str                               # 参考音频路径


@dataclass
class EngineResult:
    """引擎原始产物（未落盘）。"""

    audio: np.ndarray                        # mono float32 PCM
    sampling_rate: int
    segments: list[Segment] = field(default_factory=list)


@dataclass
class SynthesisOutcome:
    """编排层合成产物：音频 + 输出文件 + 元数据。"""

    audio: np.ndarray
    sampling_rate: int
    out_path: str = ""                       # save 后为 WAV 绝对路径（未落盘时为空）
    duration_sec: float = 0.0
    ref_text: str = ""                       # 本次由 ASR 转写出的参考文本（非 ASR 来源时为空）
    segments: list[Segment] = field(default_factory=list)


class Engine(Protocol):
    """引擎适配器协议（核心只依赖此形态；实现位于 ov/models/<name>/）。

    实现需保证可重入（Web 多会话并发调用），无后台常驻进程。
    """

    spec_id: str

    def provision(self, logger: Any) -> None:
        """确保运行时就绪（二进制已编译、权重已下载）；已有则零开销。"""
        ...

    def synthesize(self, req: SynthesizeRequest, logger: Any) -> EngineResult:
        """合成音频。语音克隆缺 ref_text 时抛 ValueError（由编排层先 ASR）。"""
        ...

    def transcribe(self, req: TranscribeRequest, logger: Any) -> str:
        """转写音频返回纯文本（仅 ASR 类模型实现）。"""
        raise NotImplementedError
