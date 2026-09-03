"""stub 引擎定义与注册（导入一次即注册；不加载真实模型/网络）。"""

from __future__ import annotations

import numpy as np

from ov.model import ModelSpec, register
from ov.types import EngineResult, Segment, SynthesizeRequest, TranscribeRequest


class StubTTS:
    """可重入 stub TTS：每次合成返回 0.25 s 静音，整段单 segment。"""

    spec_id = "stub-tts"

    def __init__(self) -> None:
        self.calls = 0

    def provision(self, logger) -> None:
        pass

    def synthesize(self, req: SynthesizeRequest, logger) -> EngineResult:
        self.calls += 1
        n = int(8000 * 0.25)
        return EngineResult(
            audio=np.zeros(n, dtype=np.float32), sampling_rate=8000,
            segments=[Segment(text=req.text)])

    def transcribe(self, req, logger) -> str:
        raise NotImplementedError


class StubNativeTTS(StubTTS):
    """带 native_longform 能力的 stub（分块在"引擎内"完成）。"""

    spec_id = "stub-native"


class StubASR:
    spec_id = "stub-asr"

    def provision(self, logger) -> None:
        pass

    def synthesize(self, req, logger):
        raise NotImplementedError

    def transcribe(self, req: TranscribeRequest, logger) -> str:
        return "stub 参考文本"


_stub_tts = StubTTS()
_stub_native = StubNativeTTS()
_stub_asr = StubASR()

register(ModelSpec(
    id=_stub_tts.spec_id, kind="tts",
    description="stub tts（无原生分块，走 Python 兜底）",
    capabilities=frozenset({"clone", "design", "auto"}),
    supported_params=frozenset(),
    engine=_stub_tts,
    fallback_chunk_chars=20,
))
register(ModelSpec(
    id=_stub_native.spec_id, kind="tts",
    description="stub native tts",
    capabilities=frozenset({"clone", "design", "auto", "native_longform"}),
    supported_params=frozenset(),
    engine=_stub_native,
))
register(ModelSpec(
    id=_stub_asr.spec_id, kind="asr",
    description="stub asr",
    capabilities=frozenset({"transcribe"}),
    supported_params=frozenset(),
    engine=_stub_asr,
))
