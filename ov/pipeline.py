"""编排层：与具体模型无关的合成/转写/落盘流程（唯一实现处）。

CLI / Web 一律经 ov.api 到达本层；ASR 转写、输出命名、写 WAV、
长文本兜底分块只在这里实现一次。引擎有 native_longform 能力时
长文本走引擎原生分块，否则用 ov.text 的 Python 兜底逐段合成拼接。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .audio import concat, write_wav
from .model import ModelSpec
from .text import chunk_by_length
from .types import Engine, EngineResult, Segment, SynthesizeRequest, \
    SynthesisOutcome, TranscribeRequest


def unique_out_path(out_dir: str, out_name: str) -> str:
    """<out_dir>/<out_name>.<unix 秒>.wav；同秒冲突秒数递增，绝不覆盖。"""
    import os

    ts = int(time.time())
    path = os.path.join(out_dir, f"{out_name}.{ts}.wav")
    while os.path.exists(path):
        ts += 1
        path = os.path.join(out_dir, f"{out_name}.{ts}.wav")
    return path


def transcribe(engine: Engine, req: TranscribeRequest,
               logger: logging.Logger) -> str:
    """ASR 转写（SenseVoice 类引擎）。"""
    return engine.transcribe(req, logger)


def _native_or_short(spec: ModelSpec, req: SynthesizeRequest) -> bool:
    if "native_longform" in spec.capabilities:
        return True
    return len(req.text) <= spec.fallback_chunk_chars


def _fallback_long(engine: Engine, spec: ModelSpec, req: SynthesizeRequest,
                   logger: logging.Logger) -> EngineResult:
    """无原生分块能力的长文本：按句分块逐段合成后拼接（模板兜底路径）。"""
    chunks = chunk_by_length(req.text, spec.fallback_chunk_chars)
    logger.info("长文本分块: %d 段（Python 兜底逐段合成）", len(chunks))
    audios: list = []
    sr = 0
    segments: list[Segment] = []
    for i, piece in enumerate(chunks, 1):
        logger.info("[%d/%d] 段合成中 …", i, len(chunks))
        part = engine.synthesize(
            SynthesizeRequest(text=piece, language=req.language,
                              ref_audio=req.ref_audio, ref_text=req.ref_text,
                              instruct=req.instruct, params=req.params),
            logger)
        if not sr and part.sampling_rate:
            sr = part.sampling_rate
        audios.append(part.audio)
        segments.append(Segment(text=piece))
    if not audios:
        raise RuntimeError("长文本分块合成未产出音频")
    gap = int(sr * 0.15) if sr else 0   # 段间 150 ms 停顿
    return EngineResult(audio=concat(audios, gap), sampling_rate=sr,
                        segments=segments)


def complete(
    spec: ModelSpec,
    engine: Engine,
    req: SynthesizeRequest,
    out_dir: str,
    out_name: str,
    logger: Optional[logging.Logger] = None,
    asr_text: str = "",
) -> SynthesisOutcome:
    """一次完整合成：引擎生成（含长文本策略）→ 命名 → 写 WAV。

    语音克隆缺 ref_text 的 ASR 预转写由 ov.api 负责（此处假定已就绪）。
    """
    logger = logger or logging.getLogger("ov")
    if _native_or_short(spec, req):
        result = engine.synthesize(req, logger)
    else:
        result = _fallback_long(engine, spec, req, logger)

    import os
    os.makedirs(out_dir, exist_ok=True)
    path = unique_out_path(out_dir, out_name)
    write_wav(path, result.audio, result.sampling_rate)
    return SynthesisOutcome(
        audio=result.audio,
        sampling_rate=result.sampling_rate,
        out_path=path,
        duration_sec=(len(result.audio) / result.sampling_rate
                      if len(result.audio) else 0.0),
        ref_text=asr_text,
        segments=result.segments,
    )
