"""统一入口门面：CLI / Web / 脚本只依赖 ov.api，不触碰引擎细节。

职责：
- 模型解析（注册表）、日志初始化、ASR 预转写（克隆缺 ref_text 时）
- 输出目录/文件名的默认取值
- settings 顶部变量 → GenParams（CLI 用）
"""

from __future__ import annotations

import logging
from typing import Optional

from . import logs, pipeline, settings
from .model import get_model
from .types import GenParams, SynthesizeRequest, SynthesisOutcome, \
    TranscribeRequest


def _logger() -> logging.Logger:
    level = getattr(logging, str(settings.LOG_LEVEL).upper(),
                    logging.INFO)
    return logs.setup(level)


def default_params() -> GenParams:
    """settings 顶部变量（NUM_STEP 等，None=引擎默认）→ GenParams。"""
    return GenParams(
        num_step=settings.NUM_STEP,
        denoise=settings.DENOISE,
        audio_chunk_duration=settings.AUDIO_CHUNK_DURATION,
        audio_chunk_threshold=settings.AUDIO_CHUNK_THRESHOLD,
        duration=settings.DURATION,
    )


def synthesize(
    text: str,
    *,
    model_id: str = "omnivoice",
    ref_audio: str = "",
    ref_text: str = "",
    instruct: str = "",
    language: str = "",
    params: Optional[GenParams] = None,
    out_dir: str = "",
    out_name: str = "output",
    logger: Optional[logging.Logger] = None,
) -> SynthesisOutcome:
    """一次完整 TTS 合成（自动音色/声音设计/语音克隆按入参区分）。

    语音克隆且缺 ref_text 时，自动用 settings.ASR_ENGINE_ID 转写参考音频。
    """
    logger = logger or _logger()
    spec = get_model(model_id)
    engine = spec.engine

    asr_text = ""
    if ref_audio and not ref_text:
        asr_spec = get_model(settings.ASR_ENGINE_ID)
        logger.info("ref_text 未提供，用 %s 转写参考音频 …", asr_spec.id)
        asr_text = pipeline.transcribe(
            asr_spec.engine, TranscribeRequest(audio=ref_audio), logger)
        logger.info("参考文本: %s", asr_text)
        if not asr_text:
            raise ValueError("ASR 转写结果为空（参考音频可能为静音）")
        ref_text = asr_text

    req = SynthesizeRequest(
        text=text.strip(),
        language=language or "",
        ref_audio=ref_audio or "",
        ref_text=ref_text if ref_audio else "",
        instruct=instruct or "",
        params=params or GenParams(),
    )
    out_dir = out_dir or (settings.OUTPUT_DIR
                          or str(settings.project_root() / "out"))
    return pipeline.complete(spec, engine, req, out_dir, out_name,
                             logger, asr_text=asr_text)


def transcribe(
    audio: str,
    *,
    model_id: str = "",
    logger: Optional[logging.Logger] = None,
) -> str:
    """ASR 转写（默认 settings.ASR_ENGINE_ID），返回纯文本。"""
    logger = logger or _logger()
    model_id = model_id or settings.ASR_ENGINE_ID
    spec = get_model(model_id)
    return pipeline.transcribe(spec.engine,
                               TranscribeRequest(audio=audio), logger)


def models_info() -> list[dict]:
    """模型清单（Web/CLI 展示用）。"""
    from .model import models

    return [
        {"id": m.id, "kind": m.kind, "description": m.description,
         "capabilities": sorted(m.capabilities)}
        for m in models()
    ]
