"""
统一编排层：vc.py / web.py 共用的完整合成流程。

synthesize(): ASR 转写（clone 且缺 ref_text 时）→ gguf.generate → 输出命名
              → 写 WAV → SynthesisResult（含 out_path / 分段信息 / ASR 文本）。
draw():       连续合成 N 次（抽卡），返回结果列表。

输出命名规则（唯一事实来源）：
    <out_dir>/<out_name>.<unix秒时间戳>.wav；同秒冲突时秒数递增。
CLI 传文本文件名，Web 传启动时间戳（与旧行为兼容）。

vc.py / web.py 不再直接调用后端 generate / ASR，统一走本模块。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .config import Config
from .funasr import _transcribe_ref
from .gguf import ChunkInfo, generate, _load_model as _gguf_load_model

# 参考音频转写文本缓存（key: 绝对路径+大小+mtime+asr_model）
# 同一参考音频在多轮抽卡间只转写一次
_ASR_TEXT_CACHE: dict[tuple, str] = {}


@dataclass
class SynthesisResult:
    """一次完整合成的产物（音频 + 输出文件 + 元数据）。"""

    audio: np.ndarray          # mono float32 PCM（24 kHz）
    sampling_rate: int
    out_path: str              # 已写入的 WAV 文件绝对路径
    duration_sec: float
    ref_text: str = ""         # 本次由 ASR 转写出的参考文本（用户提供时为空）
    chunks: list[ChunkInfo] = field(default_factory=list)


def _unique_out_path(out_dir: str, out_name: str) -> str:
    """<out_name>.<unix 秒时间戳>.wav；同秒冲突则秒数递增，避免覆盖。"""
    ts = int(time.time())
    path = os.path.join(out_dir, f"{out_name}.{ts}.wav")
    while os.path.exists(path):
        ts += 1
        path = os.path.join(out_dir, f"{out_name}.{ts}.wav")
    return path


def synthesize(
    cfg: Config,
    logger: logging.Logger,
    *,
    text: str,
    language: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    out_dir: str,
    out_name: str,
    gen_kwargs: Optional[dict] = None,
) -> SynthesisResult:
    """一次完整合成：ASR（clone 且缺 ref_text）→ generate → 写 WAV。

    返回 SynthesisResult；输出 WAV 由本函数写入 out_dir/out_name 命名。
    ref_text 由调用方传入时不重复转写；缺省且需要时自动用 SenseVoice
    转写（结果记录在 result.ref_text，供 UI 展示）。
    """
    # 唯一后端：GGUF（C++/GGML，src/gguf.py）。pipeline 不再经注册机制
    model = _gguf_load_model(cfg, logger)

    asr_out = ""
    if ref_audio and not ref_text:
        # 参考音频转写缓存：同一音频（路径+大小+mtime+asr_model 均一致）只转写
        # 一次。web 抽卡每轮都走 synthesize 且不传 ref_text，若无缓存每轮都会
        # 重跑一遍 SenseVoice 子进程（vc 已前置转写并传入，天然命中）
        asr_cfg = replace(cfg, ref_audio=ref_audio)
        key = (os.path.abspath(ref_audio),
               os.path.getsize(ref_audio),
               os.path.getmtime(ref_audio),
               os.path.abspath(asr_cfg.asr_model) if asr_cfg.asr_model else "")
        ref_text = _ASR_TEXT_CACHE.get(key)
        if ref_text is None:
            logger.info("ref_text 未提供，用 SenseVoiceSmall-GGUF 转写参考音频 …")
            ref_text = _transcribe_ref(asr_cfg, logger)
            if not ref_text:
                raise ValueError("ASR 转写结果为空（参考音频可能为静音）")
            _ASR_TEXT_CACHE[key] = ref_text
        asr_out = ref_text
        logger.info("参考文本: %s", ref_text)

    result = generate(
        cfg, logger,
        text=text.strip(),
        language=language or None,
        ref_audio=ref_audio or None,
        ref_text=ref_text if ref_audio else None,
        instruct=instruct or None,
        **(gen_kwargs or {}),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = _unique_out_path(out_dir, out_name)

    import soundfile as sf
    sf.write(out_path, result.audio, result.sampling_rate)

    return SynthesisResult(
        audio=result.audio,
        sampling_rate=result.sampling_rate,
        out_path=out_path,
        duration_sec=(len(result.audio) / result.sampling_rate
                      if len(result.audio) else 0.0),
        ref_text=asr_out,
        chunks=result.chunks,
    )


def draw(cfg: Config, logger: logging.Logger, count: int, **kwargs) -> list[SynthesisResult]:
    """连续合成 count 次（抽卡），返回结果列表。"""
    return [synthesize(cfg, logger, **kwargs) for _ in range(count)]
