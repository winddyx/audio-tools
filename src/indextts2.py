"""
IndexTTS-2.5 模型核心（audiocpp `--family index_tts2`）

IndexTTS-2.5（IndexTeam/bilibili）多语言零样本 TTS：在 audio.cpp 中作为
`index_tts2` 族的 variant 2.5 实现（model config version 字段区分），
语音克隆走 `--task clon --voice-ref`。权重为 audio.cpp 专用 GGUF 包
（audio-cpp/audio.cpp-gguf，index-tts2_5-q8_0/f16/orig），本地
MODELS_DIR/IndexTTS2.5-GGUF/ 优先，缺失时经 HF 下载。

只做语音克隆（ref_audio 必填；参考文本转写省略时由 SenseVoice 提供）。
"""

from __future__ import annotations

import logging
import os
import tempfile

from .audiocpp import AudioResult, _backend_flag, _ensure_binary, ensure_tmp_dir, run_cli
from .config import (
    Config,
    GEN_SEED,
    INDEXTTS_TEMPERATURE,
    INDEXTTS_TOP_K,
    INDEXTTS_TOP_P,
    MODELS_DIR,
    TMP_DIR,
)
from .hf import _hf_download

# 模型族名（audiocpp --family 取值；2.0/2.5 共用 index_tts2，由模型 config 区分）
FAMILY = "index_tts2"

# 权重：本地路径优先；HF 仓库为下载兜底（默认 Q8_0 包，可换 f16/orig）
GGUF_LOCAL = os.path.join(MODELS_DIR, "IndexTTS2.5-GGUF", "index-tts2_5-q8_0.gguf")
GGUF_HF_REPO = "audio-cpp/audio.cpp-gguf"
GGUF_HF_FILE = "IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf"

# 仅支持 zh / en（IndexTTS2.5 默认语言集）
_SUPPORTED_LANGS = {"", "zh", "en"}

# 生成参数 → audiocpp CLI 参数映射（config 顶部常量；空值 = 不传 = 引擎默认）
# IndexTTS-2.5 的 gpt 层采样参数（引擎默认 top_k=30 / top_p=0.8 / temperature=0.8）
_OPT_MAP = {
    "top_k": ("--top-k", INDEXTTS_TOP_K),                    # int
    "top_p": ("--top-p", INDEXTTS_TOP_P),                    # float
    "temperature": ("--temperature", INDEXTTS_TEMPERATURE),  # float
    "seed": ("--seed", GEN_SEED),                            # int; -1 = 不传
}


def _gen_flags() -> list[str]:
    """把非空的生成参数常量拼成 CLI flag 列表（0/空/-1 = 引擎默认，不发）。"""
    flags: list[str] = []
    for k, (flag, v) in _OPT_MAP.items():
        if v is None or v == "":
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if k == "seed" and int(v) < 0:
                continue            # -1 = 随机
            if k == "top_k" and int(v) <= 0:
                continue            # 0 = 引擎默认
        flags += [flag, str(v)]
    return flags


def _ensure_model(logger: logging.Logger) -> str:
    """定位 IndexTTS2.5 GGUF：本地 MODELS_DIR 优先，缺失则 HF 下载。"""
    if os.path.isfile(GGUF_LOCAL):
        return GGUF_LOCAL
    logger.info("本地未找到 %s，从 HuggingFace 下载 %s/%s …",
                GGUF_LOCAL, GGUF_HF_REPO, GGUF_HF_FILE)
    return _hf_download(GGUF_HF_REPO, GGUF_HF_FILE)


def generate(cfg: Config, logger: logging.Logger, **kwargs) -> AudioResult:
    """IndexTTS-2.5 语音克隆：ref_audio（+ref_text 可选）→ 音频。

    kwargs 支持：text / language（zh|en）/ ref_audio / ref_text（可选）。
    生成参数 top_k / top_p / temperature / seed 走 config.py 顶部常量
    （空值 = 引擎默认）。输出采样率 22.05 kHz（以产出 wav 实际为准）。
    """
    text = kwargs.pop("text", "")
    language = kwargs.pop("language", None)
    ref_audio = kwargs.pop("ref_audio", None)
    ref_text = kwargs.pop("ref_text", None)
    if not ref_audio:
        raise ValueError("IndexTTS-2.5 语音克隆需要 ref_audio")
    ref_audio = os.path.abspath(ref_audio)
    if language and str(language).lower() not in _SUPPORTED_LANGS:
        logger.info("IndexTTS-2.5 仅支持 zh/en，忽略语言 %s", language)
        language = ""

    binary = _ensure_binary(logger)
    model = _ensure_model(logger)
    ensure_tmp_dir()

    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="indextts-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "clon", "--family", FAMILY,
               "--model", model, "--backend", _backend_flag(cfg.device),
               "--text", text, "--voice-ref", ref_audio, "--out", out_wav]
        if language:
            cmd += ["--language", str(language)]
        # 生成参数：config 顶部常量（0/空/-1 = 引擎默认，不发 flag）
        cmd += _gen_flags()
        if kwargs:
            logger.info("IndexTTS-2.5 不支持以下生成参数，已忽略: %s",
                        ", ".join(sorted(str(k) for k in kwargs)))
        run_cli(cmd, cfg.device, logger)
        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError("IndexTTS-2.5 未产出 WAV 文件")
        import soundfile as sf
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        return AudioResult(audio=data, sampling_rate=sr)
    finally:
        if os.path.exists(out_wav):
            try:
                os.remove(out_wav)
            except OSError:
                pass
