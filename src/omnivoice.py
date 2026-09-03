"""
OmniVoice 模型核心（audiocpp `--family omnivoice`）

k2-fsa/OmniVoice 的多语言零样本 TTS（600+ 语言），在 audio.cpp 中支持
语音克隆（`--voice-ref` + `--reference-text`）。权重为 audio.cpp 专用单文件
GGUF 包（audio-cpp/audio.cpp-gguf，omnivoice-bf16/f16/q8_0），本地
MODELS_DIR/OmniVoice-GGUF/ 优先，缺失时经 HF 下载。

只做语音克隆（ref_audio + ref_text 必填）；声音设计（design/instruct）不在
本版本范围内。
"""

from __future__ import annotations

import logging
import os
import tempfile

from .audiocpp import AudioResult, _ensure_binary, ensure_tmp_dir, run_cli
from .config import Config, MODELS_DIR, TMP_DIR
from .hf import _hf_download

# 模型族名（audiocpp --family 取值）
FAMILY = "omnivoice"

# 权重：本地路径优先；HF 仓库为下载兜底
GGUF_LOCAL = os.path.join(MODELS_DIR, "OmniVoice-GGUF", "omnivoice-bf16.gguf")
GGUF_HF_REPO = "audio-cpp/audio.cpp-gguf"
GGUF_HF_FILE = "OmniVoice-GGUF/omnivoice-bf16.gguf"

# 生成参数 → audiocpp CLI 参数映射（支持子集；其余忽略并提示）
_OPT_MAP = {
    "num_inference_steps": "--num-inference-steps",   # int
    "guidance_scale": "--guidance-scale",             # float
}


def _ensure_model(logger: logging.Logger) -> str:
    """定位 OmniVoice GGUF：本地 MODELS_DIR 优先，缺失则 HF 下载。"""
    if os.path.isfile(GGUF_LOCAL):
        return GGUF_LOCAL
    logger.info("本地未找到 %s，从 HuggingFace 下载 %s/%s …",
                GGUF_LOCAL, GGUF_HF_REPO, GGUF_HF_FILE)
    return _hf_download(GGUF_HF_REPO, GGUF_HF_FILE)


def generate(cfg: Config, logger: logging.Logger, **kwargs) -> AudioResult:
    """OmniVoice 语音克隆：ref_audio + ref_text → 音频。

    kwargs 支持：text / language / ref_audio / ref_text /
    num_inference_steps / guidance_scale（GGUF 后端参数名子集）。
    输出采样率 24 kHz（以产出 wav 实际为准）。
    """
    text = kwargs.pop("text", "")
    language = kwargs.pop("language", None)
    ref_audio = kwargs.pop("ref_audio", None)
    ref_text = kwargs.pop("ref_text", None)
    if not ref_audio or not ref_text:
        raise ValueError(
            "OmniVoice 语音克隆需要 ref_audio 与 ref_text"
            "（参考音频转写文本；CLI/web 会自动用 SenseVoice 转写）")
    ref_audio = os.path.abspath(ref_audio)

    binary = _ensure_binary(logger)
    model = _ensure_model(logger)
    ensure_tmp_dir()

    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="omni-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "tts", "--family", FAMILY,
               "--model", model, "--backend", _backend_of(cfg),
               "--text", text, "--voice-ref", ref_audio,
               "--reference-text", ref_text, "--out", out_wav]
        if language:
            cmd += ["--language", str(language)]
        for k, flag in _OPT_MAP.items():
            if k in kwargs and kwargs[k] is not None:
                cmd += [flag, str(kwargs[k])]
        ignored = set(kwargs) - set(_OPT_MAP)
        if ignored:
            logger.info("OmniVoice 不支持以下生成参数，已忽略: %s",
                        ", ".join(sorted(str(x) for x in ignored)))
        run_cli(cmd, cfg.device, logger)
        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError("OmniVoice 未产出 WAV 文件")
        import soundfile as sf
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        return AudioResult(audio=data, sampling_rate=sr)
    finally:
        if os.path.exists(out_wav):
            try:
                os.remove(out_wav)
            except OSError:
                pass


def _backend_of(cfg: Config) -> str:
    from .audiocpp import _backend_flag
    return _backend_flag(cfg.device)
