"""
OmniVoice 模型核心（audiocpp `--family omnivoice`）

k2-fsa/OmniVoice 的多语言零样本 TTS（600+ 语言），在 audio.cpp 中支持
语音克隆（`--voice-ref` + `--reference-text`）。权重为 audio.cpp 专用单文件
GGUF 包（audio-cpp/audio.cpp-gguf，omnivoice-bf16/f16/q8_0）。模型文件只在
HF 默认缓存（~/.cache/huggingface/hub）：缺失时自动经 HF 下载并在缓存内生成
引擎可用的 .gguf 别名（见 hf._ensure_gguf_file）；项目 models/ 仅支持手工放置。

只做语音克隆（ref_audio + ref_text 必填）；声音设计（design/instruct）不在
本版本范围内。
"""

from __future__ import annotations

import logging
import os
import tempfile

from .audiocpp import (
    AudioResult,
    _chunk_flags,
    _ensure_binary,
    ensure_tmp_dir,
    run_cli,
)
from .config import (
    Config,
    GEN_SEED,
    MODELS_DIR,
    OMNI_GUIDANCE_SCALE,
    OMNI_INFERENCE_STEPS,
    TMP_DIR,
)
from .hf import _ensure_gguf_file

# 模型族名（audiocpp --family 取值）
FAMILY = "omnivoice"

# 权重：项目 models/ 仅支持手工放置；默认放 HF 缓存（引擎须真实 .gguf 路径）
GGUF_LOCAL = os.path.join(MODELS_DIR, "OmniVoice-GGUF", "omnivoice-q8_0.gguf")
GGUF_HF_REPO = "audio-cpp/audio.cpp-gguf"
GGUF_HF_FILE = "OmniVoice-GGUF/omnivoice-q8_0.gguf"

# 生成参数 → audiocpp CLI 参数映射（每调用 kwargs 优先，其次 config 顶部常量；
# 其余未知参数忽略并提示）
_OPT_MAP = {
    "num_inference_steps": ("--num-inference-steps", OMNI_INFERENCE_STEPS),  # int
    "guidance_scale": ("--guidance-scale", OMNI_GUIDANCE_SCALE),             # float
    "seed": ("--seed", GEN_SEED),                                            # int; -1 = 不传
}


def _opt_value(k: str, kwargs: dict) -> str | None:
    """取某生成参数值：调用方 kwargs 优先；无则 config 顶部常量；空/0/-1 = 不传。"""
    flag, default = _OPT_MAP[k]
    if k in kwargs and kwargs[k] is not None:
        v = kwargs[k]
    else:
        v = default
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if k == "seed" and int(v) < 0:
            return None          # -1 = 随机
        if k == "num_inference_steps" and int(v) <= 0:
            return None          # 0 = 引擎默认
        return str(v)
    return str(v)


def _ensure_model(logger: logging.Logger) -> str:
    """定位 OmniVoice GGUF：手工放置的本地文件优先，否则经 HF 下载。

    audio.cpp 按真实文件扩展名识别 GGUF，HF 缓存 blob/软链路径不能直接用，
    _ensure_gguf_file 会在 HF 默认缓存仓库目录内生成带 .gguf 的硬链接别名并
    返回。模型不落工程目录（GGUF_LOCAL 仅支持用户手工放置）。
    """
    if os.path.isfile(GGUF_LOCAL):
        return GGUF_LOCAL
    return _ensure_gguf_file(GGUF_HF_REPO, GGUF_HF_FILE, logger)


def generate(cfg: Config, logger: logging.Logger, **kwargs) -> AudioResult:
    """OmniVoice 语音克隆：ref_audio + ref_text → 音频。

    kwargs 支持：text / language / ref_audio / ref_text / 生成参数
    （num_inference_steps / guidance_scale / seed，kwargs 优先，缺省用
    config.py 顶部常量；空值 = 引擎默认）。输出采样率 24 kHz（以产出
    wav 实际为准）。
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
        # 长文本分块（config.TEXT_CHUNK_SIZE/MODE，自动选 endline/default）
        cmd += _chunk_flags(text)
        if language:
            cmd += ["--language", str(language)]
        # 生成参数：kwargs 优先 → config 顶部常量兜底；空值不发 flag
        for k in _OPT_MAP:
            v = _opt_value(k, kwargs)
            if v is not None:
                cmd += [_OPT_MAP[k][0], v]
        ignored = set(kwargs) - set(_OPT_MAP) - {"text", "language",
                                                 "ref_audio", "ref_text"}
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
