"""
Fish Audio S2-Pro 模型核心（audiocpp `--family fish_audio`）

Fish Audio S2-Pro：参考语音零样本克隆 TTS，audio.cpp 中以 `fish_audio`
族实现（variant s2-pro）：克隆走 `--task tts` + `--voice-ref`（不是 clon，
同 OmniVoice/Qwen3-TTS），`--reference-text` 可选；语言由引擎自动处理
（引擎侧无 language 选项，传入的语言提示忽略）。

权重为 audio.cpp 专用单文件 GGUF 包（audio-cpp/audio.cpp-gguf，
Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf，含 config/tokenizer/
codec 全套资源）。模型文件只在 HF 默认缓存（~/.cache/huggingface/hub）：
缺失时自动经 HF 下载并在缓存内生成引擎可用的 .gguf 别名（见
hf._ensure_gguf_file）；项目 models/ 仅支持手工放置。

该族支持 seed（GEN_SEED；-1 = 随机）。多参考条件（multi_reference_cond
扬声器标签）不在本项目范围（只做单参考语音克隆）。
"""

from __future__ import annotations

import logging
import os
import tempfile

from .audiocpp import (
    AudioResult,
    _backend_flag,
    _chunk_flags,
    _ensure_binary,
    ensure_tmp_dir,
    run_cli,
)
from .config import (
    Config,
    FISH_AUDIO_MAX_NEW_TOKENS,
    FISH_AUDIO_TEMPERATURE,
    FISH_AUDIO_TOP_K,
    FISH_AUDIO_TOP_P,
    GEN_SEED,
    MODELS_DIR,
    TMP_DIR,
)
from .hf import _ensure_gguf_file

# 模型族名（audiocpp --family 取值）
FAMILY = "fish_audio"

# 权重：项目 models/ 仅支持手工放置；默认放 HF 缓存（引擎须真实 .gguf 路径）
GGUF_LOCAL = os.path.join(MODELS_DIR, "Fish-Audio-S2-Pro-GGUF",
                          "fish-audio-s2-pro-q8_0.gguf")
GGUF_HF_REPO = "audio-cpp/audio.cpp-gguf"
GGUF_HF_FILE = ("Fish-Audio-S2-Pro-GGUF/"
                "fish-audio-s2-pro-q8_0.gguf")

# 生成参数 → audiocpp CLI 参数映射（config 顶部常量；空/0 = 不传 = 引擎默认）
# Fish Audio 官方基准（同引擎默认）：temperature 0.8 / top-p 0.8 / top-k 30 /
# max_new_tokens 1024（单 chunk 语义 token 上限）。
_OPT_MAP = {
    "temperature": ("--temperature", FISH_AUDIO_TEMPERATURE),      # float
    "top_p": ("--top-p", FISH_AUDIO_TOP_P),                        # float
    "top_k": ("--top-k", FISH_AUDIO_TOP_K),                        # int
    "max_new_tokens": ("--max-new-tokens", FISH_AUDIO_MAX_NEW_TOKENS),  # int
}


def _opt_value(k: str, kwargs: dict) -> str | None:
    """取某生成参数值：调用方 kwargs 优先；无则 config 顶部常量；空/0 = 不传。"""
    flag, default = _OPT_MAP[k]
    if k in kwargs and kwargs[k] is not None:
        v = kwargs[k]
    else:
        v = default
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if k in ("top_k", "max_new_tokens") and int(v) <= 0:
            return None            # 0 = 引擎默认
        return str(v)
    return str(v)


def _ensure_model(logger: logging.Logger) -> str:
    """定位 Fish Audio S2-Pro GGUF：手工放置的本地文件优先，否则经 HF 下载。

    audio.cpp 按真实文件扩展名识别 GGUF，HF 缓存 blob/软链路径不能直接用，
    _ensure_gguf_file 会在 HF 默认缓存仓库目录内生成带 .gguf 的硬链接别名并
    返回。模型不落工程目录（GGUF_LOCAL 仅支持用户手工放置）。
    """
    if os.path.isfile(GGUF_LOCAL):
        return GGUF_LOCAL
    return _ensure_gguf_file(GGUF_HF_REPO, GGUF_HF_FILE, logger)


def generate(cfg: Config, logger: logging.Logger, **kwargs) -> AudioResult:
    """Fish Audio S2-Pro 零样本语音克隆：ref_audio（+ref_text）→ 音频。

    kwargs 支持：text / language（引擎无语言选项，忽略）/ ref_audio /
    ref_text（可选）/ 生成参数（temperature / top_p / top_k /
    max_new_tokens，kwargs 优先，缺省用 config.py 顶部常量；空值 = 引擎
    默认）。克隆任务用 --task tts。输出采样率以产出 wav 实际为准。
    """
    text = kwargs.pop("text", "")
    language = kwargs.pop("language", None)
    ref_audio = kwargs.pop("ref_audio", None)
    ref_text = kwargs.pop("ref_text", None)
    if not ref_audio:
        raise ValueError("Fish Audio 语音克隆需要 ref_audio（参考音频）")
    ref_audio = os.path.abspath(ref_audio)
    if language:
        logger.info("Fish Audio 语言由引擎自动处理，忽略提示 %s", language)

    binary = _ensure_binary(logger)
    model = _ensure_model(logger)
    ensure_tmp_dir()

    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="fish-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "tts", "--family", FAMILY,
               "--model", model, "--backend", _backend_flag(cfg.device),
               "--text", text, "--voice-ref", ref_audio, "--out", out_wav]
        # 长文本分块（config.TEXT_CHUNK_SIZE/MODE，自动选 endline/default）
        cmd += _chunk_flags(text)
        if ref_text:
            cmd += ["--reference-text", ref_text]
        # 生成参数：kwargs 优先 → config 顶部常量兜底；空值不发 flag
        for k in _OPT_MAP:
            v = _opt_value(k, kwargs)
            if v is not None:
                cmd += [_OPT_MAP[k][0], v]
        # 种子：GEN_SEED >= 0 时显式传（同值可复现）；-1 = 随机不传
        if GEN_SEED >= 0:
            cmd += ["--seed", str(GEN_SEED)]
        ignored = set(kwargs) - set(_OPT_MAP) - {"text", "language",
                                                 "ref_audio", "ref_text"}
        if ignored:
            logger.info("Fish Audio 不支持以下生成参数，已忽略: %s",
                        ", ".join(sorted(str(x) for x in ignored)))
        run_cli(cmd, cfg.device, logger)
        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError("Fish Audio 未产出 WAV 文件")
        import soundfile as sf
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        return AudioResult(audio=data, sampling_rate=sr)
    finally:
        if os.path.exists(out_wav):
            try:
                os.remove(out_wav)
            except OSError:
                pass
