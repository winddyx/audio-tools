"""
FireRedTTS-3 模型核心（audiocpp `--family fireredtts3`）

bilibili FireRedTTS-3 Base 的多语言零样本 TTS（官方 24 语言标签 + 21 个
中文方言标签），audio.cpp 中以 `fireredtts3` 族实现：Base 包支持零样本语音
克隆（`--task clon` + `--voice-ref` + `--reference-text`）；Instruct 包额外
支持声音设计/编辑（本文件只接 Base 克隆，与本期"只做语音克隆"一致）。

权重为 audio.cpp 专用 GGUF（audio-cpp/audio.cpp-gguf，
FireRedTTS3-Base-GGUF/fireredtts3-base-q8_0.gguf）。模型文件只在 HF 默认
缓存（~/.cache/huggingface/hub）：缺失时自动经 HF 下载并在缓存内生成引擎
可用的 .gguf 别名（见 hf._ensure_gguf_file）；项目 models/ 仅支持手工放置。

注意：语言标签须用 FireRed 官方全称（Chinese/English/Cantonese/... 及
ZH_* 方言），本核心负责把 web/CLI 的 ISO 语言码映射过去。
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
    FIREREDTTS3_GUIDANCE_SCALE,
    FIREREDTTS3_INFERENCE_STEPS,
    FIREREDTTS3_STOP_THRESHOLD,
    GEN_SEED,
    MODELS_DIR,
    TMP_DIR,
)
from .hf import _ensure_gguf_file

# 模型族名（audiocpp --family 取值）
FAMILY = "fireredtts3"

# 权重：项目 models/ 仅支持手工放置；默认放 HF 缓存（引擎须真实 .gguf 路径）
GGUF_LOCAL = os.path.join(MODELS_DIR, "FireRedTTS3-Base-GGUF",
                          "fireredtts3-base-q8_0.gguf")
GGUF_HF_REPO = "audio-cpp/audio.cpp-gguf"
GGUF_HF_FILE = "FireRedTTS3-Base-GGUF/fireredtts3-base-q8_0.gguf"

# 语言映射：web/CLI 的 ISO 码 / 方言码 → FireRed 官方标签（引擎仅认标签，
# 默认 Chinese）。未列出的已知官方标签（如 "Chinese"、"ZH_Sichuan"）原样透传。
_LANG_MAP = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese",
    "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "vi": "Vietnamese",
    "th": "Thai", "tr": "Turkish", "hi": "Hindi", "ar": "Arabic",
    "pl": "Polish", "uk": "Ukrainian", "el": "Greek", "cs": "Czech",
    "fi": "Finnish", "id": "Indonesian", "ro": "Romanian",
}

# 生成参数 → `--request-option key=value`（config 顶部常量；空/0 = 不传 =
# 引擎默认）。FireRedTTS-3 Base 官方推荐：4 步 / CFG 2.0 / 停止阈值 0.5；
# 种子沿用 GEN_SEED（-1 = 随机，不传则引擎固定 1234，结果可复现）。
_OPT_MAP = {
    "num_inference_steps": ("num_inference_steps", FIREREDTTS3_INFERENCE_STEPS),  # int
    "guidance_scale": ("guidance_scale", FIREREDTTS3_GUIDANCE_SCALE),              # float
    "stop_threshold": ("stop_threshold", FIREREDTTS3_STOP_THRESHOLD),              # float
    "seed": ("seed", GEN_SEED),                                                    # int; -1 = 不传
}


def _opt_value(k: str, kwargs: dict) -> str | None:
    """取某生成参数值：调用方 kwargs 优先；无则 config 顶部常量；空/0/-1 = 不传。"""
    key, default = _OPT_MAP[k]
    v = kwargs[k] if (k in kwargs and kwargs[k] is not None) else default
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if k == "seed" and int(v) < 0:
            return None            # -1 = 随机
        if k == "num_inference_steps" and int(v) <= 0:
            return None            # 0 = 引擎默认
    return str(v)


def _language_tag(language) -> str | None:
    """把 ISO 码/方言码映射为 FireRed 官方语言标签；未知/空返回 None（引擎默认 Chinese）。"""
    if not language:
        return None
    tag = str(language).strip()
    if tag in _LANG_MAP:
        return _LANG_MAP[tag]
    # 已是官方标签（如 Chinese / Cantonese / ZH_Sichuan）则原样透传
    return tag if (tag.startswith("ZH_") or tag[:1].isupper()) else None


def _ensure_model(logger: logging.Logger) -> str:
    """定位 FireRedTTS-3 GGUF：手工放置的本地文件优先，否则经 HF 下载。

    audio.cpp 按真实文件扩展名识别 GGUF，HF 缓存 blob/软链路径不能直接用，
    _ensure_gguf_file 会在 HF 默认缓存仓库目录内生成带 .gguf 的硬链接别名并
    返回。模型不落工程目录（GGUF_LOCAL 仅支持用户手工放置）。
    """
    if os.path.isfile(GGUF_LOCAL):
        return GGUF_LOCAL
    return _ensure_gguf_file(GGUF_HF_REPO, GGUF_HF_FILE, logger)


def generate(cfg: Config, logger: logging.Logger, **kwargs) -> AudioResult:
    """FireRedTTS-3 Base 零样本语音克隆：ref_audio（+ref_text）→ 音频。

    kwargs 支持：text / language / ref_audio / ref_text / 生成参数
    （num_inference_steps / guidance_scale / stop_threshold / seed，kwargs
    优先，缺省用 config.py 顶部常量；空值 = 引擎默认）。语言码自动映射为
    FireRed 官方标签。输出采样率以产出 wav 实际为准。
    """
    text = kwargs.pop("text", "")
    language = kwargs.pop("language", None)
    ref_audio = kwargs.pop("ref_audio", None)
    ref_text = kwargs.pop("ref_text", None)
    if not ref_audio:
        raise ValueError("FireRedTTS-3 零样本克隆需要 ref_audio（参考音频）")
    ref_audio = os.path.abspath(ref_audio)

    binary = _ensure_binary(logger)
    model = _ensure_model(logger)
    ensure_tmp_dir()

    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="firered-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "clon", "--family", FAMILY,
               "--model", model, "--backend", _backend_flag(cfg.device),
               "--text", text, "--voice-ref", ref_audio, "--out", out_wav]
        # 长文本分块（config.TEXT_CHUNK_SIZE/MODE，自动选 endline/default）
        cmd += _chunk_flags(text)
        tag = _language_tag(language)
        if tag:
            cmd += ["--language", tag]
        if ref_text:
            cmd += ["--reference-text", ref_text]
        # 生成参数：kwargs 优先 → config 顶部常量兜底；空值不发 option
        for k in _OPT_MAP:
            v = _opt_value(k, kwargs)
            if v is not None:
                cmd += ["--request-option", f"{_OPT_MAP[k][0]}={v}"]
        ignored = set(kwargs) - set(_OPT_MAP) - {"text", "language",
                                                 "ref_audio", "ref_text"}
        if ignored:
            logger.info("FireRedTTS-3 不支持以下生成参数，已忽略: %s",
                        ", ".join(sorted(str(x) for x in ignored)))
        run_cli(cmd, cfg.device, logger)
        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError("FireRedTTS-3 未产出 WAV 文件")
        import soundfile as sf
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        return AudioResult(audio=data, sampling_rate=sr)
    finally:
        if os.path.exists(out_wav):
            try:
                os.remove(out_wav)
            except OSError:
                pass
