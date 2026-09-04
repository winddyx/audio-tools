"""
CosyVoice-3 模型核心（audiocpp `--family cosyvoice3`）

阿里 CosyVoice-3 的零样本语音克隆 TTS（zh/en/ja/ko/yue 等语言），audio.cpp
中以 `cosyvoice3` 族实现：`--task clon` + `--voice-ref`（+可选
`--reference-text`），模板 zero_shot。注意该族目前只在引擎 dev 分支实现
（main 未合并），engine clone/构建分支由 config.AUDIOCPP_REF（默认 dev）
决定。

权重为 audio.cpp 专用 GGUF（audio-cpp/audio.cpp-gguf，
CosyVoice3-GGUF/cosyvoice3-q8_0.gguf）。模型文件只在 HF 默认缓存
（~/.cache/huggingface/hub）：缺失时自动经 HF 下载并在缓存内生成引擎可用
的 .gguf 别名（见 hf._ensure_gguf_file）；项目 models/ 仅支持手工放置。
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
    COSYVOICE3_INFERENCE_STEPS,
    COSYVOICE3_TOP_K,
    GEN_SEED,
    MODELS_DIR,
    TMP_DIR,
)
from .hf import _ensure_gguf_file

# 模型族名（audiocpp --family 取值）
FAMILY = "cosyvoice3"

# 权重：项目 models/ 仅支持手工放置；默认放 HF 缓存（引擎须真实 .gguf 路径）
GGUF_LOCAL = os.path.join(MODELS_DIR, "CosyVoice3-GGUF", "cosyvoice3-q8_0.gguf")
GGUF_HF_REPO = "audio-cpp/audio.cpp-gguf"
GGUF_HF_FILE = "CosyVoice3-GGUF/cosyvoice3-q8_0.gguf"

# CosyVoice-3 支持语言码（信息用；引擎侧无 --language 选项，语言由引擎/
# 模板自行处理，此处不传）
_SUPPORTED_LANGS = {"", "zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru",
                    "yue"}

# 克隆模板：本项目只做零样本语音克隆（zero_shot；instruct/cross_lingual 不在列）
_TEMPLATE = "zero_shot"

# 生成参数 → `--request-option key=value`（config 顶部常量；空/0 = 不传 =
# 引擎默认）。官方基准：AR top-k 25、flow 10 步；种子沿用 GEN_SEED（-1 =
# 随机，不传则引擎固定 1986，可复现）。
_OPT_MAP = {
    "top_k": ("top_k", COSYVOICE3_TOP_K),                    # int
    "num_inference_steps": ("num_inference_steps", COSYVOICE3_INFERENCE_STEPS),  # int
    "seed": ("seed", GEN_SEED),                              # int; -1 = 不传
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
        if int(v) <= 0:
            return None            # 0 = 引擎默认
    return str(v)


def _ensure_model(logger: logging.Logger) -> str:
    """定位 CosyVoice-3 GGUF：手工放置的本地文件优先，否则经 HF 下载。

    audio.cpp 按真实文件扩展名识别 GGUF，HF 缓存 blob/软链路径不能直接用，
    _ensure_gguf_file 会在 HF 默认缓存仓库目录内生成带 .gguf 的硬链接别名并
    返回。模型不落工程目录（GGUF_LOCAL 仅支持用户手工放置）。
    """
    if os.path.isfile(GGUF_LOCAL):
        return GGUF_LOCAL
    return _ensure_gguf_file(GGUF_HF_REPO, GGUF_HF_FILE, logger)


def generate(cfg: Config, logger: logging.Logger, **kwargs) -> AudioResult:
    """CosyVoice-3 零样本语音克隆：ref_audio（+ref_text）→ 音频。

    kwargs 支持：text / language（仅记录，引擎无此选项）/ ref_audio /
    ref_text / 生成参数（top_k / num_inference_steps / seed，kwargs 优先，
    缺省用 config.py 顶部常量；空值 = 引擎默认）。输出采样率以产出 wav
    实际为准。
    """
    text = kwargs.pop("text", "")
    language = kwargs.pop("language", None)
    ref_audio = kwargs.pop("ref_audio", None)
    ref_text = kwargs.pop("ref_text", None)
    if not ref_audio:
        raise ValueError("CosyVoice-3 语音克隆需要 ref_audio（参考音频）")
    ref_audio = os.path.abspath(ref_audio)
    if language and str(language).lower() not in _SUPPORTED_LANGS:
        logger.info("CosyVoice-3 不支持语言 %s，已忽略", language)
    elif language:
        logger.info("CosyVoice-3 由引擎按模板处理语言，忽略 %s", language)

    binary = _ensure_binary(logger)
    model = _ensure_model(logger)
    ensure_tmp_dir()

    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="cosyvoice-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "clon", "--family", FAMILY,
               "--model", model, "--backend", _backend_flag(cfg.device),
               "--text", text, "--voice-ref", ref_audio, "--out", out_wav]
        # 长文本分块（config.TEXT_CHUNK_SIZE/MODE，自动选 endline/default）
        cmd += _chunk_flags(text)
        cmd += ["--request-option", f"template_name={_TEMPLATE}"]
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
            logger.info("CosyVoice-3 不支持以下生成参数，已忽略: %s",
                        ", ".join(sorted(str(x) for x in ignored)))
        run_cli(cmd, cfg.device, logger)
        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError("CosyVoice-3 未产出 WAV 文件")
        import soundfile as sf
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        return AudioResult(audio=data, sampling_rate=sr)
    finally:
        if os.path.exists(out_wav):
            try:
                os.remove(out_wav)
            except OSError:
                pass
