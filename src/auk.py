"""
AuK 模型核心（audiocpp `--family auk`，实验性）

腾讯混元 AuK / AuK-Flash：指令驱动的语音生成与编辑（1.5B rectified-flow
DiT + Qwen2.5-Omni-3B 条件编码器 + 24 kHz VAE）。本文件只用其零样本语音
克隆能力：参考音频（`--voice-ref`）+ 待朗读文本 → 同音色朗读。

AuK 与其它族的两点差别：
1. 权重是多文件目录（`config/auk-base.yaml` + `config/auk-flash.yaml` +
   `tokenizer/` + 生成器 / Qwen 条件编码器 / VAE 三组 GGUF），`--model` 传
   目录，组件经 `--session-option auk.*` 指定；下载后按真实文件名在 HF 默认
   缓存内生成整棵别名目录（hf._ensure_gguf_tree），工程
   models/AuK-Base-and-Flash-GGUF/ 可手工放置整个目录（优先）。
2. AuK 接受自然语言"指令"而非裸文本：不给 `instruct` 时 `--text` 必须自己
   是完整上游指令，故默认套 AUK_ZERO_SHOT_TEMPLATE 的 zero-shot 模板
   （"Say the following with the same voice: ..."，见上游 COOKBOOK）；
   给了 AUK_INSTRUCT 则文本按待朗读内容传入，由引擎包成 instruct TTS 指令。
   输出时长也须显式给（不给就按参考音频长度输出），见 _estimate_duration。

变体由 TTS_MODEL 决定：`auk` = Base（32 步高质量档）、`auk_flash` = Flash
（上游蒸馏的固定 4 步档）。语音编辑（--task gen）不在本工具范围内。

引擎限制：AuK 的 native session 目前硬性要求 CUDA 后端（`--backend cuda`），
在 macOS/Metal 与 CPU 上会直接报错（NVIDIA GPU 环境才可用）。
"""

from __future__ import annotations

import logging
import os
import tempfile

from .audiocpp import (
    AudioResult,
    _backend_flag,
    _ensure_binary,
    ensure_tmp_dir,
    run_cli,
)
from .config import (
    AUK_ATTENTION,
    AUK_CHARS_PER_SECOND,
    AUK_DURATION_SCALE,
    AUK_DURATION_SEC,
    AUK_GUIDANCE_SCALE,
    AUK_GGUF_DTYPE,
    AUK_INFERENCE_STEPS,
    AUK_INSTRUCT,
    AUK_LOCAL_DIR,
    AUK_MAX_DURATION,
    AUK_MEM_SAVER,
    AUK_MIN_DURATION,
    AUK_QWEN_GGUF,
    AUK_REPO,
    AUK_SWAY_SAMPLING_COEF,
    AUK_VAE_GGUF,
    AUK_ZERO_SHOT_TEMPLATE,
    GEN_SEED,
    TMP_DIR,
    TTS_MODEL,
    Config,
)
from .hf import _ensure_gguf_tree

# 模型族名（audiocpp --family 取值）
FAMILY = "auk"
# HF 缓存内别名目录名（与 HF 仓库同名，便于人工核对）
_ALIAS_DIR = "AuK-Base-and-Flash-GGUF"
# 生成器精度档候选（AUK_GGUF_DTYPE 优先，其余按质量降序兜底）
_DTYPES = ("f32", "f16", "q8_0")
# 生成参数 → CLI 参数映射（kwargs 优先，其次 config 顶部常量；未知参数忽略）
_OPT_MAP = {
    "num_inference_steps": ("--request-option", "num_inference_steps",
                            AUK_INFERENCE_STEPS),
    "guidance_scale": ("--request-option", "guidance_scale", AUK_GUIDANCE_SCALE),
    "sway_sampling_coef": ("--request-option", "sway_sampling_coef",
                           AUK_SWAY_SAMPLING_COEF),
    "seed": ("--seed", "", GEN_SEED),
}


def _variant_of(name: str) -> str:
    """TTS_MODEL → AuK 变体：名字含 flash 用蒸馏四步档，其余为 Base。"""
    return "flash" if "flash" in (name or "").strip().lower() else "base"


def _required_files() -> list[str]:
    """引擎读取的组件文件（相对模型目录）。

    引擎加载资产时会同时校验 ``config/auk-base.yaml`` 与 ``config/auk-flash.yaml``
    都存在（audio.cpp `load_auk_assets`），故两个 yaml 都要下载，即使只用一个变体。
    """
    return [
        "config/auk-base.yaml",
        "config/auk-flash.yaml",
        AUK_QWEN_GGUF,
        AUK_VAE_GGUF,
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]


def _pick_generator(model_dir: str, variant: str) -> str | None:
    """选定生成器 GGUF 文件名：AUK_GGUF_DTYPE 优先，缺失按质量降序兜底。"""
    order = [AUK_GGUF_DTYPE] + [d for d in _DTYPES if d != AUK_GGUF_DTYPE]
    for dtype in order:
        name = f"auk-{variant}-{dtype}.gguf"
        if os.path.isfile(os.path.join(model_dir, name)):
            return name
    return None


def _local_dir(variant: str) -> str | None:
    """手工放置的完整模型目录（组件齐全才用，否则回落到 HF 别名目录）。"""
    if not os.path.isdir(AUK_LOCAL_DIR):
        return None
    if not all(os.path.isfile(os.path.join(AUK_LOCAL_DIR, p))
               for p in _required_files()):
        return None
    return AUK_LOCAL_DIR if _pick_generator(AUK_LOCAL_DIR, variant) else None


def _ensure_model(logger: logging.Logger, variant: str | None = None) -> str:
    """定位 AuK 模型目录：手工放置优先，否则经 HF 下载（留在 HF 默认缓存）。

    返回可直接传给 `--model` 的目录路径；模型不落工程目录（AUK_LOCAL_DIR
    仅支持用户手工放置整个组件目录）。
    """
    variant = variant or _variant_of(TTS_MODEL)
    local = _local_dir(variant)
    if local:
        return local
    files = _required_files()
    files.append(f"auk-{variant}-{AUK_GGUF_DTYPE}.gguf")
    return _ensure_gguf_tree(AUK_REPO, files, _ALIAS_DIR, logger)


def _model_spec_override() -> str | None:
    """AuK 的 model spec 路径（`--model-spec-override`），缺失返回 None。

    普通构建（非 AUDIOCPP_DEPLOYMENT_BUILD）不把 model_specs 编译进引擎，也没有
    内嵌 spec 的 GGUF 可读——AuK 是多文件组件包，发布方未在组件 GGUF 里嵌 spec
    （其它族的单文件 GGUF 都带）。引擎因此按"运行期数据"到工作目录/可执行文件
    附近找 `model_specs/<family>.json`；这里直接指定引擎源码里的 spec，避免依赖
    子进程工作目录/安装布局（spec 与引擎版本天然一致）。
    """
    from .audiocpp import _src_dir
    path = os.path.join(_src_dir(), "model_specs", f"{FAMILY}.json")
    return path if os.path.isfile(path) else None


def _reference_seconds(path: str) -> float | None:
    """参考音频时长（秒）；读不出返回 None。"""
    try:
        import soundfile as sf
        info = sf.info(path)
        if not info.samplerate:
            return None
        return info.frames / float(info.samplerate)
    except Exception:
        return None


def _estimate_duration(text: str, ref_audio: str, ref_text: str | None,
                       override: str | None, logger: logging.Logger) -> float:
    """估算输出时长（秒）：AuK 不给时长就按参考音频长度输出，必须显式给。

    优先 AUK_DURATION_SEC / kwargs；否则按参考音频时长 × 目标文本与参考文本
    的 UTF-8 字节比（与上游 get_gen_duration 同款），缺参考文本时按
    AUK_CHARS_PER_SECOND 估算；再乘 AUK_DURATION_SCALE 并夹在
    AUK_MIN_DURATION..AUK_MAX_DURATION 之间。
    """
    explicit = (override if override is not None else AUK_DURATION_SEC).strip()
    if explicit and _as_float(explicit, 0.0) > 0:
        return _as_float(explicit, 0.0)

    scale = _as_float(AUK_DURATION_SCALE, 1.0)
    ref_sec = _reference_seconds(ref_audio)
    if ref_sec and ref_text:
        ratio = len(text.encode("utf-8")) / max(1, len(ref_text.encode("utf-8")))
        seconds = ref_sec * ratio
        logger.info("  AuK 时长估算：参考 %.1fs × 文本字节比 %.2f = %.1fs",
                    ref_sec, ratio, seconds)
    else:
        cps = max(_as_float(AUK_CHARS_PER_SECOND, 4.5), 0.1)
        seconds = len(text) / cps
        logger.info("  AuK 时长估算：按 %.1f 字/秒 × %d 字 = %.1fs",
                    cps, len(text), seconds)
    seconds *= scale
    hi = _as_float(AUK_MAX_DURATION, 0.0)
    if hi > 0 and seconds > hi:
        logger.warning("  AuK 估算时长 %.1fs 超过上限 %.1fs，截到上限"
                       "（文本过长会被截断，建议拆分文本）", seconds, hi)
        seconds = hi
    return max(_as_float(AUK_MIN_DURATION, 1.0), seconds)


def _as_float(value, default: float) -> float:
    """字符串常量 → float；空/非法用默认值。"""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _opt_value(key: str, kwargs: dict) -> str | None:
    """取某生成参数值：kwargs 优先，其次 config 顶部常量；空/0/-1 = 不传。"""
    _, _, default = _OPT_MAP[key]
    v = kwargs.pop(key, None)
    if v is None:
        v = default
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if key == "seed" and int(v) < 0:
            return None                        # -1 = 随机
        if key == "num_inference_steps" and int(v) <= 0:
            return None                        # 0 = 引擎默认
        return str(v)
    s = str(v).strip()
    return s or None


def generate(cfg: Config, logger: logging.Logger, **kwargs) -> AudioResult:
    """AuK 零样本语音克隆：ref_audio（+ 可选 ref_text）→ 音频。

    kwargs 支持：text / ref_audio / ref_text / language（忽略，AuK 自动判语言）
    / num_inference_steps / guidance_scale / sway_sampling_coef / seed /
    duration_sec（生成参数 kwargs 优先，缺省用 config.py 顶部常量）。
    输出 24 kHz（以产出 wav 实际采样率为准）。
    """
    text = (kwargs.pop("text", "") or "").strip()
    language = kwargs.pop("language", None)
    ref_audio = kwargs.pop("ref_audio", None)
    ref_text = kwargs.pop("ref_text", None)
    duration_override = kwargs.pop("duration_sec", None)
    if not ref_audio:
        raise ValueError("AuK 语音克隆需要 ref_audio（参考音频）")
    if not text:
        raise ValueError("AuK 需要待合成文本")
    if language:
        logger.info("  AuK 不支持指定语言（自动判断），已忽略: %s", language)
    ref_audio = os.path.abspath(ref_audio)

    variant = _variant_of(cfg.tts_model or TTS_MODEL)
    backend = _backend_flag(cfg.device)
    if backend not in ("cuda", "best"):
        # 引擎的 AuK session 构造时硬性要求 CUDA 后端（非 CUDA 直接抛错），
        # 提前告警，避免下载完数 GB 权重才发现
        logger.warning("  AuK 引擎目前仅支持 CUDA 后端（NVIDIA GPU），当前后端 %s "
                       "预计会失败（见 audio.cpp community_models/auk）", backend)
    binary = _ensure_binary(logger)
    model = _ensure_model(logger, variant)
    generator = _pick_generator(model, variant)
    if not generator:
        raise RuntimeError(f"AuK 模型目录缺少 {variant} 生成器 GGUF: {model}")
    ensure_tmp_dir()

    duration = _estimate_duration(text, ref_audio, ref_text,
                                 str(duration_override) if duration_override else None,
                                 logger)
    instruction = _build_instruction(text)
    logger.info("  AuK 变体: %s（生成器 %s，参考文本 %s）", variant, generator,
                "已提供" if ref_text else "未提供")

    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="auk-", dir=TMP_DIR)
    os.close(fd)
    try:
        cmd = [binary, "--task", "tts", "--family", FAMILY, "--model", model,
               "--backend", backend,
               "--text", instruction, "--voice-ref", ref_audio, "--out", out_wav,
               "--request-option", "duration_sec=%.2f" % duration,
               "--session-option", f"auk.variant={variant}",
               "--session-option", f"auk.model_gguf={generator}",
               "--session-option", f"auk.qwen_gguf={AUK_QWEN_GGUF}",
               "--session-option", f"auk.vae_gguf={AUK_VAE_GGUF}"]
        spec = _model_spec_override()
        if spec:
            cmd += ["--model-spec-override", spec]
        if AUK_INSTRUCT:
            cmd += ["--request-option", f"instruct={AUK_INSTRUCT}"]
        # Flash 档固定 4 步并关闭 guidance，步数/引导项对它无效，故不发送
        skip = ("num_inference_steps", "guidance_scale", "sway_sampling_coef") \
            if variant == "flash" else ()
        for key in _OPT_MAP:
            if key in skip:
                kwargs.pop(key, None)
                continue
            value = _opt_value(key, kwargs)
            if value is not None:
                mode, name, _ = _OPT_MAP[key]
                cmd += [mode, f"{name}={value}" if name else value]
        if AUK_ATTENTION:
            cmd += ["--session-option", f"auk.attention={AUK_ATTENTION}"]
        if AUK_MEM_SAVER:
            cmd += ["--session-option", "auk.mem_saver=true"]
        if kwargs:
            logger.info("  AuK 不支持以下生成参数，已忽略: %s",
                        ", ".join(sorted(str(x) for x in kwargs)))
        run_cli(cmd, cfg.device, logger)
        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError("AuK 未产出 WAV 文件")
        import soundfile as sf
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        return AudioResult(audio=data, sampling_rate=sr)
    finally:
        if os.path.exists(out_wav):
            try:
                os.remove(out_wav)
            except OSError:
                pass


def _build_instruction(text: str) -> str:
    """待朗读文本 → AuK 指令：有声音描述时用裸文本，否则套 zero-shot 模板。"""
    if AUK_INSTRUCT:
        return text
    template = AUK_ZERO_SHOT_TEMPLATE or ""
    if "{text}" in template:
        return template.replace("{text}", text)
    return template + text
