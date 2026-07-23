"""
OmniVoice 配音项目 — 核心模块

提供模型加载、音频转换、模型下载缓存、配置生成及全局默认配置。
"""

import logging
import os
import shutil
import subprocess
import time

# ── 全局默认配置 ─────────────────────────────────────────────

# OmniVoice 模型 ID
OMNI_MODEL_ID = "k2-fsa/OmniVoice"

# 本地模型路径覆盖（非空时优先于 OMNI_MODEL_ID）
MODEL_PATH = ""

# 项目根目录（omni.py 所在目录）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 硬件加速（OmniVoice MPS 可能 SIGSEGV，默认 CPU）
DEVICE = "mps"

# 生成参数
NUM_STEP = 64
GUIDANCE_SCALE = 3.0
LANGUAGE = "zh"
INSTRUCT = ""

# ── 工具函数：ffmpeg 依赖检查 ────────────────────────────────


def _check_exe(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(
            f"未找到 {name}，请安装 ffmpeg: "
            "brew install ffmpeg (macOS) / winget install ffmpeg (Windows) / apt install ffmpeg (Linux)"
        )

# ── 模型缓存与下载 ──────────────────────────────────────────


def resolve_path(
    model_id: str = OMNI_MODEL_ID,
    local_path: str = "",
) -> str:
    """通用模型路径解析：优先 local_path，否则从 ModelScope 下载。

    snapshot_download 不传 cache_dir，由 ModelScope 自行决定缓存位置
    （默认 ~/.cache/modelscope/hub，或遵循 MODELSCOPE_CACHE 环境变量）。
    """
    if local_path:
        return os.path.abspath(local_path)
    from modelscope.hub.snapshot_download import snapshot_download
    return snapshot_download(model_id)

# ── 模型加载 ────────────────────────────────────────────────


def load_model(
    resolved_path: str,
    device: str = DEVICE,
    logger: logging.Logger = None,
):
    """加载 OmniVoice 模型 — MPS 触发 SIGSEGV，默认 CPU。"""
    from omnivoice import OmniVoice
    import torch

    logger = logger or logging.getLogger()
    logger.info("⏳ 加载模型 %s ...", resolved_path)
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        resolved_path,
        device_map=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    logger.info("✓ 模型加载: %.1fs (%s)", time.time() - t0, device)
    return model

# ── 音频转换 ────────────────────────────────────────────────


def convert_audio(in_path: str, out_path: str):
    """转为 24kHz 单声道 WAV。失败时抛出 RuntimeError。"""
    _check_exe("ffmpeg")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "24000", out_path],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 转换失败 ({r.returncode}): {r.stderr.strip() or '未知错误'}"
        )


def dur_sec(path: str) -> float:
    """获取音频时长（秒）。失败时返回 0.0。"""
    _check_exe("ffprobe")
    r = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", path,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return 0.0
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0

# ── 语音识别（FunASR SenseVoiceSmall）────────────────────

_SENSEVOICE_MODEL = None


def transcribe_audio(
    audio_path: str,
    device: str = DEVICE,
    logger: logging.Logger = None,
) -> str:
    """用 FunASR SenseVoiceSmall 转录音频，返回文本（含标点）。"""
    global _SENSEVOICE_MODEL
    logger = logger or logging.getLogger()

    if _SENSEVOICE_MODEL is None:
        logger.info("⏳ 加载 SenseVoiceSmall …")
        t0 = time.time()
        from funasr import AutoModel
        _SENSEVOICE_MODEL = AutoModel(
            model="iic/SenseVoiceSmall",
            device=device,
            disable_update=True,
        )
        logger.info("✓ SenseVoiceSmall 加载: %.1fs", time.time() - t0)

    import re

    result = _SENSEVOICE_MODEL.generate(input=audio_path, use_itn=True)
    raw = result[0].get("text", "").strip()
    # 去掉 FunASR 控制 token：<|zh|> <|NEUTRAL|> <|Speech|> <|woitn|> 等
    text = re.sub(r"<\|[^|]+\|>", "", raw).strip()
    if text:
        logger.info("✓ 参考文本: %s", text[:80] + ("..." if len(text) > 80 else ""))
    return text

# ── 生成配置 ────────────────────────────────────────────────


def make_gen_config(num_step: int = NUM_STEP, guidance_scale: float = GUIDANCE_SCALE):
    """创建 OmniVoiceGenerationConfig。"""
    from omnivoice import OmniVoiceGenerationConfig
    return OmniVoiceGenerationConfig(num_step=num_step, guidance_scale=guidance_scale)
