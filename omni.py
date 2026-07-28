"""
OmniVoice 配音项目 — 核心模块

提供模型加载、音频转换、模型下载缓存、配置生成及全局默认配置。
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import soundfile as sf

# ── 全局默认配置 ─────────────────────────────────────────────

# OmniVoice 模型 ID
OMNI_MODEL_ID = "k2-fsa/OmniVoice"

# 本地模型路径覆盖（非空时优先于 OMNI_MODEL_ID）
MODEL_PATH = ""

# 项目根目录（omni.py 所在目录）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 硬件加速（OmniVoice MPS 可能 SIGSEGV，如遇崩溃改为 "cpu"）
DEVICE = "mps"

# 生成参数
NUM_STEP = 64
GUIDANCE_SCALE = 3.0
LANGUAGE = "zh"
INSTRUCT = ""

# 长文本配音（txt 子命令）默认值
DEFAULT_REF_AUDIO = ""
DEFAULT_TEXT_PATH = ""
DEFAULT_DRAW_COUNT = 2
REF_TEXT = ""
OUTPUT_DIR = ""

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


# ── 长文本配音入口 ──────────────────────────────────────────


def main():
    """
    OmniVoice 长文本配音（文本文件 → 语音）

    用法:
      uv run python omni.py <ref_audio> <text_file>
      DRAW_COUNT=3 uv run python omni.py /path/to/ref.wav /path/to/text.txt
    """
    parser = argparse.ArgumentParser(description="OmniVoice 长文本配音")
    parser.add_argument("ref_audio", nargs="?", default=None,
                        help="参考音频文件 (.wav)")
    parser.add_argument("text_file", nargs="?", default=None,
                        help="文本文件路径")
    parser.add_argument("--draw-count", "-n", type=int, default=None,
                        help="生成次数（默认 2，可通过 DRAW_COUNT 环境变量覆盖）")
    args = parser.parse_args()

    logger = logging.getLogger("omni-txt")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ref_audio = args.ref_audio or os.environ.get("REF_AUDIO") or DEFAULT_REF_AUDIO
    text_path = args.text_file or os.environ.get("TEXT_PATH") or DEFAULT_TEXT_PATH
    draw_count = args.draw_count or int(os.environ.get("DRAW_COUNT", DEFAULT_DRAW_COUNT))

    if not text_path or not os.path.isfile(text_path):
        logger.error("❌ 请设置 TEXT_PATH 为有效的文本文件路径")
        sys.exit(1)
    if not ref_audio or not os.path.isfile(ref_audio):
        logger.error("❌ 请设置 REF_AUDIO 为有效的音频路径")
        sys.exit(1)

    out_dir = os.path.abspath(OUTPUT_DIR) if OUTPUT_DIR else os.path.dirname(os.path.abspath(text_path))
    os.makedirs(out_dir, exist_ok=True)
    text_basename = os.path.basename(text_path)
    out_base = os.path.join(out_dir, text_basename)

    with open(text_path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        logger.error("❌ 文本文件为空")
        sys.exit(1)

    _tmp_fd, tmp_ref = tempfile.mkstemp(suffix="_omni_txt_ref.wav")
    os.close(_tmp_fd)
    try:
        convert_audio(ref_audio, tmp_ref)

        ref_text = REF_TEXT or transcribe_audio(tmp_ref)

        resolved = resolve_path(OMNI_MODEL_ID, MODEL_PATH)
        model = load_model(resolved)

        gen_config = make_gen_config(NUM_STEP, GUIDANCE_SCALE)

        for draw in range(1, draw_count + 1):
            out_path = f"{out_base}.{draw:02d}.wav"
            logger.info("  [%d/%d] 生成中 …", draw, draw_count)
            t1 = time.time()
            audio = model.generate(
                text=text, language=LANGUAGE, ref_audio=tmp_ref,
                ref_text=ref_text, instruct=INSTRUCT or None,
                duration=None, generation_config=gen_config,
            )[0]
            sf.write(out_path, audio, model.sampling_rate)
            kb = os.path.getsize(out_path) / 1024
            logger.info("  %s  (%.0fKB, %.1fs)", os.path.basename(out_path), kb, time.time() - t1)
    finally:
        if os.path.exists(tmp_ref):
            os.unlink(tmp_ref)


if __name__ == "__main__":
    main()
