"""
OmniVoice 配音工具 — 文本转语音

提供模型加载、音频转换、模型下载缓存、配置管理及长文本配音入口。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import soundfile as sf


# ── 设备检测 ──────────────────────────────────────────────


def get_best_device() -> str:
    """自动检测最佳可用设备：CUDA > MPS > CPU。"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── 配置 ──────────────────────────────────────────────────

@dataclass
class Config:
    """全局默认配置。可通过环境变量或 CLI 参数覆盖。"""

    # ── 模型 ──
    model_id: str = "k2-fsa/OmniVoice"
    model_path: str = ""      # 非空时优先于 model_id
    device: str = ""          # 留空则自动检测（CUDA > MPS > CPU）       

    # ── 生成参数 ──
    num_step: int = 64
    guidance_scale: float = 3.0
    language: str = "zh"      # ISO 639-3；None/"none" 为语言无关
    instruct: str = ""        # 附加指令

    # ── 长文本配音 ──
    ref_audio: str = ""
    text_path: str = ""
    draw_count: int = 1       # 抽卡次数
    ref_text: str = ""        # 留空则自动转录
    output_dir: str = ""      # 留空则输出到文本文件所在目录


# LANGUAGE 可选值（ISO 639-3 代码或完整名称，不区分大小写）：
#   代码    名称              代码    名称
#   ────   ───────────       ────   ───────────
#   zh     Chinese            en     English
#   ja     Japanese           ko     Korean
#   yue    Cantonese          fr     French
#   de     German             es     Spanish
#   pt     Portuguese          it     Italian
#   ru     Russian            ar     Arabic
#   hi     Hindi              vi     Vietnamese
#   th     Thai               id     Indonesian
#   nl     Dutch              pl     Polish
#   tr     Turkish            sv     Swedish
#   da     Danish             fi     Finnish
#   cs     Czech              nb     Norwegian
#   hu     Hungarian          ro     Romanian
#   el     Greek              he     Hebrew
#   bn     Bengali            ta     Tamil
#   te     Telugu             ur     Urdu
#   ms     Malay              nan    闽南语
# 完整列表 646 种语言，传递 None 或 "none" 进入语言无关模式。

# ── 工具函数 ──────────────────────────────────────────────


def _check_exe(name: str) -> None:
    """检查可执行文件是否在 PATH 中，否则抛出 RuntimeError。"""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"未找到 {name}，请安装 ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: winget install ffmpeg\n"
            "  Linux: apt install ffmpeg"
        )


def convert_audio(in_path: str, out_path: str) -> None:
    """转为 24 kHz 单声道 WAV。失败时抛出 RuntimeError。"""
    _check_exe("ffmpeg")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "24000", out_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 转换失败 ({r.returncode}): {r.stderr.strip() or '未知错误'}"
        )


# ── 模型缓存 ──────────────────────────────────────────────

_OMNIVOICE_MODEL = None   # 全局 OmniVoice 模型缓存（单例）
_SENSEVOICE_MODEL = None  # 全局 SenseVoiceSmall 模型缓存（单例）


def resolve_path(model_id: str = "", local_path: str = "") -> str:
    """解析模型路径：优先 local_path，否则从 ModelScope 下载。

    snapshot_download 不传 cache_dir，由 ModelScope 自行决定缓存位置
    （默认 ~/.cache/modelscope/hub，或遵循 MODELSCOPE_CACHE 环境变量）。
    """
    if local_path:
        return os.path.abspath(local_path)
    from modelscope.hub.snapshot_download import snapshot_download
    return snapshot_download(model_id or Config.model_id)


def _load_omnivoice(resolved_path: str, device: str,
                    logger: logging.Logger):
    """加载 OmniVoice 模型（带全局缓存，避免重复加载）。"""
    global _OMNIVOICE_MODEL
    if _OMNIVOICE_MODEL is not None:
        return _OMNIVOICE_MODEL

    from omnivoice import OmniVoice
    import torch

    logger.info("⏳ 加载模型 %s ...", resolved_path)
    t0 = time.time()
    _OMNIVOICE_MODEL = OmniVoice.from_pretrained(
        resolved_path,
        device_map=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    logger.info("✓ 模型加载: %.1fs (%s)", time.time() - t0, device)
    return _OMNIVOICE_MODEL


def _load_sensevoice(device: str, logger: logging.Logger):
    """加载 SenseVoiceSmall（带全局缓存）。"""
    global _SENSEVOICE_MODEL
    if _SENSEVOICE_MODEL is not None:
        return _SENSEVOICE_MODEL

    logger.info("⏳ 加载 SenseVoiceSmall …")
    t0 = time.time()
    from funasr import AutoModel
    _SENSEVOICE_MODEL = AutoModel(
        model="iic/SenseVoiceSmall",
        device=device,
        disable_update=True,
    )
    logger.info("✓ SenseVoiceSmall 加载: %.1fs", time.time() - t0)
    return _SENSEVOICE_MODEL


# ── 音频转录 ──────────────────────────────────────────────


def transcribe_audio(audio_path: str, device: str = "",
                     logger: Optional[logging.Logger] = None) -> str:
    """用 SenseVoiceSmall 转录音频，返回文本（含标点）。"""
    logger = logger or logging.getLogger()
    model = _load_sensevoice(device or Config.device, logger)

    result = model.generate(input=audio_path, use_itn=True)
    raw = result[0].get("text", "").strip()
    # 去掉 FunASR 控制 token：<|zh|> <|NEUTRAL|> <|Speech|> <|woitn|> 等
    text = re.sub(r"<\|[^|]+\|>", "", raw).strip()
    if text:
        logger.info("✓ 参考文本: %s", text[:80] + ("..." if len(text) > 80 else ""))
    return text


# ── 生成配置 ──────────────────────────────────────────────


def make_gen_config(num_step: int = 0,
                    guidance_scale: float = 0.0):
    """创建 OmniVoiceGenerationConfig。"""
    from omnivoice import OmniVoiceGenerationConfig
    return OmniVoiceGenerationConfig(num_step=num_step, guidance_scale=guidance_scale)


# ── CLI 入口 ──────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="OmniVoice 长文本配音")
    parser.add_argument(
        "ref_audio", nargs="?", default=None,
        help="参考音频文件 (.wav)",
    )
    parser.add_argument(
        "text_file", nargs="?", default=None,
        help="文本文件路径",
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None,
        help="合成语言代码/名称（默认 zh，可通过 LANGUAGE 环境变量覆盖）",
    )
    parser.add_argument(
        "--draw-count", "-n", type=int, default=None,
        help="生成次数（默认 2，可通过 DRAW_COUNT 环境变量覆盖）",
    )
    return parser.parse_args(argv)


def _resolve_config(args: argparse.Namespace,
                    defaults: Optional[Config] = None) -> Config:
    """合并 CLI 参数 → 环境变量 → 默认值，返回有效的运行配置。"""
    d = defaults or Config()
    cfg = Config(
        ref_audio=args.ref_audio or os.environ.get("REF_AUDIO") or d.ref_audio,
        text_path=args.text_file or os.environ.get("TEXT_PATH") or d.text_path,
        language=args.language or os.environ.get("LANGUAGE") or d.language,
        draw_count=(args.draw_count if args.draw_count is not None
                 else int(os.environ.get("DRAW_COUNT", d.draw_count))),
        ref_text=os.environ.get("REF_TEXT") or d.ref_text,
        output_dir=os.environ.get("OUTPUT_DIR") or d.output_dir,
        instruct=os.environ.get("INSTRUCT") or d.instruct,
        device=os.environ.get("DEVICE") or d.device,
        model_path=os.environ.get("MODEL_PATH") or d.model_path,
        model_id=os.environ.get("OMNI_MODEL_ID") or d.model_id,
        num_step=int(os.environ.get("NUM_STEP", d.num_step)),
        guidance_scale=float(os.environ.get("GUIDANCE_SCALE", d.guidance_scale)),
    )
    if not cfg.device:
        cfg.device = get_best_device()
    return cfg


def _validate_inputs(ref_audio: str, text_path: str,
                     logger: logging.Logger) -> None:
    """验证输入文件是否存在，不通过则退出进程。"""
    if not text_path or not os.path.isfile(text_path):
        logger.error("❌ 请设置有效的文本文件路径")
        sys.exit(1)
    if not ref_audio or not os.path.isfile(ref_audio):
        logger.error("❌ 请设置有效的参考音频路径")
        sys.exit(1)


def main(argv: Optional[list[str]] = None) -> None:
    """
    OmniVoice 长文本配音入口。

    用法:
      uv run python omni.py <ref_audio> <text_file>
      uv run python omni.py <ref_audio> <text_file> --language en
      DRAW_COUNT=3 LANGUAGE=yue uv run python omni.py /path/to/ref.wav /path/to/text.txt
    """
    logger = logging.getLogger("omni-txt")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        args = _parse_args(argv)
        cfg = _resolve_config(args)
        logger.info("🌐 语言: %s  设备: %s", cfg.language, cfg.device)

        _validate_inputs(cfg.ref_audio, cfg.text_path, logger)

        with open(cfg.text_path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            logger.error("❌ 文本文件为空")
            sys.exit(1)

        # ── 转换参考音频 → 临时 WAV ──
        _tmp_fd, tmp_ref = tempfile.mkstemp(suffix="_omni_txt_ref.wav")
        os.close(_tmp_fd)
        try:
            convert_audio(cfg.ref_audio, tmp_ref)

            # 转录参考文本（如果未提供）
            ref_text = cfg.ref_text or transcribe_audio(tmp_ref, cfg.device, logger)

            # 加载生成模型
            resolved = resolve_path(cfg.model_id, cfg.model_path)
            model = _load_omnivoice(resolved, cfg.device, logger)

            gen_config = make_gen_config(cfg.num_step, cfg.guidance_scale)

            # ── 多轮生成 ──
            out_dir = (os.path.abspath(cfg.output_dir)
                       if cfg.output_dir
                       else os.path.dirname(os.path.abspath(cfg.text_path)))
            os.makedirs(out_dir, exist_ok=True)
            out_base = os.path.join(out_dir, os.path.basename(cfg.text_path))

            for draw in range(1, cfg.draw_count + 1):
                ts = int(time.time())
                out_path = f"{out_base}.{ts}.wav"
                logger.info("  [%d/%d] 生成中 …", draw, cfg.draw_count)
                t1 = time.time()
                audio = model.generate(
                    text=text, language=cfg.language, ref_audio=tmp_ref,
                    ref_text=ref_text, instruct=cfg.instruct or None,
                    duration=None, generation_config=gen_config,
                )[0]
                sf.write(out_path, audio, model.sampling_rate)
                kb = os.path.getsize(out_path) / 1024
                logger.info("  %s  (%.0f KB, %.1f s)",
                            os.path.basename(out_path), kb, time.time() - t1)
        finally:
            if os.path.exists(tmp_ref):
                os.unlink(tmp_ref)
    except Exception:
        logger.exception("❌ 生成失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
