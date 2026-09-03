"""
OmniVoice 配音工具 — CLI 入口（文本文件 → WAV）

用法:
  uv run python cli.py <ref_audio.wav> <text.txt> -l yue        # 语音克隆
  uv run python cli.py --text <text.txt> --instruct "female, low pitch, british accent"  # 声音设计
  uv run python cli.py --text <text.txt>                       # 自动音色
  uv run python cli.py --transcribe <ref_audio.wav>            # ASR 转写（校对/数据集用）

模型逻辑全部在 src/ 包（共享核心 + 推理后端），本文件只做参数解析、转写/生成
流程编排与文件输出；后端默认 GGUF（BF16，设置见 src/config.py），Web 界面见 web.py。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Optional

from src import (
    Config,
    get_best_device,
    _gen_kwargs,
    _quiet_hf_logs,
    _to_bool,
    _transcribe_ref,
)

# 推理后端只保留 GGUF（src/backends/gguf.py：omnivoice.cpp + GGUF 权重）；
# 可调设置统一在 src/config.py（GGUF 权重 / C++ 二进制 / ASR / Web 选项等）。
# 合成流程统一走 src/pipeline.py（ASR → 后端 → 命名 → 写盘），本文件只做
# 参数解析、进度显示与文件输出。
from src.pipeline import synthesize


def _run_transcribe(cfg: Config, logger: logging.Logger) -> None:
    """ASR 子命令：用 SenseVoiceSmall-GGUF（llama.cpp runtime）转写参考音频并打印文本。

    不加载 TTS 模型，独立于主流程运行。
    """
    print(_transcribe_ref(cfg, logger))




# ── CLI 入口 ──────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="OmniVoice 配音：600+ 语言零样本语音克隆 / 声音设计 / 自动音色"
    )
    parser.add_argument(
        "ref_audio", nargs="?", default=None,
        help="参考音频文件（语音克隆模式；省略时用 --instruct 声音设计或自动音色）",
    )
    parser.add_argument(
        "text_file", nargs="?", default=None,
        help="文本文件路径（声音设计/自动音色模式可用 --text 代替）",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="文本文件路径（声音设计/自动音色模式使用，避免与参考音频位置参数歧义）",
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None,
        help="合成语言代码/名称（如 en/zh/English；默认自动判断，可用 LANGUAGE 覆盖）",
    )
    parser.add_argument(
        "--ref-text", type=str, default=None,
        help="参考音频转写文本（省略时默认用 SenseVoiceSmall-GGUF 转写参考音频）",
    )
    parser.add_argument(
        "--instruct", type=str, default=None,
        help="声音设计指令，如 'female, low pitch, british accent'（无需参考音频）",
    )
    parser.add_argument(
        "--draw-count", "-n", type=int, default=None,
        help="生成次数（默认 2，可用 DRAW_COUNT 环境变量覆盖）",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="设备：cuda/xpu/mps/cpu（默认自动检测；xpu 需安装 PyTorch xpu 构建）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（默认文本文件所在目录，可用 OUTPUT_DIR 覆盖）",
    )
    parser.add_argument(
        "--transcribe", action="store_true",
        help="ASR 子命令：用 SenseVoiceSmall-GGUF 转写参考音频并打印文本（不生成 TTS）",
    )
    parser.add_argument(
        "--asr-model", type=str, default=None,
        help="本地 SenseVoice GGUF 文件路径（默认经 HF 下载 sensevoice-small-q8.gguf）",
    )
    parser.add_argument(
        "--lang-sym", type=str, default=None,
        help="ASR 语言代码（如 en/zh/yue/ja/ko）；留空则跟随 --language，再留空自动检测",
    )
    parser.add_argument(
        "--region-sym", type=str, default=None,
        help="（保留字段，SenseVoice 自动检测语言）",
    )
    return parser.parse_args(argv)


def _pick(cli_val, env_name: str, default, cast=None):
    """按 CLI 参数 > 环境变量 > 默认值的优先级取值（空字符串视为未设置）。"""
    v = cli_val if cli_val is not None else os.environ.get(env_name)
    if v is None or v == "":
        return default
    return cast(v) if cast else v


def _resolve_config(args: argparse.Namespace,
                    defaults: Optional[Config] = None) -> Config:
    """合并 CLI 参数 → 环境变量 → 默认值，返回有效的运行配置。"""
    d = defaults or Config()
    cfg = Config(
        ref_audio=_pick(args.ref_audio, "REF_AUDIO", d.ref_audio),
        text_path=_pick(args.text_file or getattr(args, "text", None),
                        "TEXT_PATH", d.text_path),
        language=_pick(args.language, "LANGUAGE", d.language),
        ref_text=_pick(args.ref_text, "REF_TEXT", d.ref_text),
        instruct=_pick(args.instruct, "INSTRUCT", d.instruct),
        draw_count=_pick(args.draw_count, "DRAW_COUNT", d.draw_count, int),
        output_dir=_pick(args.output_dir, "OUTPUT_DIR", d.output_dir),
        device=_pick(args.device, "DEVICE", d.device),
        dtype=_pick(None, "DTYPE", d.dtype),
        model_path=_pick(None, "MODEL_PATH", d.model_path),
        model_id=_pick(None, "OMNIVOICE_MODEL_ID", d.model_id),
        transcribe=bool(getattr(args, "transcribe", False))
                   or _to_bool(os.environ.get("TRANSCRIBE", "")),
        asr_model=_pick(getattr(args, "asr_model", None), "ASR_MODEL", d.asr_model),
        asr_hub=_pick(None, "ASR_HUB", d.asr_hub),
        asr_vad=_pick(None, "ASR_VAD", d.asr_vad),
        asr_lang_sym=_pick(getattr(args, "lang_sym", None), "ASR_LANG_SYM", d.asr_lang_sym),
        asr_region_sym=_pick(getattr(args, "region_sym", None), "ASR_REGION_SYM", d.asr_region_sym),
    )
    if not cfg.device:
        cfg.device = get_best_device()
    return cfg
def _validate_inputs(cfg: Config, logger: logging.Logger) -> None:
    """验证输入文件是否存在，不通过则退出进程。"""
    if cfg.draw_count < 1:
        logger.error("❌ DRAW_COUNT/--draw-count 必须 >= 1（当前 %d）", cfg.draw_count)
        sys.exit(1)
    if not cfg.text_path or not os.path.isfile(cfg.text_path):
        logger.error("❌ 请设置有效的文本文件路径")
        sys.exit(1)
    if cfg.ref_audio and not os.path.isfile(cfg.ref_audio):
        logger.error("❌ 参考音频不存在: %s", cfg.ref_audio)
        sys.exit(1)
    if not cfg.ref_audio and not cfg.instruct:
        logger.info("ℹ️ 未提供参考音频与指令，使用自动音色模式")


def main(argv: Optional[list[str]] = None) -> None:
    """
    OmniVoice 配音入口（语音克隆 / 声音设计 / 自动音色）。

    用法:
      uv run python cli.py <ref_audio> <text_file>
      uv run python cli.py <ref_audio> <text_file> --language en
      uv run python cli.py --text <text_file> --instruct "female, low pitch, british accent"
      uv run python cli.py --transcribe <ref_audio>                 # ASR（SenseVoiceSmall-GGUF）
      uv run python cli.py --transcribe <ref_audio> --lang-sym en
      DRAW_COUNT=3 LANGUAGE=yue uv run python cli.py /path/to/ref.wav /path/to/text.txt
    """
    logger = logging.getLogger("omni")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # HF 相关第三方库（httpx/huggingface_hub 等）的 INFO 日志压到 WARNING，
    # 保留 WARNING 及以上提示（如缺 HF_TOKEN）与业务日志
    _quiet_hf_logs()

    try:
        args = _parse_args(argv)
        cfg = _resolve_config(args)

        # ── ASR 子命令：独立于 TTS 主流程 ──
        if cfg.transcribe:
            _run_transcribe(cfg, logger)
            return

        mode = ("语音克隆" if cfg.ref_audio
                else "声音设计" if cfg.instruct
                else "自动音色")
        logger.info("🌐 模式: %s  语言: %s  设备: %s  后端: gguf",
                    mode, cfg.language or "自动", cfg.device)

        _validate_inputs(cfg, logger)

        with open(cfg.text_path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            logger.error("❌ 文本文件为空")
            sys.exit(1)

        # 参考音频直接交给模型处理（模型内部自行加载/重采样/转单声道，
        # 无需 ffmpeg 预转换）；ref_text 省略时默认用 SenseVoiceSmall-GGUF 转写参考音频
        # （不依赖 OmniVoice 内部 Whisper ASR）。转写只做一次，多轮抽卡复用
        ref_text = cfg.ref_text
        if cfg.ref_audio and not ref_text:
            logger.info("ℹ️ ref_text 未提供，用 SenseVoiceSmall-GGUF 转写参考音频 …")
            ref_text = _transcribe_ref(cfg, logger)
            logger.info("📝 参考文本: %s", ref_text)
            if not ref_text:
                logger.error("❌ ASR 转写结果为空（参考音频可能为静音）")
                sys.exit(1)

        # ── 多轮生成 ──
        out_dir = (os.path.abspath(cfg.output_dir)
                   if cfg.output_dir
                   else os.path.dirname(os.path.abspath(cfg.text_path)))
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.basename(cfg.text_path)
        gen_kwargs = _gen_kwargs()

        from tqdm import tqdm  # 进度条

        for draw in range(1, cfg.draw_count + 1):
            logger.info("  [%d/%d] 生成中 …", draw, cfg.draw_count)

            # 合成耗时不可知（后端无进度回调）：起一个后台线程实时刷新耗时
            # 进度条，synthesize 返回后关闭（仅显示 elapsed，不伪造步进百分比）
            pbar = tqdm(total=None, bar_format="{desc}", leave=False)
            stop = threading.Event()

            def _tick() -> None:
                t0 = time.time()
                while not stop.is_set():
                    pbar.set_description_str(
                        f"  ⏳ 生成中 [{draw}/{cfg.draw_count}] "
                        f"{time.time() - t0:.1f}s")
                    pbar.refresh()
                    time.sleep(0.1)

            spinner = threading.Thread(target=_tick, daemon=True)
            spinner.start()
            t1 = time.time()
            try:
                result = synthesize(
                    cfg, logger,
                    text=text,
                    language=cfg.language or None,
                    ref_audio=cfg.ref_audio or None,
                    # 克隆模式下 ref_text 已由用户提供或 SenseVoiceSmall-GGUF 转写
                    # （保证非空非 None，模型内部不会走 Whisper 兜底）；
                    # 声音设计/自动音色模式传 None
                    ref_text=ref_text if cfg.ref_audio else None,
                    instruct=cfg.instruct or None,
                    out_dir=out_dir,
                    out_name=out_name,
                    gen_kwargs=gen_kwargs,
                )
            finally:
                stop.set()
                spinner.join(timeout=1.0)
                pbar.close()
            kb = os.path.getsize(result.out_path) / 1024
            logger.info("  %s  (%.0f KB, %.1f s)",
                        os.path.basename(result.out_path), kb,
                        time.time() - t1)
    except Exception:
        # cfg 可能尚未赋值（_parse_args/_resolve_config 阶段抛错）——用局部变量兜底
        cfg = locals().get("cfg")
        logger.exception("❌ %s失败",
                         "转写" if getattr(cfg, "transcribe", False) else "生成")
        sys.exit(1)


if __name__ == "__main__":
    main()
