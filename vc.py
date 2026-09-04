"""
audio-tools — CLI 入口（参考音频 + 文本文件 → WAV，语音克隆）

用法:
  uv run python vc.py <ref_audio.wav> <text.txt>            # 语音克隆（自动 ASR 转写参考音频）
  uv run python vc.py --transcribe <ref_audio.wav>          # ASR 转写（校对/数据集用）

所有可调设置（语言 / 抽卡次数 / 输出目录 / TTS 模型 / 设备等）统一在
src/config.py 顶部变量维护（同名环境变量可覆盖），CLI 只接受数据输入参数
（引用哪个文件）。模型逻辑全部在 src/ 包，本文件只做参数解析、分阶段
流程编排与输出汇总。
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
    _quiet_hf_logs,
    _transcribe_ref,
)
from src.config import TTS_MODEL
from src.pipeline import synthesize

# 终端无多余装饰：分阶段 `[i/N]` + 关键信息，其余噪音由引擎层吞掉。


def _run_transcribe(cfg: Config, logger: logging.Logger) -> None:
    """ASR 子命令：用 SenseVoice 转写参考音频并打印文本。

    不加载任何 TTS 模型，独立于主流程运行。
    """
    logger.info("")
    logger.info("[1/1] ASR 转写（SenseVoice-Small，audiocpp sense_asr）")
    print(_transcribe_ref(cfg, logger))


# ── CLI 入口 ──────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数（仅数据输入：引用哪个文件；设置走 config 顶部变量）。"""
    parser = argparse.ArgumentParser(
        description="audio-tools：语音克隆（audiocpp 引擎，支持多 TTS 模型）"
    )
    parser.add_argument(
        "ref_audio", nargs="?", default=None,
        help="参考音频文件（语音克隆模式必填）",
    )
    parser.add_argument(
        "text_file", nargs="?", default=None,
        help="待合成文本文件路径（语音克隆模式必填）",
    )
    parser.add_argument(
        "--transcribe", action="store_true",
        help="ASR 子命令：用 SenseVoice 转写参考音频并打印文本（不生成 TTS）",
    )
    return parser.parse_args(argv)


def _resolve_config(args: argparse.Namespace) -> Config:
    """按 CLI 数据输入 + config 顶部变量（含环境覆盖）构造运行配置。"""
    from src.config import _env, _env_int
    cfg = Config(
        ref_audio=args.ref_audio or _env("REF_AUDIO", ""),
        text_path=args.text_file or _env("TEXT_PATH", ""),
        transcribe=args.transcribe or _env_bool_arg("TRANSCRIBE"),
    )
    # 设置类字段：文件顶部变量 + 同名环境变量覆盖（见 src/config.py Config 注释）
    if not cfg.transcribe:
        _apply_shared_settings(cfg)
    return cfg


def _env_bool_arg(name: str) -> bool:
    from src.config import _to_bool
    return _to_bool(os.environ.get(name, ""))


def _apply_shared_settings(cfg: Config) -> None:
    """把 config 顶部可调设置（语言/抽卡/输出/设备/模型）合入 cfg。

    环境变量可覆盖文件默认值（与 Config 字段同名），vc/web 共用。
    """
    from src.config import _env, _env_int
    cfg.language = _env("LANGUAGE", cfg.language)
    cfg.ref_text = _env("REF_TEXT", cfg.ref_text)
    cfg.draw_count = _env_int("DRAW_COUNT", cfg.draw_count)
    cfg.output_dir = _env("OUTPUT_DIR", cfg.output_dir)
    cfg.device = _env("DEVICE", cfg.device)
    cfg.tts_model = _env("TTS_MODEL", TTS_MODEL)
    cfg.asr_model = _env("ASR_MODEL", cfg.asr_model)


def _validate_inputs(cfg: Config, logger: logging.Logger) -> None:
    """验证输入文件是否存在，不通过则退出进程。"""
    if cfg.draw_count < 1:
        logger.error("DRAW_COUNT 必须 >= 1（当前 %d）", cfg.draw_count)
        sys.exit(1)
    if not cfg.text_path or not os.path.isfile(cfg.text_path):
        logger.error("请设置有效的文本文件路径")
        sys.exit(1)
    if not cfg.ref_audio or not os.path.isfile(cfg.ref_audio):
        logger.error("语音克隆需要有效的参考音频文件")
        sys.exit(1)


def main(argv: Optional[list[str]] = None) -> None:
    """
    audio-tools CLI 入口（语音克隆 / ASR 转写子命令）。

    用法:
      uv run python vc.py <ref_audio.wav> <text.txt>
      uv run python vc.py --transcribe <ref_audio.wav>
    """
    logger = logging.getLogger("audio-tools")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # HF 相关第三方库（httpx/huggingface_hub 等）的 INFO 日志压到 WARNING
    _quiet_hf_logs()

    try:
        args = _parse_args(argv)
        cfg = _resolve_config(args)

        # ── ASR 子命令：独立于 TTS 主流程 ──
        if cfg.transcribe:
            _run_transcribe(cfg, logger)
            return

        if not cfg.device:
            from src import get_best_device
            cfg.device = get_best_device()

        logger.info("语音克隆  TTS模型: %s  语言: %s  设备: %s",
                    cfg.tts_model, cfg.language or "自动", cfg.device)
        _validate_inputs(cfg, logger)

        with open(cfg.text_path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            logger.error("文本文件为空")
            sys.exit(1)

        # ── 六阶段生成（环境/模型/输入/ASR/克隆/输出，vc 与 web 同构）──
        ref_text = cfg.ref_text
        need_asr = bool(cfg.ref_audio and not ref_text)
        total_stages = 6
        stage_idx = 0

        def _section(title: str) -> None:
            nonlocal stage_idx
            stage_idx += 1
            logger.info("")
            logger.info("[%d/%d] %s", stage_idx, total_stages, title)

        # 阶段 1: 环境准备（引擎二进制，一次性，后续轮次走缓存）
        _section("环境准备")
        from src.audiocpp import _ensure_binary
        logger.info("  %s", os.path.basename(_ensure_binary(logger)))

        # 阶段 2: 模型准备（定位/下载 TTS 与 ASR 权重，缓存）
        _section("模型准备")
        _prepare_models(cfg, logger, need_asr=need_asr)

        # 阶段 3: 输入文件检查（文本/参考音频/参数合法性）
        _section("输入文件检查")
        logger.info("  文本: %s", os.path.basename(cfg.text_path))
        logger.info("  参考音频: %s", os.path.basename(cfg.ref_audio))

        # 阶段 4: ASR（SenseVoice 转写参考音频；已提供 ref_text 则跳过）
        _section("ASR 转写")
        if need_asr:
            logger.info("  ref_text 未提供，用 SenseVoice 转写参考音频 …")
            ref_text = _transcribe_ref(cfg, logger)
            if not ref_text:
                logger.error("ASR 转写结果为空（参考音频可能为静音）")
                sys.exit(1)
        else:
            logger.info("  ref_text 已提供，跳过转写")

        # 阶段 5: VOICECLONE（多轮抽卡）
        _section("VOICECLONE 生成")
        out_dir = (os.path.abspath(cfg.output_dir)
                   if cfg.output_dir
                   else os.path.dirname(os.path.abspath(cfg.text_path)))
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.basename(cfg.text_path)
        results: list = []

        for draw in range(1, cfg.draw_count + 1):
            logger.info("  [第 %d/%d 次] 生成中 …", draw, cfg.draw_count)
            # 引擎无进度回调：后台线程每 10s 打一行耗时，避免长合成时终端
            # 长时间无输出
            stop = threading.Event()

            def _heartbeat() -> None:
                t0 = time.time()
                while not stop.wait(10):
                    logger.info("    已用时 %.0f s，继续生成中 …",
                                time.time() - t0)

            hb = threading.Thread(target=_heartbeat, daemon=True)
            hb.start()
            t1 = time.time()
            try:
                result = synthesize(
                    cfg, logger,
                    text=text,
                    language=cfg.language or None,
                    ref_audio=cfg.ref_audio,
                    ref_text=ref_text,
                    out_dir=out_dir,
                    out_name=out_name,
                )
            finally:
                stop.set()
                hb.join(timeout=1.0)
            results.append(result)
            kb = os.path.getsize(result.out_path) / 1024
            logger.info("  生成完成: %.1fs", time.time() - t1)
            logger.info("  输出文件: %s  (%.0f KB, %.1f s)",
                        os.path.basename(result.out_path), kb,
                        result.duration_sec)

        # 阶段 6: 输出文件规范（汇总清单）
        _section("输出")
        for r in results:
            logger.info("  %s", r.out_path)
        logger.info("共 %d 个文件，写入 %s", len(results), out_dir)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("已手动中断，未完成的生成已放弃（已生成的文件已写入）")
        sys.exit(130)
    except Exception:
        cfg = locals().get("cfg")
        logger.exception("%s失败",
                         "转写" if getattr(cfg, "transcribe", False) else "生成")
        sys.exit(1)


def _prepare_models(cfg: Config, logger: logging.Logger, *, need_asr: bool) -> None:
    """模型准备：定位/下载 TTS 模型（与 ASR 模型若需要）GGUF。"""
    tts_name = (cfg.tts_model or "omnivoice").strip().lower()
    if tts_name in ("omnivoice", "omni"):
        from src.omnivoice import _ensure_model as _m
    elif tts_name.startswith("indextts"):
        from src.indextts2 import _ensure_model as _m
    elif tts_name.startswith("firered"):
        from src.fireredtts3 import _ensure_model as _m
    else:
        logger.warning("  未知 TTS_MODEL %s，跳过模型准备", tts_name)
        return
    path = _m(logger)
    logger.info("  TTS 模型(%s): %s", tts_name, os.path.basename(path))
    if need_asr:
        from src.sensevoice import _ensure_model as _a
        apath = _a(logger)
        logger.info("  ASR 模型: %s", os.path.basename(apath))


if __name__ == "__main__":
    main()
