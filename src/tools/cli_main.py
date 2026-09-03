"""CLI 入口：合成（默认）/ ASR 转写（--asr）。

参数约定（项目规则）：除"引用哪个文件"外不设启动参数；语言、指令、
次数、输出目录、设备等全部在 src/settings.py 顶部变量控制。

用法:
  uv run python cli.py <text.txt> [<ref.wav>]   # 合成（见模式判定）
  uv run python cli.py --asr <ref.wav>          # ASR 转写并打印文本

模式判定（无 ref.wav）：DEFAULT_INSTRUCT 非空=声音设计，否则=自动音色；
克隆模式缺 ref_text 时自动用 SenseVoice 转写（REF_TEXT_FILE 可指文本文件）。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from src import api, settings


def _read_text_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _mode_label(ref_audio: str, instruct: str) -> str:
    if ref_audio:
        return "语音克隆"
    return "声音设计" if instruct else "自动音色"


def _run_asr(audio: str) -> int:
    if not os.path.isfile(audio):
        print(f"错误: 音频文件不存在: {audio}", file=sys.stderr)
        return 1
    text = api.transcribe(audio=audio)
    print(text)
    return 0


def _run_synth(text_file: str, ref_audio: str) -> int:
    logger = api._logger()
    if not os.path.isfile(text_file):
        logger.error("错误: 文本文件不存在: %s", text_file)
        return 1
    if ref_audio and not os.path.isfile(ref_audio):
        logger.error("错误: 参考音频不存在: %s", ref_audio)
        return 1
    text = _read_text_file(text_file)
    if not text:
        logger.error("错误: 文本文件为空: %s", text_file)
        return 1

    instruct = settings.DEFAULT_INSTRUCT
    ref_text = ""
    if ref_audio:
        # 用户可经顶部变量 REF_TEXT_FILE 提供转写文本，否则自动 ASR
        if settings.REF_TEXT_FILE and os.path.isfile(settings.REF_TEXT_FILE):
            ref_text = _read_text_file(settings.REF_TEXT_FILE)
    elif instruct:
        logger.info("模式: 声音设计  指令: %s", instruct)
    else:
        logger.info("模式: 自动音色")

    language = settings.DEFAULT_LANGUAGE
    out_dir = (os.path.abspath(settings.OUTPUT_DIR)
               if settings.OUTPUT_DIR
               else os.path.dirname(os.path.abspath(text_file)))
    out_name = os.path.splitext(os.path.basename(text_file))[0]
    params = api.default_params()
    logger.info("语言: %s  设备: %s  后端: %s", language or "自动",
                settings.DEVICE or "自选", "gguf(native)")

    for i in range(1, settings.DRAW_COUNT + 1):
        logger.info("[%d/%d] 生成中 …", i, settings.DRAW_COUNT)
        t0 = time.time()
        outcome = api.synthesize(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct if not ref_audio else "",
            language=language,
            params=params,
            out_dir=out_dir,
            out_name=out_name,
            logger=logger,
        )
        kb = os.path.getsize(outcome.out_path) / 1024
        logger.info("%s  (%.0f KB, %.1f s, %d 段)",
                    os.path.basename(outcome.out_path), kb,
                    time.time() - t0, len(outcome.segments))
        if outcome.ref_text:
            logger.info("参考文本(ASR): %s", outcome.ref_text)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    # --asr <文件>：只接受一个音频文件
    if args and args[0] == "--asr":
        if len(args) != 2:
            print("用法: python cli.py --asr <ref.wav>", file=sys.stderr)
            return 2
        return _run_asr(args[1])

    if len(args) < 1 or len(args) > 2:
        print("用法: python cli.py <text.txt> [<ref.wav>]",
              file=sys.stderr)
        return 2
    try:
        return _run_synth(args[0], args[1] if len(args) > 1 else "")
    except Exception as e:
        api._logger().exception("生成失败: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
