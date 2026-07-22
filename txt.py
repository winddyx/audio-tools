#!/usr/bin/env python3
"""
omni-txt.py — OmniVoice 长文本配音（文本文件 → 语音）

用法:
  uv run python txt.py <ref_audio> <text_file>
  DRAW_COUNT=3 uv run python txt.py /path/to/ref.wav /path/to/text.txt
"""

import argparse
import logging
import os
import sys
import tempfile
import time

import soundfile as sf

from omni import (
    OMNI_MODEL_ID, MODEL_PATH, MODELS_DIR,
    NUM_STEP, GUIDANCE_SCALE, LANGUAGE, INSTRUCT,
    setup_cache, resolve_path, convert_audio, load_model, make_gen_config,
    transcribe_audio,
)

# ===========================
# 默认值（可被 CLI 参数或环境变量覆盖）
# ===========================

DEFAULT_REF_AUDIO = ''
DEFAULT_TEXT_PATH = ''
DEFAULT_DRAW_COUNT = 2
REF_TEXT = ""
OUTPUT_DIR = ""

# ===========================
# 主程序
# ===========================


def main():
    parser = argparse.ArgumentParser(description="OmniVoice 长文本配音")
    parser.add_argument("ref_audio", nargs="?", default=None,
                        help="参考音频文件 (.wav)")
    parser.add_argument("text_file", nargs="?", default=None,
                        help="文本文件路径")
    parser.add_argument("--draw-count", "-n", type=int, default=None,
                        help="生成次数（默认 2，可通过 DRAW_COUNT 环境变量覆盖）")
    args = parser.parse_args()

    setup_cache(MODELS_DIR)

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

    text = open(text_path, encoding="utf-8").read().strip()
    if not text:
        logger.error("❌ 文本文件为空")
        sys.exit(1)

    _tmp_fd, tmp_ref = tempfile.mkstemp(suffix="_omni_txt_ref.wav")
    os.close(_tmp_fd)
    try:
        convert_audio(ref_audio, tmp_ref)

        ref_text = REF_TEXT or transcribe_audio(tmp_ref)

        resolved = resolve_path(OMNI_MODEL_ID, MODEL_PATH, MODELS_DIR)
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
