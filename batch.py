#!/usr/bin/env python3
"""
omni-batch.py — OmniVoice 批量配音生成

用法:
  REF_AUDIO=/path/to/ref.wav TEXT_FILE=/path/to/lines.txt uv run python omni-batch.py
"""

import logging
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime

import soundfile as sf

from omni import (
    OMNI_MODEL_ID, NUM_STEP, GUIDANCE_SCALE, LANGUAGE, INSTRUCT, DEVICE, _PROJECT_ROOT,
    resolve_path, convert_audio, load_model, make_gen_config,
    transcribe_audio,
)

# ===========================
# 配置变量
# ===========================

MODEL_ID = OMNI_MODEL_ID
MODEL_PATH = ""
TEXT_FILE = ""
TEXT = ""
REF_AUDIO = ""
REF_TEXT = ""
OUTPUT_DIR = ""
NUM_GEN = 2
FILENAME_FORMAT = "{line:03d}_{gen:02d}_{timestamp}.wav"

# ===========================
# 主程序
# ===========================


def main():
    logger = logging.getLogger("omni-batch")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    base = _PROJECT_ROOT

    ref_audio = os.path.abspath(REF_AUDIO) if REF_AUDIO else ""
    ref_text = REF_TEXT or ""
    text_file = os.path.abspath(TEXT_FILE) if TEXT_FILE else ""
    output_dir = os.path.abspath(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(base, "output")

    if not ref_audio or not os.path.exists(ref_audio):
        logger.error("❌ 请设置 REF_AUDIO 为有效的音频路径")
        sys.exit(1)

    if not ref_text:
        # 统一转为 24kHz WAV 再转录
        _tmp_fd, tmp_ref = tempfile.mkstemp(suffix="_omni_batch_ref.wav")
        os.close(_tmp_fd)
        convert_audio(ref_audio, tmp_ref)
        ref_text = transcribe_audio(tmp_ref)
    else:
        tmp_ref = ref_audio

    if text_file and os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        raw_lines = [l.strip() for l in (TEXT or "").split("\n") if l.strip()]
        if not raw_lines:
            logger.error("❌ 请设置 TEXT_FILE 或 TEXT")
            sys.exit(1)
        lines = raw_lines

    total = len(lines) * NUM_GEN
    os.makedirs(output_dir, exist_ok=True)
    logger.info("输出: %s  |  文本 %d 行 × %d 次 = %d 个文件", output_dir, len(lines), NUM_GEN, total)

    resolved = resolve_path(OMNI_MODEL_ID, MODEL_PATH)
    model = load_model(resolved)

    prompt = model.create_voice_clone_prompt(
        ref_audio=tmp_ref, ref_text=ref_text,
    )
    gen_config = make_gen_config(NUM_STEP, GUIDANCE_SCALE)

    completed = 0
    t_start = time.time()

    try:
        for line_idx, text in enumerate(lines):
            for gen_num in range(1, NUM_GEN + 1):
                now = datetime.now()
                fname = FILENAME_FORMAT.format(
                    line=line_idx + 1, gen=gen_num,
                    timestamp=f"{now:%y%m%d%H%M%S}",
                )
                out_path = os.path.join(output_dir, fname)

                if os.path.exists(out_path):
                    completed += 1
                    continue

                try:
                    logger.info("  [%3d/%d] 生成中 %s …", completed + 1, total, fname)
                    audio = model.generate(
                        text=text, language=LANGUAGE,
                        voice_clone_prompt=prompt, instruct=INSTRUCT or None,
                        generation_config=gen_config,
                    )[0]
                    sf.write(out_path, audio, model.sampling_rate)
                    completed += 1
                    elapsed = time.time() - t_start
                    logger.info(
                        "[%3d/%d] %s  (%.0fKB, %.1fs/个)",
                        completed, total, fname,
                        os.path.getsize(out_path) / 1024,
                        elapsed / completed,
                    )
                except Exception as e:
                    logger.error("❌ 第%d行 第%d次: %s", line_idx + 1, gen_num, e)
                    traceback.print_exc()
    finally:
        if tmp_ref != ref_audio and os.path.exists(tmp_ref):
            os.unlink(tmp_ref)


if __name__ == "__main__":
    main()
