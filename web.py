#!/usr/bin/env python3
"""
audio-tools Web Demo — Gradio 单页语音克隆

模型加载 / 路径解析 / ASR 转写 / 生成全部复用 src/ 包（audiocpp 引擎 +
TTS 模型核心 + SenseVoice ASR；本文件只做 UI 封装）。本期只做语音克隆
（ref_audio + text → 多轮抽卡），无音色设计。

引擎与模型按需加载：启动只启动 Web 界面，不做任何预热；首次点击"生成"
时 synthesize 内部才定位/自动构建引擎、定位/下载模型 GGUF（日志可见），
每次点击生成结束立即释放引擎进程内状态（src/pipeline.release()），
长时间运行不留存任何引擎/模型（模型本就在 audiocpp_cli 子进程内按次
加载，子进程退出即卸载）。

用法:
    uv run python web.py

设置（监听地址/端口/模型/设备/语言等）统一在 src/config.py 顶部变量，
无命令行参数。
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import time

import gradio as gr
import gradio.processing_utils as _gradio_proc

from src import (
    Config,
    _quiet_hf_logs,
)
from src.config import (
    TMP_DIR,
    TTS_MODEL,
    WEB_AUTO_OPEN_BROWSER,
    WEB_IP,
    WEB_PORT,
)
from src.pipeline import release, synthesize

logger = logging.getLogger("audio-tools-web")

# 克隆页抽卡结果槽位数（也是抽卡次数上限，默认 2）
_MAX_DRAWS = 8

# 生成临时目录：项目根目录下 .tmp（正常退出时 atexit 清理）
_TMP_DIR = TMP_DIR
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)


def _cleanup_leftover_tmp() -> None:
    """每次启动清理项目 .tmp 目录（上次异常退出残留的生成文件）。"""
    if os.path.isdir(_TMP_DIR):
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
        logger.info("已清理项目临时目录: %s", _TMP_DIR)


def _patch_gradio_audio_probe() -> None:
    """兼容 Windows 常见"有 ffmpeg 无 ffprobe"环境。"""
    _orig = _gradio_proc.audio_is_playable

    def _safe(path: str) -> bool:
        try:
            return _orig(path)
        except Exception:
            return True

    _gradio_proc.audio_is_playable = _safe


_patch_gradio_audio_probe()


def build_demo() -> gr.Blocks:
    os.makedirs(_TMP_DIR, exist_ok=True)

    # ── 共用生成核心（统一走 src/pipeline.synthesize）────

    def _gen_core(text, language, ref_audio, ref_text, out_name):
        """单次抽卡生成。返回 (输出 wav 路径, 状态消息, ASR 参考文本)。"""
        if not text or not text.strip():
            return None, "请输入待合成文本。", ""
        if language == "Auto":
            language = None
        cfg = Config(tts_model=TTS_MODEL, device=_DEVICE)
        try:
            if not ref_audio:
                return None, "请上传参考音频。", ""
            result = synthesize(
                cfg, logger,
                text=text,
                language=language or None,
                ref_audio=ref_audio,
                ref_text=ref_text or None,
                out_dir=_TMP_DIR,
                out_name=out_name,
                gen_kwargs=None,
            )
        except Exception as e:
            logger.exception("生成失败")
            return None, f"错误: {type(e).__name__}: {e}", ""
        if result.ref_text:
            logger.info("参考文本: %s", result.ref_text)
        return result.out_path, "生成完成", result.ref_text

    # 关闭 gradio 分析上报（避免无谓联网）
    with gr.Blocks(title="audio-tools — 语音克隆", analytics_enabled=False) as demo:
        gr.Markdown(
            "# audio-tools 语音克隆\n"
            "参考音频 + 文本 → 语音克隆。模型由 src/config.py 的 "
            f"TTS_MODEL 决定（当前: {TTS_MODEL}）。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(
                    label="待合成文本 Text to Synthesize",
                    lines=4,
                    placeholder="输入需要合成的文本…",
                )
                language = gr.Dropdown(
                    label="语言 (可选)",
                    choices=["Auto", "zh", "en", "yue", "ja", "ko"],
                    value="Auto",
                    info="选择 Auto 以自动检测语种。",
                )
                ref_audio = gr.Audio(label="参考音频 Reference Audio",
                                     type="filepath")
                ref_text = gr.Textbox(
                    label="参考文本 (可选) Reference Text",
                    lines=2,
                    placeholder="留空则用 SenseVoice 自动转写参考音频",
                )
                draw_count = gr.Number(
                    label="抽卡次数 Draw Count",
                    value=2, precision=0, minimum=1, maximum=_MAX_DRAWS,
                    info=f"生成几个结果供挑选（默认为 2，最多 {_MAX_DRAWS}）。",
                )
                btn = gr.Button("生成 Generate", variant="primary")
            with gr.Column(scale=1):
                asr_text = gr.Textbox(
                    label="ASR 参考文本 Reference Text (ASR)",
                    lines=3,
                    interactive=False,
                    placeholder="未填参考文本时，SenseVoice 转写出的参考音频文本会显示在这里…",
                )
                status = gr.Textbox(label="状态 Status", lines=2)
                outputs = [
                    gr.Audio(label=f"结果 {i + 1} Result {i + 1}",
                             type="filepath", visible=False)
                    for i in range(_MAX_DRAWS)
                ]

                def _clone_fn(text_v, lang_v, ref_aud, ref_txt, draw_count_v):
                    draw_count_v = max(1, min(int(draw_count_v or 2), _MAX_DRAWS))
                    ts = int(time.time())
                    results: list = []
                    asr = ""
                    try:
                        for i in range(draw_count_v):
                            out, msg, asr = _gen_core(
                                text=text_v, language=lang_v, ref_audio=ref_aud,
                                ref_text=ref_txt, out_name=f"{ts}-{i + 1}",
                            )
                            if out is None:
                                return (*([gr.update()] * _MAX_DRAWS),
                                        gr.update(value=asr), msg)
                            results.append(out)
                        slots = [
                            gr.update(visible=True, value=results[i])
                            if i < draw_count_v
                            else gr.update(visible=False)
                            for i in range(_MAX_DRAWS)
                        ]
                        return (*slots, gr.update(value=asr),
                                f"生成完成 共 {draw_count_v} 个结果")
                    finally:
                        # 生成完（无论成败）立即卸载引擎/模型进程内状态：
                        # 模型本就在 audiocpp_cli 子进程内按次加载、退出即
                        # 卸载；此处清掉 Python 侧二进制路径缓存，保证两次
                        # 点击之间进程内不残留任何引擎状态（下次点击重新
                        # 探测），长时间运行不积攒资源。
                        release()

                btn.click(
                    _clone_fn,
                    inputs=[text, language, ref_audio, ref_text, draw_count],
                    outputs=[*outputs, asr_text, status],
                )
    return demo


# 推理设备（启动时确定，全局共用）
_DEVICE: str = ""


def main() -> int:
    global _DEVICE
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    _quiet_hf_logs()
    _cleanup_leftover_tmp()

    # 启动只启动 Web，引擎/模型不做预热：首次点击"生成"时 synthesize 内部
    # 才定位/自动构建引擎、定位/下载模型 GGUF（见 src/audiocpp.py 与各模型
    # 核心的 _ensure_binary/_ensure_model），生成完立即释放（_clone_fn 的
    # finally 调 pipeline.release()）。设备仅做平台探测，不加载任何资源。
    from src import get_best_device
    _DEVICE = os.environ.get("DEVICE", "") or get_best_device()
    logger.info("设备: %s（引擎/模型按需加载，启动不预热）", _DEVICE)

    demo = build_demo()
    url = f"http://localhost:{WEB_PORT}"
    logger.info("启动 Web 界面: %s", url)
    demo.queue().launch(
        server_name=WEB_IP,
        server_port=WEB_PORT,
        share=False,
        inbrowser=WEB_AUTO_OPEN_BROWSER,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
