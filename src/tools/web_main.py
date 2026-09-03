"""Web 界面入口（gradio）：合成 / ASR 两个标签页。

页面只依赖 src.api 与 src.settings；渲染参数控件时按模型注册表的
supported_params 生成（模板化：新模型注册后自动出现可调项）。
运行设置（IP/端口/自动开浏览器）在 src/settings.py 顶部变量。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import gradio as gr

from src import api, settings
from src.model import get_model
from src.types import GenParams

# ── 纯 UI 内容：语言下拉（引擎内置 600+ 语言，这里给常用子集）──
_LANGS: list[tuple[str, str]] = [
    ("自动检测", ""),
    ("English (en)", "en"),
    ("中文 (zh)", "zh"),
    ("粤语 (yue)", "yue"),
    ("日本語 (ja)", "ja"),
    ("한국어 (ko)", "ko"),
    ("Français (fr)", "fr"),
    ("Deutsch (de)", "de"),
    ("Español (es)", "es"),
    ("Русский (ru)", "ru"),
    ("العربية (ar)", "ar"),
]
_LANG_CODE = {label: code for label, code in _LANGS}


def _patch_gradio_audio_probe() -> None:
    """Windows 常见"有 ffmpeg 无 ffprobe"时，gradio 音频可播放性探测
    抛 FFExecutableNotFoundError 导致页面崩溃；探测失败一律视为可播放
    （wav 浏览器原生支持）。"""
    import gradio.processing_utils as proc

    orig = proc.audio_is_playable

    def _safe(path: str) -> bool:
        try:
            return orig(path)
        except Exception:
            return True

    proc.audio_is_playable = _safe


_patch_gradio_audio_probe()


def _out_dir() -> str:
    from src import settings as s

    return s.OUTPUT_DIR or str(s.project_root() / "out")


def _synth(
    mode: str, text: str, lang_label: str, ref_audio: Optional[str],
    ref_text: str, instruct: str, num_step: Optional[int],
    denoise: bool,
):
    """单次合成。返回 (音频路径, 状态, 参考文本, 分段摘要)。"""
    if not text or not text.strip():
        return None, "请输入待合成文本。", "", ""
    if mode == "clone" and not ref_audio:
        return None, "请上传参考音频（语音克隆）。", "", ""
    try:
        outcome = api.synthesize(
            text=text,
            language=_LANG_CODE.get(lang_label or "", "") or "",
            ref_audio=ref_audio or "",
            ref_text=ref_text.strip() or "",
            instruct=instruct.strip() if mode == "design" else "",
            params=GenParams(num_step=num_step, denoise=denoise or None),
            out_dir=_out_dir(),
            out_name=str(int(time.time())),
        )
    except Exception as e:
        return None, f"错误: {type(e).__name__}: {e}", "", ""
    chunks = " | ".join(s.text for s in outcome.segments[:10])
    return (outcome.out_path, "生成完成", outcome.ref_text,
            chunks if chunks else f"单段，共 {len(outcome.segments)} 段")


def _asr(audio: Optional[str]):
    if not audio:
        return "请上传要转写的音频。"
    try:
        return api.transcribe(audio=audio)
    except Exception as e:
        return f"错误: {type(e).__name__}: {e}"


def build_demo() -> gr.Blocks:
    tts = get_model("omnivoice")
    natively_long = "native_longform" in tts.capabilities

    with gr.Blocks(title="OmniVoice Web", analytics_enabled=False) as demo:
        gr.Markdown(
            "# OmniVoice Web\n"
            "GGUF 本地推理：语音克隆 / 声音设计 / 自动音色 / ASR 转写。"
            " 设置见 src/settings.py 顶部变量。"
        )
        with gr.Tabs():
            # ── 合成 ────────────────────────────────────
            with gr.TabItem("合成 Synthesize"):
                mode = gr.Radio(
                    choices=[("自动音色", "auto"),
                             ("声音设计", "design"),
                             ("语音克隆", "clone")],
                    value="auto", label="模式",
                )
                text = gr.Textbox(label="待合成文本", lines=4,
                                  placeholder="输入要合成的文本…")
                with gr.Row():
                    ref_audio = gr.Audio(label="参考音频（语音克隆）",
                                         type="filepath")
                    ref_text = gr.Textbox(
                        label="参考文本（可选；留空则自动 ASR 转写）",
                        lines=2)
                with gr.Row():
                    lang = gr.Dropdown(label="语言", choices=_LANGS,
                                       value="自动检测")
                    instruct = gr.Textbox(
                        label="声音设计指令（如 female, low pitch, british）",
                        lines=2)
                with gr.Accordion("生成参数（可选）", open=False):
                    num_step = gr.Slider(label="steps（步数）", minimum=1,
                                         maximum=100, value=None, step=1)
                    denoise = gr.Checkbox(label="denoise（去噪）", value=True)
                btn = gr.Button("生成", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)
                ref_out = gr.Textbox(label="参考文本（ASR 转写结果）",
                                     interactive=False)
                seg_out = gr.Textbox(label="分段信息", interactive=False)
                audio_out = gr.Audio(label="合成音频", type="filepath")

                btn.click(_synth, [mode, text, lang, ref_audio, ref_text,
                                   instruct, num_step, denoise],
                          [audio_out, status, ref_out, seg_out])
                if natively_long:
                    gr.Markdown(
                        "长文本由引擎原生分块（按句末标点 + 换行硬切），"
                        "分段信息会逐段列出。")

            # ── ASR ─────────────────────────────────────
            with gr.TabItem("ASR 转写"):
                asr_audio = gr.Audio(label="音频", type="filepath")
                asr_btn = gr.Button("转写", variant="primary")
                asr_out = gr.Textbox(label="转写文本", lines=5,
                                     interactive=False)
                asr_btn.click(_asr, [asr_audio], [asr_out])

        gr.Markdown(
            "输出 wav 默认写入项目 out/ 目录（OUTPUT_DIR 顶部变量可改）。"
            " 页面文本均不含 emoji / 特殊符号。")
    return demo


def main(argv: Optional[list[str]] = None) -> int:
    logger = api._logger()
    logger.info("启动 Web: http://%s:%d", settings.WEB_IP, settings.WEB_PORT)
    demo = build_demo()
    demo.launch(
        server_name=settings.WEB_IP,
        server_port=settings.WEB_PORT,
        inbrowser=settings.WEB_AUTO_OPEN_BROWSER,
        show_api=False,
        quiet=True,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
