#!/usr/bin/env python3
"""
audio-tools Web Demo — Gradio 双页（生成 / 配置）语音克隆

引擎与模型按需加载：启动只启动 Web 界面，不做任何预热；点击"生成"时
synthesize 内部才定位/自动构建引擎、定位/下载模型 GGUF（日志可见），点击
结束立即释放引擎进程内状态（src/pipeline.release()），长时间运行不留存
任何引擎/模型。

页面：
- 生成页：左栏（参考音频 → ASR 自动转写文本 → txt 文本文件 → 待合成文本）
  + 右栏（状态 + 按抽卡次数展示生成音频）。
- 配置页：模型选择（omnivoice / indextts2 / fireredtts3）、设备、语言、
  抽卡次数、当前模型的生成参数。配置为进程内运行期设置，仅对当前进程生效；
  持久化修改仍以 src/config.py 顶部变量（或同名环境变量）为准。

用法:
    uv run python web.py

设置（监听地址/端口等）统一在 src/config.py 顶部变量，无命令行参数。
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
    _transcribe_ref,
)
from src.config import (
    FIREREDTTS3_GUIDANCE_SCALE,
    FIREREDTTS3_INFERENCE_STEPS,
    FIREREDTTS3_STOP_THRESHOLD,
    INDEXTTS_TEMPERATURE,
    INDEXTTS_TOP_K,
    INDEXTTS_TOP_P,
    OMNI_GUIDANCE_SCALE,
    OMNI_INFERENCE_STEPS,
    TMP_DIR,
    TTS_MODEL,
    WEB_AUTO_OPEN_BROWSER,
    WEB_IP,
    WEB_PORT,
)
from src.pipeline import release, synthesize

logger = logging.getLogger("audio-tools-web")

# 克隆页抽卡结果槽位数（也是抽卡次数上限）
_MAX_DRAWS = 8

# 生成临时目录：项目根目录下 .tmp（正常退出时 atexit 清理）
_TMP_DIR = TMP_DIR
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

# 可选项（与 src/config.py 顶部 TTS_MODEL / 设备检测保持一致）
_MODEL_CHOICES = ["omnivoice", "indextts2", "fireredtts3"]
_DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu", "xpu"]   # auto = 引擎自动（cuda>mps>cpu）
_LANG_CHOICES = ["Auto", "zh", "en", "yue", "ja", "ko"]

# 各模型生成参数键（config.py 顶部常量是文件默认；UI 覆盖为空时走常量/引擎默认）
_MODEL_PARAMS: dict[str, list[str]] = {
    "omnivoice": ["num_inference_steps", "guidance_scale"],
    "indextts2": ["top_k", "top_p", "temperature"],
    "fireredtts3": ["num_inference_steps", "guidance_scale", "stop_threshold"],
}


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


def _cfg(model: str, device: str, **kw) -> Config:
    """构造本次请求的 Config：device=auto 时留空由引擎自行选择后端。"""
    return Config(tts_model=model or "omnivoice",
                  device="" if device == "auto" else device, **kw)


def _model_gen_kwargs(model_v, omni_s, omni_c, it2_k, it2_p, it2_t,
                      fr3_s, fr3_c, fr3_st) -> dict | None:
    """按模型把配置页生成参数 UI 值转成 gen_kwargs（空值跳过 → 常量/引擎默认）。"""
    m = (model_v or "omnivoice").strip().lower()
    table = {
        "omnivoice": {"num_inference_steps": omni_s,
                      "guidance_scale": omni_c},
        "indextts2": {"top_k": it2_k, "top_p": it2_p,
                      "temperature": it2_t},
        "fireredtts3": {"num_inference_steps": fr3_s,
                        "guidance_scale": fr3_c,
                        "stop_threshold": fr3_st},
    }
    kw = {}
    for key, v in table.get(m, {}).items():
        if v is not None and v != "":
            try:
                kw[key] = int(v) if key in ("num_inference_steps",
                                            "top_k") else float(v)
            except (TypeError, ValueError):
                continue
    return kw or None


def build_demo() -> gr.Blocks:
    os.makedirs(_TMP_DIR, exist_ok=True)

    with gr.Blocks(title="audio-tools — 语音克隆",
                   analytics_enabled=False) as demo:
        gr.Markdown(
            "# audio-tools 语音克隆\n"
            "参考音频 + 文本 → 语音克隆。模型在下方「配置」页选择，"
            "当前默认: " + (TTS_MODEL or "omnivoice") + "。"
        )

        with gr.Tabs():
            # ── Tab1 生成页 ────────────────────────────────
            with gr.Tab("生成 Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_audio = gr.Audio(
                            label="1. 参考音频 Reference Audio",
                            type="filepath",
                        )
                        asr_text = gr.Textbox(
                            label="2. 参考文本（ASR 自动转写，可修改）",
                            lines=3, interactive=True,
                            placeholder="上传参考音频后自动填入转写文本…",
                        )
                        txt_file = gr.UploadButton(
                            "3. 文本文件（.txt，可选）",
                            file_types=[".txt"], file_count="single",
                        )
                        text = gr.Textbox(
                            label="4. 待合成文本 Text to Synthesize",
                            lines=6,
                            placeholder="输入需要合成的文本（或上传 txt 自动填入）…",
                        )
                        btn = gr.Button("生成 Generate", variant="primary")
                    with gr.Column(scale=1):
                        status = gr.Textbox(
                            label="状态 Status", lines=3, interactive=False)
                        outputs = [
                            gr.Audio(label=f"结果 {i + 1} Result {i + 1}",
                                     type="filepath", visible=False)
                            for i in range(_MAX_DRAWS)
                        ]
            # ── Tab2 配置页 ────────────────────────────────
            with gr.Tab("配置 Settings"):
                model = gr.Radio(
                    label="模型选择 TTS Model",
                    choices=_MODEL_CHOICES,
                    value=(TTS_MODEL or "omnivoice"),
                    info="omnivoice / indextts2 / fireredtts3；首次使用自动下载权重",
                )
                with gr.Group():
                    gr.Markdown("**基本设置（运行期生效，仅当前进程）**")
                    device = gr.Dropdown(
                        label="推理设备 Device",
                        choices=_DEVICE_CHOICES, value="auto",
                        info="auto = 引擎自动选择（cuda > mps > cpu）",
                    )
                    language = gr.Dropdown(
                        label="语言 Language (默认)",
                        choices=_LANG_CHOICES, value="Auto",
                        info="选 Auto 以自动检测语种。",
                    )
                    draw_count = gr.Slider(
                        label="抽卡次数 Draw Count",
                        minimum=1, maximum=_MAX_DRAWS, step=1, value=2,
                        info="一次生成几个结果供挑选。",
                    )

                param_note = gr.Markdown("")
                with gr.Group(visible=(TTS_MODEL or "omnivoice") == "omnivoice") as g_omni:
                    gr.Markdown("**OmniVoice 生成参数**")
                    omni_steps = gr.Number(
                        label="去噪步数 num_inference_steps",
                        precision=0, value=OMNI_INFERENCE_STEPS, minimum=0,
                        info="0 = 引擎默认",
                    )
                    omni_cfg = gr.Number(
                        label="CFG 引导 guidance_scale",
                        value=float(OMNI_GUIDANCE_SCALE or 0), minimum=0, step=0.1,
                        info="0 = 引擎默认",
                    )
                with gr.Group(visible=(TTS_MODEL or "omnivoice").startswith("indextts")) as g_it2:
                    gr.Markdown("**IndexTTS-2.5 生成参数（gpt 层采样）**")
                    it2_topk = gr.Number(
                        label="top-k", precision=0, value=INDEXTTS_TOP_K, minimum=0,
                        info="0 = 引擎默认",
                    )
                    it2_topp = gr.Number(
                        label="top-p", value=float(INDEXTTS_TOP_P or 0),
                        minimum=0, maximum=1, step=0.05, info="0 = 引擎默认",
                    )
                    it2_temp = gr.Number(
                        label="temperature", value=float(INDEXTTS_TEMPERATURE or 0),
                        minimum=0, step=0.05, info="0 = 引擎默认",
                    )
                with gr.Group(visible=(TTS_MODEL or "omnivoice").startswith("firered")) as g_fr3:
                    gr.Markdown("**FireRedTTS-3 生成参数（Base 零样本克隆）**")
                    fr3_steps = gr.Number(
                        label="flow 步数 num_inference_steps",
                        precision=0, value=FIREREDTTS3_INFERENCE_STEPS, minimum=0,
                        info="0 = 引擎默认",
                    )
                    fr3_cfg = gr.Number(
                        label="CFG 引导 guidance_scale",
                        value=float(FIREREDTTS3_GUIDANCE_SCALE or 0),
                        minimum=0, step=0.1, info="0 = 引擎默认",
                    )
                    fr3_stop = gr.Number(
                        label="停止阈值 stop_threshold",
                        value=float(FIREREDTTS3_STOP_THRESHOLD or 0),
                        minimum=0, maximum=1, step=0.05, info="0 = 引擎默认",
                    )
                gr.Markdown(
                    "持久化修改请编辑 **src/config.py** 顶部变量或设置同名环境变量；"
                    "本页设置只在当前进程内生效，重启后回到 config.py 默认值。"
                )

        # ── 事件 ─────────────────────────────────────────

        def _asr_on_upload(audio, device_v):
            """上传参考音频后立即用 SenseVoice 转写，文本回填 ASR 显示框。"""
            if not audio:
                return gr.update(value=""), "已清除参考音频。"
            cfg = _cfg("", device_v, ref_audio=audio)
            try:
                text_out = _transcribe_ref(cfg, logger)
                if not text_out:
                    return gr.update(value=""), "ASR 转写结果为空（参考音频可能为静音）。"
                return gr.update(value=text_out), "ASR 转写完成，可修改后用于生成。"
            except Exception as e:
                logger.exception("参考音频 ASR 失败")
                return gr.update(value=""), f"ASR 失败: {type(e).__name__}: {e}"

        def _load_txt(file_path):
            """读取 txt 文件内容填入待合成文本框（UploadButton 值兼容单路径/列表）。"""
            if isinstance(file_path, (list, tuple)):
                file_path = file_path[0] if file_path else None
            if not file_path:
                return gr.update()
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.exception("读取 txt 失败")
                return gr.update()
            return gr.update(value=content)

        def _model_changed(model_v):
            """模型切换：只显示当前模型的生成参数组。"""
            m = (model_v or "omnivoice").strip().lower()
            return (
                gr.update(visible=m == "omnivoice"),
                gr.update(visible=m.startswith("indextts")),
                gr.update(visible=m.startswith("firered")),
                f"当前模型 **{m}** 的生成参数（留空/0 = config.py 默认或引擎默认）",
            )

        def _clone_fn(text_v, ref_aud, ref_txt, model_v, device_v, lang_v,
                      draw_v, omni_s, omni_c, it2_k, it2_p, it2_t,
                      fr3_s, fr3_c, fr3_st):
            """点击生成：按配置页模型/参数逐次抽卡，输出到右栏音频槽。"""
            draw_v = max(1, min(int(draw_v or 2), _MAX_DRAWS))
            if not text_v or not text_v.strip():
                return (*([gr.update()] * _MAX_DRAWS), "请输入待合成文本。")
            if not ref_aud:
                return (*([gr.update()] * _MAX_DRAWS), "请上传参考音频。")
            m = (model_v or "omnivoice").strip().lower()
            cfg = _cfg(m, device_v)
            gen_kwargs = _model_gen_kwargs(m, omni_s, omni_c, it2_k, it2_p,
                                           it2_t, fr3_s, fr3_c, fr3_st)
            lang = None if (lang_v or "Auto") == "Auto" else lang_v
            ts = int(time.time())
            results: list = []
            try:
                for i in range(draw_v):
                    result = synthesize(
                        cfg, logger,
                        text=text_v,
                        language=lang,
                        ref_audio=ref_aud,
                        ref_text=(ref_txt or None),
                        out_dir=_TMP_DIR,
                        out_name=f"{ts}-{i + 1}",
                        gen_kwargs=gen_kwargs,
                    )
                    results.append(result.out_path)
                slots = [
                    gr.update(visible=True, value=results[i])
                    if i < draw_v else gr.update(visible=False)
                    for i in range(_MAX_DRAWS)
                ]
                return (*slots,
                        f"生成完成 共 {draw_v} 个结果（模型 {m}）")
            except Exception as e:
                logger.exception("生成失败")
                return (*([gr.update()] * _MAX_DRAWS),
                        f"错误: {type(e).__name__}: {e}")
            finally:
                # 生成完（无论成败）立即卸载引擎/模型进程内状态
                release()

        ref_audio.change(
            _asr_on_upload,
            inputs=[ref_audio, device],
            outputs=[asr_text, status],
        )
        txt_file.upload(_load_txt, inputs=[txt_file], outputs=[text])
        model.change(
            _model_changed,
            inputs=[model],
            outputs=[g_omni, g_it2, g_fr3, param_note],
        )
        btn.click(
            _clone_fn,
            inputs=[text, ref_audio, asr_text, model, device, language,
                    draw_count, omni_steps, omni_cfg, it2_topk, it2_topp,
                    it2_temp, fr3_steps, fr3_cfg, fr3_stop],
            outputs=[*outputs, status],
        )
    return demo


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    _quiet_hf_logs()
    _cleanup_leftover_tmp()

    # 启动只启动 Web，引擎/模型不做预热：首次点击"生成"或上传参考音频
    # （ASR）时 synthesize/_transcribe_ref 内部才定位/自动构建引擎、下载
    # 模型 GGUF（见 src/audiocpp.py 与各模型核心的 _ensure_binary/
    # _ensure_model），每次任务结束立即释放（release()）。
    logger.info("启动 Web 界面（引擎/模型按需加载，启动不预热）")

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
