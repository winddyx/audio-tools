#!/usr/bin/env python3
"""
OmniVoice Web Demo — Gradio 交互界面

提供语音克隆与音色设计两大功能模块。
核心参数引用 omni.py 的 Config，集中设置。

用法:
    uv run python web.py
    uv run python web.py --model-path /path/to/OmniVoice --port 8000 --share
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# 核心配置与工具函数引用 omni.py（唯一数据源）
from omni import Config, resolve_path, convert_audio, generate_with_breaks

# ── 日志 ──────────────────────────────────────────────────

logger = logging.getLogger("omnivoice-web")


# ── 设备检测 ──────────────────────────────────────────────

def get_best_device() -> str:
    """自动检测最佳可用设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── 语言列表 ──────────────────────────────────────────────

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ── 音色设计属性 ──────────────────────────────────────────

_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童", "Teenager / 少年", "Young Adult / 青年",
        "Middle-aged / 中年", "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调", "Low Pitch / 低音调",
        "Moderate Pitch / 中音调", "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音", "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音", "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音", "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音", "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音", "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话", "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话", "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话", "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话", "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话", "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话", "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}


# ── 模型加载（带全局缓存）──────────────────────────────────

_OMNIVOICE_MODEL: Optional[OmniVoice] = None


def load_model(
    resolved_path: str,
    device: str = "",
    load_asr: bool = False,
    asr_model_name: str = "",
) -> OmniVoice:
    """加载 OmniVoice 模型（带全局缓存）。全部使用 float32。"""
    global _OMNIVOICE_MODEL
    if _OMNIVOICE_MODEL is not None:
        return _OMNIVOICE_MODEL

    device = device or get_best_device()
    logger.info("⏳ 加载模型 %s (device=%s, float32) …", resolved_path, device)
    t0 = time.time()
    _OMNIVOICE_MODEL = OmniVoice.from_pretrained(
        resolved_path,
        device_map=device,
        dtype=torch.float32,
        load_asr=load_asr,
        asr_model_name=asr_model_name or None,
        local_files_only=os.path.isdir(resolved_path),
    )
    logger.info("✓ 模型加载: %.1fs", time.time() - t0)
    return _OMNIVOICE_MODEL


# ── 构建参数解析器 ────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-web",
        description="启动 OmniVoice Web 演示（Gradio）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("OMNI_MODEL_ID", Config.model_id),
        help=f"模型 ID（默认: {Config.model_id}）",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("OMNI_MODEL_PATH", Config.model_path),
        help="本地模型路径（非空时优先于 --model-id）",
    )
    parser.add_argument(
        "--device", default=None,
        help=f"推理设备（默认自动检测，Config: {Config.device}）",
    )
    parser.add_argument(
        "--ip", default="0.0.0.0", help="监听地址（默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("OMNI_PORT", "38001")),
        help="监听端口（默认 38001）",
    )
    parser.add_argument(
        "--root-path", default=None, help="反向代理根路径",
    )
    parser.add_argument(
        "--share", action="store_true", default=False,
        help="创建公开链接",
    )
    parser.add_argument(
        "--no-asr", action="store_true", default=False,
        help="跳过加载 Whisper ASR 模型，参考文本自动转录不可用",
    )
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo"),
        help="ASR 模型路径或 HuggingFace repo id",
    )
    return parser


# ── 构建 Gradio 界面 ─────────────────────────────────────

def build_demo(model: OmniVoice) -> gr.Blocks:
    sampling_rate = model.sampling_rate
    # 临时目录，进程退出时自动清理
    _tmp_dir = tempfile.mkdtemp(prefix="omnivoice_")
    atexit.register(shutil.rmtree, _tmp_dir, ignore_errors=True)

    # ── 共用生成核心 ──────────────────────────────────────

    def _gen_core(
        text: str,
        language: str,
        ref_audio: Optional[str],
        instruct: Optional[str],
        num_step: int,
        guidance_scale: float,
        denoise: bool,
        speed: float,
        duration: Optional[float],
        preprocess_prompt: bool,
        postprocess_output: bool,
        mode: str,  # "clone" | "design"
        ref_text: Optional[str] = None,
    ):
        if not text or not text.strip():
            return None, "请输入待合成文本。"

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or Config.num_step),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else Config.guidance_scale,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config,
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "请上传参考音频。"
            # 音频格式转换（convert_audio 引用自 omni.py）
            _tmp_fd, tmp_wav = tempfile.mkstemp(suffix="_omni_ref.wav")
            os.close(_tmp_fd)
            try:
                convert_audio(ref_audio, tmp_wav)
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=tmp_wav,
                    ref_text=ref_text,
                )
            finally:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            # 按句末标点切段逐段生成，段间插入确定停顿（单句时与直接生成等价）
            audio = generate_with_breaks(
                model, kw.pop("text"), logger=logger, **kw
            )
        except Exception as e:
            logger.exception("生成失败")
            return None, f"错误: {type(e).__name__}: {e}"

        waveform = (audio * 32767).astype(np.int16)

        if mode == "clone":
            ref_name = os.path.basename(ref_audio)
            utc_ts = int(time.time())
            out_name = f"{ref_name}.{utc_ts}.wav"
            out_path = os.path.join(_tmp_dir, out_name)
            sf.write(out_path, waveform, sampling_rate)
            return out_path, "生成完成 ✓"

        return (sampling_rate, waveform), "生成完成 ✓"

    # ── 主题与样式 ────────────────────────────────────────

    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # ── 可复用的语言下拉框 ────────────────────────────────

    def _lang_dropdown(label="语言 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="选择 Auto 以自动检测语种。",
        )

    # ── 可复用的生成参数折叠面板 ──────────────────────────

    def _gen_settings():
        with gr.Accordion("生成参数 (可选)", open=False):
            sp = gr.Slider(
                0.5, 1.5, value=1.0, step=0.05,
                label="语速 Speed",
                info="1.0=正常。>1 加快，<1 减慢。设置时长后此项无效。",
            )
            du = gr.Number(
                value=None,
                label="时长 Duration (秒)",
                info="留空使用语速控制，设置固定时长将覆盖语速。",
            )
            ns = gr.Slider(
                4, 64, value=Config.num_step, step=1,
                label="推理步数 Inference Steps",
                info=f"默认 {Config.num_step}。越低越快，越高质量越好。",
            )
            dn = gr.Checkbox(
                label="降噪 Denoise", value=True,
                info="启用后对输出进行降噪处理。",
            )
            gs = gr.Slider(
                0.0, 4.0, value=Config.guidance_scale, step=0.1,
                label="引导强度 Guidance Scale (CFG)",
                info=f"默认 {Config.guidance_scale}。",
            )
            pp = gr.Checkbox(
                label="预处理参考音频 Preprocess Prompt", value=True,
                info="对参考音频做静音去除和裁剪，为参考文本末尾补标点。",
            )
            po = gr.Checkbox(
                label="后处理输出 Postprocess Output", value=True,
                info="去除生成音频中的长静音。",
            )
        return ns, gs, dn, sp, du, pp, po

    # ═════════════════════════════════════════════════════
    # 界面布局
    # ═════════════════════════════════════════════════════

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Web Demo") as demo:
        gr.Markdown(
            "# 🎤 OmniVoice Web Demo\n"
            "基于 OmniVoice 的语音合成演示 — 支持语音克隆与音色设计。"
        )

        with gr.Tabs():
            # ── Tab 1: 语音克隆 ────────────────────────────
            with gr.TabItem("语音克隆 Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        clone_text = gr.Textbox(
                            label="待合成文本 Text to Synthesize",
                            lines=4,
                            placeholder="输入需要合成的文本…",
                        )
                        clone_lang = _lang_dropdown("语言 (可选)", "Auto")
                        clone_ref_audio = gr.Audio(
                            label="参考音频 Reference Audio",
                            type="filepath",
                        )
                        clone_ref_text = gr.Textbox(
                            label="参考文本 (可选) Reference Text",
                            lines=2,
                            placeholder="留空则自动转录（需 ASR 模型）",
                        )
                        clone_instruct = gr.Textbox(
                            label="附加指令 (可选) Instruct",
                            lines=2,
                            placeholder="例如：用温柔的语气朗读…",
                        )
                        clone_btn = gr.Button("生成 Generate", variant="primary")
                    with gr.Column(scale=1):
                        clone_output = gr.Audio(
                            label="合成结果 Output Audio",
                            type="filepath",
                        )
                        clone_status = gr.Textbox(label="状态 Status", lines=2)
                        (
                            clone_ns, clone_gs, clone_dn,
                            clone_sp, clone_du, clone_pp, clone_po,
                        ) = _gen_settings()

                    def _clone_fn(
                        text, lang, ref_aud, ref_txt, instruct,
                        ns, gs, dn, sp, du, pp, po,
                    ):
                        return _gen_core(
                            text=text, language=lang,
                            ref_audio=ref_aud, instruct=instruct,
                            num_step=ns, guidance_scale=gs,
                            denoise=dn, speed=sp, duration=du,
                            preprocess_prompt=pp, postprocess_output=po,
                            mode="clone", ref_text=ref_txt or None,
                        )

                    clone_btn.click(
                        _clone_fn,
                        inputs=[
                            clone_text, clone_lang, clone_ref_audio,
                            clone_ref_text, clone_instruct,
                            clone_ns, clone_gs, clone_dn,
                            clone_sp, clone_du, clone_pp, clone_po,
                        ],
                        outputs=[clone_output, clone_status],
                    )

            # ── Tab 2: 音色设计 ────────────────────────────
            with gr.TabItem("音色设计 Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ds_text = gr.Textbox(
                            label="待合成文本 Text to Synthesize",
                            lines=4,
                            placeholder="输入需要合成的文本…",
                        )
                        ds_lang = _lang_dropdown("语言 (可选)", "Auto")
                        ds_instruct = gr.Textbox(
                            label="附加指令 / 风格描述 (可选)",
                            lines=3,
                            placeholder="例如：用温柔的女性声音朗读…",
                        )
                        ds_btn = gr.Button("生成 Generate", variant="primary")
                        gr.Markdown(
                            "**音色属性参考** — 可在指令中组合使用：\n\n"
                            + "\n".join(
                                f"- **{cat}**: {', '.join(opts)}"
                                + (f"  \n  _{_ATTR_INFO.get(cat, '')}_"
                                   if cat in _ATTR_INFO else "")
                                for cat, opts in _CATEGORIES.items()
                            )
                        )
                    with gr.Column(scale=1):
                        ds_output = gr.Audio(
                            label="合成结果 Output Audio",
                            type="filepath",
                        )
                        ds_status = gr.Textbox(label="状态 Status", lines=2)
                        (
                            ds_ns, ds_gs, ds_dn,
                            ds_sp, ds_du, ds_pp, ds_po,
                        ) = _gen_settings()

                        def _design_fn(
                            text, lang, instruct,
                            ns, gs, dn, sp, du, pp, po,
                        ):
                            return _gen_core(
                                text=text, language=lang,
                                ref_audio=None, instruct=instruct,
                                num_step=ns, guidance_scale=gs,
                                denoise=dn, speed=sp, duration=du,
                                preprocess_prompt=pp, postprocess_output=po,
                                mode="design",
                            )

                        ds_btn.click(
                            _design_fn,
                            inputs=[
                                ds_text, ds_lang, ds_instruct,
                                ds_ns, ds_gs, ds_dn,
                                ds_sp, ds_du, ds_pp, ds_po,
                            ],
                            outputs=[ds_output, ds_status],
                        )

    return demo


# ── 主入口 ────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()
    load_asr = not args.no_asr

    # 模型路径解析（resolve_path 引用自 omni.py）
    resolved_path = resolve_path(
        model_id=args.model_id,
        local_path=args.model_path,
    )
    logger.info("模型路径: %s", resolved_path)

    # 模型加载（全程 float32）
    model = load_model(
        resolved_path=resolved_path,
        device=device,
        load_asr=load_asr,
        asr_model_name=args.asr_model,
    )

    demo = build_demo(model)

    logger.info(
        "启动 Web 界面 → http://%s:%d",
        args.ip, args.port,
    )
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
