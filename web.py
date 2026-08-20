#!/usr/bin/env python3
"""
OmniVoice Web Demo — Gradio 交互界面（基于官方 gradio 模板）

模型加载 / 路径解析 / ASR 转写 / 生成参数 全部复用 omni.py 的模块
（omni.py 是唯一实现核心，CLI 见 cli.py；本文件只做 UI 封装，不重写模型逻辑）：
- 模型加载: omni._load_model()（内部经 resolve_path 解析路径，本地优先、
  命中缓存则跳过联网，带全局缓存，复用 CLI 同款加载路径）
- 参考文本转写: omni._transcribe_ref()（FunASR/SenseVoiceSmall，懒加载）
- 生成参数: 参数名与 omni._GEN_PARAM_ENVS 完全一致

用法:
    uv run python web.py
    uv run python web.py --port 38001 --share
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional

import gradio as gr
import gradio.processing_utils as _gradio_proc
import numpy as np
import soundfile as sf

from omnivoice.utils.lang_map import LANG_NAME_TO_ID, lang_display_name

# 模型逻辑全部复用 omni.py（唯一实现）
from omni import (
    Config,
    generate,
    get_best_device,
    _load_model,
    _transcribe_ref,
    _GEN_PARAM_ENVS,
)

logger = logging.getLogger("omnivoice-web")

# ── 主题与样式（gradio 6: launch() 时传入）────────────────
_THEME = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
_CSS = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}
"""

# 推理设备（启动时确定，全局共用）
_DEVICE: str = ""

# 克隆页抽卡结果槽位数（也是抽卡次数上限，默认 2）
_MAX_DRAWS = 8

# 生成临时目录：项目根目录下 .tmp（非系统 /tmp），正常退出时 atexit 清理
_TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)


def _cleanup_leftover_tmp() -> None:
    """每次启动清理项目 .tmp 目录（上次异常退出残留的生成文件）。

    程序正常退出由 atexit 清理 .tmp；异常退出（如 SIGKILL）残留的
    生成文件在此处统一清扫。
    """
    if os.path.isdir(_TMP_DIR):
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
        logger.info("已清理项目临时目录: %s", _TMP_DIR)


def _patch_gradio_audio_probe() -> None:
    """兼容 Windows 常见"有 ffmpeg 无 ffprobe"环境。

    gradio 展示音频文件路径时调用 processing_utils.audio_is_playable()
    探测可播放性（内部走 ffprobe），ffprobe 缺失时抛
    FFExecutableNotFoundError 导致生成成功但界面崩溃。该函数语义本就是
    "探测失败视为可播放"（wav 浏览器原生可播），只是漏捕获了 ffprobe
    可执行文件缺失的异常；这里补全为任何探测失败均返回 True。
    """
    _orig = _gradio_proc.audio_is_playable

    def _safe(path: str) -> bool:
        try:
            return _orig(path)
        except Exception:
            return True

    _gradio_proc.audio_is_playable = _safe


_patch_gradio_audio_probe()

# ── 启动行为配置（直接改这里的值，无需任何命令行参数/环境变量）──
AUTO_OPEN_BROWSER = False  # True = 启动后自动用默认浏览器打开界面


# ── 语言列表（显示名 → 代码）──────────────────────────────
# LANG_NAME_TO_ID: 小写语言名 → ISO 639-3 代码（OmniVoice 支持 600+ 语言）
_ALL_LANGUAGES = ["Auto"] + sorted(
    lang_display_name(n) for n in LANG_NAME_TO_ID
)


def _lang_code(display: str) -> str:
    """UI 显示名 → OmniVoice 语言代码（'English' → 'en'）；Auto/空 → ""。"""
    if not display or display == "Auto":
        return ""
    return LANG_NAME_TO_ID.get(display.strip().lower(), "")


# ── 音色设计属性参考（纯 UI 内容）────────────────────────

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


# ── 生成参数（复用 omni.py 的参数名，与 CLI 环境变量完全一致）──
# 从 omni._GEN_PARAM_ENVS 派生：env 名 → (generate 参数名, cast)
_WEB_PARAMS = {
    "num_step": ("推理步数 Inference Steps",
                 dict(minimum=4, maximum=64, step=1, value=32),
                 "越高越慢、质量越好。"),
    "guidance_scale": ("引导强度 Guidance Scale (CFG)",
                       dict(minimum=0.0, maximum=4.0, step=0.1, value=2.0),
                       "默认 2.0。"),
    "t_shift": ("时间偏移 T Shift",
                dict(minimum=0.0, maximum=1.0, step=0.05, value=0.1), ""),
    "denoise": ("降噪 Denoise", None, "启用后对输出进行降噪处理。"),
    "postprocess_output": ("后处理输出 Postprocess Output", None,
                           "去除生成音频中的长静音。"),
    "normalize_text": ("归一化文本 Normalize Text", None,
                       "文本数字/符号转读法（需 num2words）。"),
    "speed": ("语速 Speed",
              dict(minimum=0.5, maximum=1.5, step=0.05, value=1.0),
              "1.0=正常。>1 加快，<1 减慢。设置时长后此项无效。"),
    "duration": ("时长 Duration (秒)", None,
                 "留空使用语速控制，设置固定时长将覆盖语速。"),
}


def _gen_kwargs_from_ui(ui: Dict[str, Any]) -> Dict[str, Any]:
    """把 UI 参数打包成与 omni.py 环境变量同名同值的生成参数。

    只透传用户显式设置/调整过的项，其余交给模型默认值——
    与 omni.py 的 _gen_kwargs() 语义一致（参数名取自 _GEN_PARAM_ENVS）。
    """
    kw: Dict[str, Any] = {}
    for env, (param, cast) in _GEN_PARAM_ENVS.items():
        v = ui.get(param)
        if v is not None:
            kw[param] = cast(v) if isinstance(v, str) else v
    return kw


# ── 构建 Gradio 界面 ─────────────────────────────────────

def build_demo() -> gr.Blocks:
    # 生成临时目录：项目根目录下 .tmp（退出时 atexit 清理，启动时清扫残留）
    os.makedirs(_TMP_DIR, exist_ok=True)

    # ── 共用生成核心（模型调用全部走 omni.py）──────────────

    def _gen_core(
        text: str,
        language: str,
        ref_audio: Optional[str],
        ref_text: Optional[str],
        instruct: Optional[str],
        ui: Dict[str, Any],
        mode: str,  # "clone" | "design"
    ):
        if not text or not text.strip():
            return None, "请输入待合成文本。"

        # 复用 omni.py：Config + 模型加载（全局缓存，本地优先）
        cfg = Config(device=_DEVICE)
        model = _load_model(cfg, logger)

        lang = _lang_code(language) or None
        kw = _gen_kwargs_from_ui(ui)

        try:
            if mode == "clone":
                if not ref_audio:
                    return None, "请上传参考音频。"
                if not ref_text:
                    # 复用 omni.py 的 SenseVoiceSmall 转写（懒加载；语言代码随 cfg 传入，
                    # 由 omni._asr_language 映射为 SenseVoice 语言代码强制转写）
                    asr_cfg = Config(device=_DEVICE, ref_audio=ref_audio,
                                     language=lang or "")
                    ref_text = _transcribe_ref(asr_cfg, logger)
                audios = generate(
                    cfg, logger,
                    text=text.strip(),
                    language=lang,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    instruct=instruct or None,
                    **kw,
                )
            else:
                audios = generate(
                    cfg, logger,
                    text=text.strip(),
                    language=lang,
                    instruct=instruct or None,
                    **kw,
                )
        except Exception as e:
            logger.exception("生成失败")
            return None, f"错误: {type(e).__name__}: {e}"

        # 转 numpy + 裁剪到 [-1, 1]：上游 generate() 可能返回 torch.Tensor
        # （无 .astype），且超界采样点直接 cast 会 int16 回绕产生爆音
        arr = audios[0]
        if hasattr(arr, "cpu"):  # torch.Tensor
            arr = arr.cpu().numpy()
        waveform = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        # 文件名 = 生成完成时的 unix 时间戳（秒）；同秒内冲突则递增秒数
        ts = int(time.time())
        out_path = os.path.join(_TMP_DIR, f"{ts}.wav")
        while os.path.exists(out_path):
            ts += 1
            out_path = os.path.join(_TMP_DIR, f"{ts}.wav")
        sf.write(out_path, waveform, model.sampling_rate)
        # 返回文件路径而非 (sr, data) 元组：路径分支让 gradio 以文件 basename
        # （unix 时间戳）作为下载文件名；gradio 对路径的 ffprobe 可播放性探测
        # 已由 _patch_gradio_audio_probe() 兜底（ffprobe 缺失时视为可播放），
        # Windows 常见"有 ffmpeg 无 ffprobe"不再抛 FFExecutableNotFoundError。
        return out_path, "生成完成 ✓"

    # ── 主题与样式（gradio 6: 传参到 launch()，不传 Blocks 构造器）────

    # 关闭 gradio 分析上报（避免无谓的联网）
    with gr.Blocks(title="OmniVoice Web Demo", analytics_enabled=False) as demo:
        gr.Markdown(
            "# 🎤 OmniVoice Web Demo\n"
            "基于 OmniVoice 的语音合成演示 — 支持语音克隆与音色设计。"
        )

        # ── 可复用的语言下拉框 ────────────────────────────

        def _lang_dropdown(label="语言 (可选)", value="Auto"):
            return gr.Dropdown(
                label=label,
                choices=_ALL_LANGUAGES,
                value=value,
                allow_custom_value=False,
                interactive=True,
                info="选择 Auto 以自动检测语种。",
            )

        # ── 可复用的生成参数折叠面板 ──────────────────────

        def _gen_settings(include_ref_text: bool = False,
                          include_instruct: bool = False):
            with gr.Accordion("生成参数 (可选)", open=False):
                rt = it = None
                if include_ref_text:
                    rt = gr.Textbox(
                        label="参考文本 (可选) Reference Text",
                        lines=2,
                        placeholder="留空则用 SenseVoiceSmall 自动转写参考音频",
                    )
                if include_instruct:
                    it = gr.Textbox(
                        label="附加指令 (可选) Instruct",
                        lines=2,
                        placeholder="例如：female, low pitch（支持中英混合，仅限下表列出的词条）",
                    )
                ns = gr.Slider(
                    label=_WEB_PARAMS["num_step"][0],
                    info=_WEB_PARAMS["num_step"][2],
                    **_WEB_PARAMS["num_step"][1],
                )
                gs = gr.Slider(
                    label=_WEB_PARAMS["guidance_scale"][0],
                    info=_WEB_PARAMS["guidance_scale"][2],
                    **_WEB_PARAMS["guidance_scale"][1],
                )
                ts = gr.Slider(
                    label=_WEB_PARAMS["t_shift"][0],
                    info=_WEB_PARAMS["t_shift"][2],
                    **_WEB_PARAMS["t_shift"][1],
                )
                dn = gr.Checkbox(
                    label=_WEB_PARAMS["denoise"][0],
                    info=_WEB_PARAMS["denoise"][2], value=True,
                )
                po = gr.Checkbox(
                    label=_WEB_PARAMS["postprocess_output"][0],
                    info=_WEB_PARAMS["postprocess_output"][2], value=True,
                )
                nm = gr.Checkbox(
                    label=_WEB_PARAMS["normalize_text"][0],
                    info=_WEB_PARAMS["normalize_text"][2], value=False,
                )
                sp = gr.Slider(
                    label=_WEB_PARAMS["speed"][0],
                    info=_WEB_PARAMS["speed"][2],
                    **_WEB_PARAMS["speed"][1],
                )
                du = gr.Number(
                    label=_WEB_PARAMS["duration"][0],
                    info=_WEB_PARAMS["duration"][2], value=None,
                )
            if include_ref_text and include_instruct:
                return ns, gs, ts, dn, po, nm, sp, du, rt, it
            if include_instruct:
                return ns, gs, ts, dn, po, nm, sp, du, it
            return ns, gs, ts, dn, po, nm, sp, du

        # ═════════════════════════════════════════════════
        # 界面布局
        # ═════════════════════════════════════════════════

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
                        clone_draw_count = gr.Number(
                            label="抽卡次数 Draw Count",
                            value=2, precision=0, minimum=1, maximum=_MAX_DRAWS,
                            info=f"生成几个结果供挑选（默认为 2，最多 {_MAX_DRAWS}）。",
                        )
                        clone_btn = gr.Button("生成 Generate", variant="primary")
                    with gr.Column(scale=1):
                        clone_status = gr.Textbox(label="状态 Status", lines=2)
                        clone_outputs = [
                            gr.Audio(
                                label=f"结果 {i + 1} Result {i + 1}",
                                type="filepath",
                                visible=False,
                            )
                            for i in range(_MAX_DRAWS)
                        ]
                        (
                            clone_ns, clone_gs, clone_ts, clone_dn,
                            clone_po, clone_nm, clone_sp, clone_du,
                            clone_ref_text, clone_instruct,
                        ) = _gen_settings(include_ref_text=True,
                                          include_instruct=True)

                    def _clone_fn(
                        text, lang, ref_aud, ref_txt, instruct, draw_count,
                        ns, gs, ts, dn, po, nm, sp, du,
                    ):
                        draw_count = max(1, min(int(draw_count or 2), _MAX_DRAWS))
                        results: list = []
                        for i in range(draw_count):
                            out, msg = _gen_core(
                                text=text, language=lang,
                                ref_audio=ref_aud, ref_text=ref_txt or None,
                                instruct=instruct,
                                ui=dict(
                                    num_step=ns, guidance_scale=gs, t_shift=ts,
                                    denoise=dn, postprocess_output=po,
                                    normalize_text=nm, speed=sp, duration=du,
                                ),
                                mode="clone",
                            )
                            if out is None:
                                # 出错：不破坏已有结果，只更新状态
                                return (*([gr.update()] * _MAX_DRAWS), msg)
                            results.append(out)
                        # 前 draw_count 个槽位显示本次结果，其余隐藏
                        slots = [
                            gr.update(visible=True, value=results[i])
                            if i < draw_count
                            else gr.update(visible=False)
                            for i in range(_MAX_DRAWS)
                        ]
                        return (*slots, f"生成完成 ✓ 共 {draw_count} 个结果")

                    clone_btn.click(
                        _clone_fn,
                        inputs=[
                            clone_text, clone_lang, clone_ref_audio,
                            clone_ref_text, clone_instruct, clone_draw_count,
                            clone_ns, clone_gs, clone_ts, clone_dn,
                            clone_po, clone_nm, clone_sp, clone_du,
                        ],
                        outputs=[*clone_outputs, clone_status],
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
                            ds_ns, ds_gs, ds_ts, ds_dn,
                            ds_po, ds_nm, ds_sp, ds_du,
                            ds_instruct,
                        ) = _gen_settings(include_instruct=True)

                        def _design_fn(
                            text, lang, instruct,
                            ns, gs, ts, dn, po, nm, sp, du,
                        ):
                            return _gen_core(
                                text=text, language=lang,
                                ref_audio=None, ref_text=None,
                                instruct=instruct,
                                ui=dict(
                                    num_step=ns, guidance_scale=gs, t_shift=ts,
                                    denoise=dn, postprocess_output=po,
                                    normalize_text=nm, speed=sp, duration=du,
                                ),
                                mode="design",
                            )

                        ds_btn.click(
                            _design_fn,
                            inputs=[
                                ds_text, ds_lang, ds_instruct,
                                ds_ns, ds_gs, ds_ts, ds_dn,
                                ds_po, ds_nm, ds_sp, ds_du,
                            ],
                            outputs=[ds_output, ds_status],
                        )

    return demo


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
        help=f"推理设备（默认自动检测: CUDA > XPU > MPS > CPU）",
    )
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("ASR_MODEL", Config.asr_model),
        help=f"SenseVoice 模型 ID/本地目录（默认: FunAudioLLM/SenseVoiceSmall）",
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
    return parser


# ── 主入口 ────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    global _DEVICE

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args = build_parser().parse_args(argv)
    _DEVICE = args.device or get_best_device()

    # 清扫上次异常退出残留的临时生成文件
    _cleanup_leftover_tmp()

    # 模型预热加载（复用 omni.py 的 _load_model：本地优先、带全局缓存；
    # 内部会经 resolve_path 解析并打印模型目录）
    cfg = Config(
        model_id=args.model_id,
        model_path=args.model_path,
        device=_DEVICE,
        asr_model=args.asr_model or "",
    )
    _load_model(cfg, logger)

    demo = build_demo()
    url = f"http://localhost:{args.port}"
    logger.info("启动 Web 界面 → %s", url)
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        theme=_THEME,
        css=_CSS,
        # 是否自动打开浏览器由 web.py 顶部的 AUTO_OPEN_BROWSER 变量控制
        inbrowser=AUTO_OPEN_BROWSER,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
