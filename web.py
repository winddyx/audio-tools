#!/usr/bin/env python3
"""
audio-tools Web Demo — Gradio 三页（语音克隆 / 粤语翻译 / SRT 字幕生成）

引擎与模型按需加载：启动只启动 Web 界面，不做任何预热；点击"生成"时
synthesize 内部才定位/自动构建引擎、定位/下载模型 GGUF（日志可见），点击
结束立即释放引擎进程内状态（src/pipeline.release()），长时间运行不留存
任何引擎/模型。

页面：
- 语音克隆 VoiceClone：左栏（参考音频 → ASR 自动转写文本 → txt 文本文件 →
  待合成文本）+ 右栏（状态 + 按抽卡次数展示生成音频）；页面底部为合并后的
  「模型与运行设置」（模型选择 + 设备 / 语言 / 抽卡次数）。
- 粤语翻译页：左栏（普通话文案输入 + 执行翻译按钮）+ 右栏（可编辑的翻译
  提示词模板 + 粤语译文输出）。推理用 Hy-MT2-1.8B（llama.cpp 的
  llama-completion 子进程，见 src/hymt2.py；LLAMA_CLI 留空时首次使用自动
  clone + 编译 vendor/llama.cpp），模型首次使用自动经 HF 下载到默认缓存。

生成参数（步数 / 采样等）不在界面暴露：统一由 src/config.py 顶部常量
（或同名环境变量）控制，默认即最高质量档。底部「模型与运行设置」只覆盖
模型 / 设备 / 语言 / 抽卡次数，为进程内运行期设置，重启后回到 config.py
默认值。

用法:
    uv run python web.py

设置（监听地址/端口等）统一在 src/config.py 顶部变量，无命令行参数。
"""

from __future__ import annotations

import atexit
import contextvars
import html as _html
import logging
import os
import shutil
import time
from datetime import datetime

import gradio as gr
import gradio.processing_utils as _gradio_proc

from src import (
    Config,
    _quiet_hf_logs,
    _transcribe_ref,
)
from src.config import (
    SRT_ASR,
    SRT_ENUM_COMMA_AS_SPACE,
    SRT_HOTWORDS,
    SRT_HOTWORDS_FILE,
    SRT_ITN,
    SRT_MAX_BLOCK_SECONDS,
    SRT_MAX_GAP_SECONDS,
    SRT_MAX_LINES,
    SRT_MAX_LINE_WIDTH,
    SRT_MIN_BLOCK_SECONDS,
    SRT_MIN_CUE_WIDTH,
    SRT_PUNCTUATION,
    SRT_VAD_MERGE_GAP,
    SRT_VAD_MIN_SPEECH,
    TMP_DIR,
    TTS_MODEL,
    WEB_AUTO_OPEN_BROWSER,
    WEB_IP,
    WEB_PORT,
)
from src.hymt2 import default_prompt as hymt2_default_prompt
from src.hymt2 import translate as hymt2_translate
from src.pipeline import release, synthesize
from src.subtitle import subtitles

# SRT 页 ASR 模型可选值（与 src/config.py 顶部 SRT_ASR 一致）
_SRT_MODEL_CHOICES = ["sensevoice", "qwen3_asr"]

logger = logging.getLogger("audio-tools-web")

# ── 语音克隆页"终端日志"（替代原状态框）────────────────
# 只展示从浏览器打开页面（该 gradio 会话）起产生的日志：事件处理器在进入时
# 把本会话的日志缓冲列表写入 _CTX_BUF，_SessionLogHandler 捕获期间发出的
# logging 记录（引擎/ASR/编排同款控制台信息）；无滚动条、高度固定、最新置底
# （超出部分从顶部裁掉）。日志缓冲按 gradio 会话隔离（gr.State），刷新页面
# 即清空重来。
_TERM_HEIGHT = 100                       # 终端高度（px），固定
_TERM_MAX_LINES = 600                    # 会话缓冲行数上限（防无限增长）
_CTX_BUF: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "web_term_buf", default=None)


class _SessionLogHandler(logging.Handler):
    """把有页面会话上下文期间发出的日志记录追加到该会话的终端缓冲。"""

    def emit(self, record: logging.LogRecord) -> None:
        buf = _CTX_BUF.get()
        if buf is None:
            return
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            buf.append(f"{ts} {record.levelname:<7} {record.getMessage()}")
            if len(buf) > _TERM_MAX_LINES:
                del buf[: len(buf) - _TERM_MAX_LINES]
        except Exception:
            pass


_SESSION_LOG_HANDLER = _SessionLogHandler(level=logging.INFO)
logging.getLogger().addHandler(_SESSION_LOG_HANDLER)


def _term_html(buf: list[str] | None) -> str:
    """按固定高度渲染终端：白底灰字（默认主题观感）、底部对齐、无滚动条。"""
    lines = (buf or [])[-240:]
    body = "<br>".join(_html.escape(x).replace("\n", "<br>") for x in lines)
    if not body:
        body = '<span style="color:#b6bcc4">…（暂无操作日志）</span>'
    return (
        '<div style="position:relative;height:%dpx;overflow:hidden;'
        'background:#ffffff;color:#7a7f87;font:12px/1.55 Menlo,Consolas,monospace;'
        'border:1px solid #e2e4e8;border-radius:6px;box-sizing:border-box;">'
        '<div style="position:absolute;left:0;right:0;bottom:0;padding:5px 8px;'
        'word-break:break-all;">%s</div></div>'
    ) % (_TERM_HEIGHT, body)


def _term_line(level: str, msg: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    return f"{ts} {level:<7} {msg}"

# 克隆页抽卡结果槽位数（也是抽卡次数上限）
_MAX_DRAWS = 8

# 生成临时目录：项目根目录下 .tmp（正常退出时 atexit 清理）
_TMP_DIR = TMP_DIR
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

# 可选项（与 src/config.py 顶部 TTS_MODEL / 设备检测保持一致）
_MODEL_CHOICES = ["omnivoice", "indextts2", "fireredtts3", "cosyvoice3",
                  "moss_tts_local", "qwen3_tts", "fish_audio"]
_DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu", "xpu"]   # auto = 引擎自动（cuda>mps>cpu）
_LANG_CHOICES = ["Auto", "zh", "en", "yue", "ja", "ko"]


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


def _read_hotwords() -> str:
    """SRT 页热词框初始值：根目录 hotword.txt 优先，其次 config.SRT_HOTWORDS。"""
    if SRT_HOTWORDS_FILE:
        try:
            with open(SRT_HOTWORDS_FILE, encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                return text
        except OSError:
            pass
    return SRT_HOTWORDS or ""


def _write_hotwords(text: str) -> None:
    """把本次提交的热词写回根目录 hotword.txt（热记录：每次提交写一次）。"""
    if not SRT_HOTWORDS_FILE:
        return
    try:
        with open(SRT_HOTWORDS_FILE, "w", encoding="utf-8") as f:
            f.write((text or "").strip() + "\n")
    except OSError as e:
        logger.warning("热词记录写入失败（%s）: %s", SRT_HOTWORDS_FILE, e)


def build_demo() -> gr.Blocks:
    os.makedirs(_TMP_DIR, exist_ok=True)

    with gr.Blocks(title="audio-tools",
                   analytics_enabled=False) as demo:
        gr.Markdown(
            "# audio-tools made by David \n"
            "参考音频 + 文本 → 语音克隆；音频 → SRT 字幕。各页设置在其底部"
            "「模型与运行设置」中，当前默认模型: " + (TTS_MODEL or "omnivoice") + "。"
        )

        # ── 终端日志（页面级共享）──────────────────────────────
        # 位于标题下方、Tab 栏上方：三个 Tab 的事件处理器共用同一个会话缓冲，
        # 只记录从打开本页起产生的日志；按 gradio 会话隔离（gr.State），
        # 刷新页面即清空重来。
        gr.Markdown("**终端日志 Terminal（自本页打开起；最新在底部）**")
        log_state = gr.State([])
        terminal = gr.HTML(_term_html([]))

        with gr.Tabs():
            # ── Tab1 语音克隆 ──────────────────────────────
            with gr.Tab("语音克隆 VoiceClone"):
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
                        outputs = [
                            gr.Audio(label=f"结果 {i + 1} Result {i + 1}",
                                     type="filepath", visible=False)
                            for i in range(_MAX_DRAWS)
                        ]
                # ── 模型与运行设置（合并原「配置」页；生成参数不在此暴露，
                #    统一由 src/config.py 顶部常量控制）──────────────
                with gr.Accordion(
                        "模型与运行设置 Model & Runtime Settings",
                        open=False):
                    model = gr.Radio(
                        label="模型选择 TTS Model",
                        choices=_MODEL_CHOICES,
                        value=(TTS_MODEL or "omnivoice"),
                        info="omnivoice / indextts2 / fireredtts3 / cosyvoice3 / "
                             "moss_tts_local / qwen3_tts / fish_audio；首次使用自动下载权重",
                    )
                    with gr.Row():
                        device = gr.Dropdown(
                            label="推理设备 Device",
                            choices=_DEVICE_CHOICES, value="auto", scale=1,
                            info="auto = 引擎自动（cuda > mps > cpu）",
                        )
                        language = gr.Dropdown(
                            label="语言 Language（默认）",
                            choices=_LANG_CHOICES, value="Auto", scale=1,
                            info="Auto = 自动检测语种。",
                        )
                        draw_count = gr.Slider(
                            label="抽卡次数 Draw Count",
                            minimum=1, maximum=_MAX_DRAWS, step=1, value=2,
                            scale=1,
                            info="一次生成几个结果（1-%d）。" % _MAX_DRAWS,
                        )
                    gr.Markdown(
                        "本区设置只在当前进程内生效（重启回到默认）；持久化修改"
                        "请编辑 **src/config.py** 顶部变量或设置同名环境变量——"
                        "生成参数（步数 / 采样等）同样在 config.py 顶部调整。"
                    )

            # ── Tab2 粤语翻译（Hy-MT2-1.8B，llama.cpp 子进程）──────
            with gr.Tab("粤语翻译 Translate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        hy_src = gr.Textbox(
                            label="1. 普通话文案 Source (普通话)",
                            lines=9, interactive=True,
                            placeholder="输入需要翻译成粤语的普通话/中文文案…",
                        )
                        hy_btn = gr.Button(
                            "执行翻译 Translate", variant="primary")
                    with gr.Column(scale=1):
                        hy_prompt = gr.Textbox(
                            label="2. 翻译提示词 Prompt（可编辑；"
                                  "{text} 会被源文案替换）",
                            lines=6, interactive=True, value=hymt2_default_prompt(),
                        )
                        hy_out = gr.Textbox(
                            label="3. 粤语译文 Result (粤语)",
                            lines=9, interactive=False,
                            placeholder="翻译结果将显示在这里…",
                        )

            # ── Tab3 SRT 字幕生成（音频 → 字幕；见 src/subtitle.py）──
            with gr.Tab("SRT 字幕生成 SRT Subtitles"):
                with gr.Row():
                    with gr.Column(scale=1):
                        srt_hotwords = gr.Textbox(
                            label="1. ASR 热词/上下文 Hotwords（仅 Qwen3-ASR）",
                            lines=3, interactive=True,
                            value=_read_hotwords(),
                            placeholder="专有名词/术语，如：赣州、贵阳、腾讯会议…"
                                        "（每次生成写入根目录 hotword.txt）",
                        )
                        srt_audio = gr.Audio(
                            label="2. 音频文件 Audio (wav)",
                            type="filepath",
                        )
                        srt_btn = gr.Button(
                            "生成字幕 Generate SRT", variant="primary")
                    with gr.Column(scale=1):
                        srt_file = gr.File(
                            label="3. SRT 字幕文件（点击下载）",
                            interactive=False,
                        )
                        srt_preview = gr.Textbox(
                            label="4. SRT 预览 Preview",
                            lines=14, interactive=False,
                            placeholder="生成后在此预览字幕内容…",
                        )
                # ── 模型与 ASR 设置（SRT 页专用）──────────────────
                with gr.Accordion("模型与 ASR 设置 Model & ASR Settings",
                                  open=False):
                    srt_model = gr.Radio(
                        label="ASR 模型 Model",
                        choices=_SRT_MODEL_CHOICES,
                        value=(SRT_ASR or "qwen3_asr"),
                        info="qwen3_asr（默认）= Qwen3-ASR + Qwen3-ForcedAligner，"
                             "词级时间轴并按标点断句（首次约 2.3 GB 下载）；"
                             "sensevoice = silero VAD 分段 + SenseVoice，段级时间轴"
                             "（复用已缓存权重，无下载，精度较低）。",
                    )
                    with gr.Row():
                        srt_device = gr.Dropdown(
                            label="推理设备 Device",
                            choices=_DEVICE_CHOICES, value="auto", scale=1,
                            info="auto = 引擎自动（cuda > mps > cpu）",
                        )
                        srt_language = gr.Dropdown(
                            label="语种 Language",
                            choices=_LANG_CHOICES, value="Auto", scale=1,
                            info="Auto = 模型自动判断。",
                        )
                    with gr.Row():
                        srt_gap = gr.Slider(
                            label="分段合并间隙 Merge Gap (s)",
                            minimum=0.0, maximum=2.0, step=0.1, scale=1,
                            value=float(SRT_VAD_MERGE_GAP),
                            info="VAD 段：间隔小于该值的相邻语音段并成一句。",
                        )
                        srt_min_speech = gr.Slider(
                            label="最短语音段 Min Speech (s)",
                            minimum=0.0, maximum=2.0, step=0.1, scale=1,
                            value=float(SRT_VAD_MIN_SPEECH),
                            info="VAD 段：短于该时长的段丢弃。",
                        )
                    with gr.Row():
                        srt_itn = gr.Checkbox(
                            label="文本规范化 ITN", value=SRT_ITN, scale=1,
                            info="数字/标点规范化（仅 sensevoice）。",
                        )
                        srt_punctuation = gr.Checkbox(
                            label="输出标点 Punctuation", value=SRT_PUNCTUATION,
                            scale=1,
                            info="关掉则字幕只留正文（断句仍按标点判断）。",
                        )
                        srt_enum_space = gr.Checkbox(
                            label="顿号转空格 、→空格",
                            value=SRT_ENUM_COMMA_AS_SPACE, scale=1,
                            info="列举用空格分隔：赣州、贵阳 → 赣州 贵阳。",
                        )
                    with gr.Row():
                        srt_width = gr.Slider(
                            label="每行宽度 Max Line Width",
                            minimum=16, maximum=64, step=2, scale=1,
                            value=SRT_MAX_LINE_WIDTH,
                            info="CJK 按 2 计（32 ≈ 16 汉字）。",
                        )
                        srt_lines = gr.Slider(
                            label="每屏行数 Max Lines",
                            minimum=1, maximum=3, step=1, scale=1,
                            value=SRT_MAX_LINES,
                        )
                        srt_min_cue = gr.Slider(
                            label="单条最小宽度 Min Cue Width",
                            minimum=0, maximum=24, step=2, scale=1,
                            value=SRT_MIN_CUE_WIDTH,
                            info="低于此宽度的碎条并入相邻条（0 = 关闭）。",
                        )
                    with gr.Row():
                        srt_block = gr.Slider(
                            label="单条最长秒数 Max Block (s)",
                            minimum=1.0, maximum=12.0, step=0.5, scale=1,
                            value=float(SRT_MAX_BLOCK_SECONDS),
                            info="超时后优先顺延到最近的标点处断条。",
                        )
                        srt_max_gap = gr.Slider(
                            label="句间断句间隙 Max Gap (s)",
                            minimum=0.2, maximum=3.0, step=0.1, scale=1,
                            value=float(SRT_MAX_GAP_SECONDS),
                            info="词间停顿超过该值即断条。",
                        )
                        srt_min_block = gr.Slider(
                            label="单条最短秒数 Min Block (s)",
                            minimum=0.0, maximum=3.0, step=0.1, scale=1,
                            value=float(SRT_MIN_BLOCK_SECONDS),
                            info="不足则向后延长显示。",
                        )
                    gr.Markdown(
                        "断条优先标点（句末成句即断、句内标点作为回退点），"
                        "避免把词组从中间切开；本区设置只在当前进程内生效，"
                        "持久化修改请编辑 **src/config.py** 顶部变量或设置同名"
                        "环境变量。"
                    )

        # ── 事件 ─────────────────────────────────────────

        def _page_banner():
            """页面（会话）打开时写入起始行：之后日志只从此刻开始记录。"""
            buf = []
            buf.append(_term_line(
                "INFO", "会话开始（本页于 %s 打开）"
                % datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            return buf, _term_html(buf)

        def _asr_on_upload(audio, device_v, buf):
            """上传参考音频后立即用 SenseVoice 转写，文本回填 ASR 显示框。"""
            if not audio:
                buf.append(_term_line("INFO", "已清除参考音频。"))
                return gr.update(value=""), _term_html(buf), buf
            _CTX_BUF.set(buf)
            try:
                cfg = _cfg("", device_v, ref_audio=audio)
                text_out = _transcribe_ref(cfg, logger)
                if not text_out:
                    buf.append(_term_line(
                        "WARN", "ASR 转写结果为空（参考音频可能为静音）。"))
                    return gr.update(value=""), _term_html(buf), buf
                buf.append(_term_line(
                    "INFO", "ASR 转写完成，可修改后用于生成。"))
                return gr.update(value=text_out), _term_html(buf), buf
            except Exception as e:
                logger.exception("参考音频 ASR 失败")
                buf.append(_term_line(
                    "ERROR", f"ASR 失败: {type(e).__name__}: {e}"))
                return gr.update(value=""), _term_html(buf), buf
            finally:
                _CTX_BUF.set(None)

        def _load_txt(file_path, buf):
            """读取 txt 文件内容填入待合成文本框（UploadButton 值兼容单路径/列表）。"""
            if isinstance(file_path, (list, tuple)):
                file_path = file_path[0] if file_path else None
            if not file_path:
                return gr.update(), _term_html(buf), buf
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.exception("读取 txt 失败")
                buf.append(_term_line(
                    "ERROR", f"读取 txt 失败: {type(e).__name__}: {e}"))
                return gr.update(), _term_html(buf), buf
            buf.append(_term_line(
                "INFO", "已读入文本文件 %s（%d 字符）"
                % (os.path.basename(file_path), len(content))))
            return gr.update(value=content), _term_html(buf), buf

        def _clone_slots(paths: list) -> list:
            """按已产出的结果列表生成 _MAX_DRAWS 个音频槽更新（未产出则隐藏）。"""
            return [
                gr.update(visible=True, value=paths[i])
                if i < len(paths) else gr.update(visible=False)
                for i in range(_MAX_DRAWS)
            ]

        def _clone_fn(text_v, ref_aud, ref_txt, model_v, device_v, lang_v,
                      draw_v, buf):
            """点击生成：按底部「模型与运行设置」逐次抽卡，日志写入终端框。

            生成器事件处理器：每合成一个结果立即 yield 上屏（该结果与当前
            终端日志同一次更新），不再等全部抽卡结束才显示。

            生成参数不来自界面：config.py 顶部常量（默认最高质量档）由各
            模型核心在拼 CLI 时消费，此处不传 gen_kwargs。
            """
            draw_v = max(1, min(int(draw_v or 2), _MAX_DRAWS))
            if not text_v or not text_v.strip():
                buf.append(_term_line("WARN", "未输入待合成文本，已取消。"))
                yield (*_clone_slots([]), _term_html(buf), buf)
                return
            if not ref_aud:
                buf.append(_term_line("WARN", "未上传参考音频，已取消。"))
                yield (*_clone_slots([]), _term_html(buf), buf)
                return
            _CTX_BUF.set(buf)
            results: list = []
            try:
                m = (model_v or "omnivoice").strip().lower()
                cfg = _cfg(m, device_v)
                lang = None if (lang_v or "Auto") == "Auto" else lang_v
                # 输出命名：<参考音频名，去扩展名>.<unix秒>.wav（同秒冲突由
                # pipeline 递增秒数）；gradio 上传路径保留原始文件名
                audio_base = (os.path.splitext(os.path.basename(ref_aud or ""))[0]
                              or "audio")
                for i in range(draw_v):
                    if i:
                        buf.append(_term_line(
                            "INFO", f"第 {i + 1}/{draw_v} 次抽卡 …"))
                        # 重绘一次终端：上一轮结果已上屏，先报进度再开跑
                        yield (*_clone_slots(results), _term_html(buf), buf)
                    result = synthesize(
                        cfg, logger,
                        text=text_v,
                        language=lang,
                        ref_audio=ref_aud,
                        ref_text=(ref_txt or None),
                        out_dir=_TMP_DIR,
                        out_name=audio_base,
                    )
                    results.append(result.out_path)
                    buf.append(_term_line(
                        "INFO", "第 %d/%d 个结果完成: %s（%.1f 秒）"
                        % (i + 1, draw_v, os.path.basename(result.out_path),
                           result.duration_sec)))
                    yield (*_clone_slots(results), _term_html(buf), buf)
                buf.append(_term_line(
                    "INFO", f"生成完成 共 {draw_v} 个结果（模型 {m}）。"))
                yield (*_clone_slots(results), _term_html(buf), buf)
            except Exception as e:
                logger.exception("生成失败")
                buf.append(_term_line(
                    "ERROR", f"生成失败: {type(e).__name__}: {e}"))
                # 已成功的结果保留，便于直接试听
                yield (*_clone_slots(results), _term_html(buf), buf)
            finally:
                # 生成完（无论成败）立即卸载引擎/模型进程内状态
                _CTX_BUF.set(None)
                release()

        def _hy_translate_fn(src_v, prompt_v, buf):
            """点击执行翻译：Hy-MT2（llama-cli 子进程）推理，日志写入终端框。"""
            if not src_v or not src_v.strip():
                buf.append(_term_line("WARN", "粤语翻译：未输入源文案，已取消。"))
                return gr.update(value=""), _term_html(buf), buf
            _CTX_BUF.set(buf)
            try:
                out = hymt2_translate(
                    text=src_v, prompt=prompt_v or None, logger=logger)
                buf.append(_term_line(
                    "INFO", "粤语翻译完成（%d 字符）。" % len(out)))
                return gr.update(value=out), _term_html(buf), buf
            except Exception as e:
                logger.exception("粤语翻译失败")
                buf.append(_term_line(
                    "ERROR", f"粤语翻译失败: {type(e).__name__}: {e}"))
                return gr.update(value=""), _term_html(buf), buf
            finally:
                _CTX_BUF.set(None)

        ref_audio.change(
            _asr_on_upload,
            inputs=[ref_audio, device, log_state],
            outputs=[asr_text, terminal, log_state],
        )
        txt_file.upload(
            _load_txt,
            inputs=[txt_file, log_state],
            outputs=[text, terminal, log_state],
        )
        btn.click(
            _clone_fn,
            inputs=[text, ref_audio, asr_text, model, device, language,
                    draw_count, log_state],
            outputs=[*outputs, terminal, log_state],
        )
        def _srt_fn(hotwords_v, audio, model_v, device_v, lang_v, itn_v, punct_v,
                    enum_space_v, gap_v, min_speech_v, width_v, lines_v,
                    min_cue_v, block_v, max_gap_v, min_block_v, buf):
            """点击生成字幕：src.subtitle.subtitles（Qwen3-ASR+ForcedAligner 词级
            或 VAD+SenseVoice 段级）→ SRT 文件 + 预览，日志入终端框。

            热词框内容每次提交都写回根目录 hotword.txt（热记录），并经
            cfg.srt_hotwords 传给 subtitle（仅 qwen3_asr 路径生效）。
            """
            if not audio:
                buf.append(_term_line("WARN", "未上传音频文件，已取消。"))
                return gr.update(), gr.update(), _term_html(buf), buf
            _CTX_BUF.set(buf)
            try:
                m = (model_v or "qwen3_asr").strip().lower()
                hot_v = (hotwords_v or "").strip()
                _write_hotwords(hot_v)
                cfg = _cfg("", device_v, srt_asr=m, srt_audio=audio,
                           srt_hotwords=hot_v,
                           language="" if (lang_v or "Auto") == "Auto"
                           else lang_v)
                audio_base = (os.path.splitext(os.path.basename(audio))[0]
                              or "audio")
                buf.append(_term_line("INFO", "生成字幕（ASR 模型 %s）…" % m))
                result = subtitles(
                    cfg, logger,
                    audio=audio,
                    out_dir=_TMP_DIR,
                    out_name=audio_base,
                    asr_backend=m,
                    itn=bool(itn_v),
                    punctuation=bool(punct_v),
                    enum_comma_space=bool(enum_space_v),
                    vad_merge_gap=float(gap_v or 0),
                    vad_min_speech=float(min_speech_v or 0),
                    max_line_width=int(width_v) if width_v else None,
                    max_lines=int(lines_v) if lines_v else None,
                    min_cue_width=int(min_cue_v) if min_cue_v else 0,
                    max_block_seconds=float(block_v) if block_v else None,
                    max_gap_seconds=float(max_gap_v) if max_gap_v else None,
                    min_block_seconds=(float(min_block_v)
                                       if min_block_v else 0.0),
                )
                buf.append(_term_line(
                    "INFO", "字幕完成：%d 条（%s）→ %s"
                    % (len(result.cues), result.asr_backend,
                       os.path.basename(result.srt_path))))
                return (gr.update(value=result.srt_path),
                        gr.update(value=result.srt_text),
                        _term_html(buf), buf)
            except Exception as e:
                logger.exception("字幕生成失败")
                buf.append(_term_line(
                    "ERROR", f"字幕生成失败: {type(e).__name__}: {e}"))
                return gr.update(), gr.update(), _term_html(buf), buf
            finally:
                # 生成完（无论成败）立即卸载引擎/模型进程内状态
                _CTX_BUF.set(None)
                release()

        srt_btn.click(
            _srt_fn,
            inputs=[srt_hotwords, srt_audio, srt_model, srt_device, srt_language,
                    srt_itn, srt_punctuation, srt_enum_space, srt_gap,
                    srt_min_speech, srt_width, srt_lines, srt_min_cue,
                    srt_block, srt_max_gap, srt_min_block, log_state],
            outputs=[srt_file, srt_preview, terminal, log_state],
        )
        hy_btn.click(
            _hy_translate_fn,
            inputs=[hy_src, hy_prompt, log_state],
            outputs=[hy_out, terminal, log_state],
        )
        demo.load(_page_banner, outputs=[log_state, terminal])
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
