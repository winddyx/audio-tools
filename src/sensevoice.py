"""
SenseVoice-Small ASR 核心（audiocpp `--family sense_asr`）

FunAudioLLM/SenseVoice-Small 的多语言语音转写（23 语言标签 + 事件/情感元
标签），在 audio.cpp 中以 `sense_asr` 族实现，内置 silero VAD 做长音频
切分（替代旧 llama-funasr 的 fsmn-vad）。音频输入须为 16 kHz mono WAV
（本模块内部自动重采样）。

权重为 audiocpp 专用 GGUF 包
（FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp，sensevoice-small-q8-audiocpp-v1.gguf）。
模型文件只在 HF 默认缓存（~/.cache/huggingface/hub）：缺失时自动经 HF 下载并
在缓存内生成引擎可用的 .gguf 别名（见 hf._ensure_gguf_file）；项目 models/ 仅
支持手工放置。

供语音克隆参考音频转写与 `--transcribe` 子命令共用。
"""

from __future__ import annotations

import logging
import os
import tempfile

from .audiocpp import _ensure_binary, ensure_tmp_dir, _src_dir
from .config import (
    ASR_GGUF_BASE,
    ASR_GGUF_REPO,
    Config,
    MODELS_DIR,
    TMP_DIR,
)
from .hf import _ensure_gguf_file

GGUF_LOCAL = os.path.join(MODELS_DIR, "SenseVoice-Small-GGUF", ASR_GGUF_BASE)


def _ensure_model(logger: logging.Logger) -> str:
    """定位 SenseVoice GGUF：手工放置的本地文件优先，否则经 HF 下载。

    audio.cpp 按真实文件扩展名识别 GGUF，HF 缓存 blob/软链路径不能直接用，
    _ensure_gguf_file 会在 HF 默认缓存仓库目录内生成带 .gguf 的硬链接别名并
    返回。模型不落工程目录（GGUF_LOCAL 仅支持用户手工放置）。
    """
    if os.path.isfile(GGUF_LOCAL):
        return GGUF_LOCAL
    if os.environ.get("ASR_GGUF_LOCAL") and os.path.isfile(os.environ["ASR_GGUF_LOCAL"]):
        return os.environ["ASR_GGUF_LOCAL"]
    return _ensure_gguf_file(ASR_GGUF_REPO, ASR_GGUF_BASE, logger)


def _to_16k_mono(src: str, dst: str) -> None:
    """把任意采样率/声道 WAV 转成 16 kHz mono（线性重采样，纯 numpy）。"""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(src, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    data = data[:, 0]
    if sr != 16000 and len(data):
        # 线性插值重采样到 16 kHz（音频较短，成本可忽略）
        n_out = int(round(len(data) * 16000 / sr))
        x_old = np.linspace(0.0, 1.0, len(data))
        x_new = np.linspace(0.0, 1.0, n_out)
        data = np.interp(x_new, x_old, data).astype("float32")
    sf.write(dst, data, 16000)


def _transcribe_ref(cfg: Config, logger: logging.Logger) -> str:
    """用 audiocpp sense_asr 转写参考音频，返回纯文本。

    - 供 --transcribe 子命令与 TTS 语音克隆路径共用；
    - 音频自动重采样为 16 kHz mono（audiocpp silero VAD 要求）；
    - 输出解析 `text_output=` 字段，默认不含 <|lang|>/<|event|> 标签。
    """
    if not cfg.ref_audio or not os.path.isfile(cfg.ref_audio):
        # 抛异常而非 sys.exit：web.py 在进程内调用本函数，exit 会杀死
        # Gradio 服务器；CLI 侧由 vc.py 的 main() 捕获并退出
        raise ValueError(
            "请设置有效的参考音频路径（--transcribe <ref_audio>）")

    binary = _ensure_binary(logger)
    model = _ensure_model(logger)
    if cfg.asr_model and os.path.isfile(cfg.asr_model):
        model = os.path.abspath(cfg.asr_model)
    ensure_tmp_dir()

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="asr-16k-", dir=TMP_DIR)
    os.close(fd)
    try:
        logger.info("音频重采样 16 kHz mono …")
        _to_16k_mono(cfg.ref_audio, tmp_wav)
        # silero_vad 资源相对 audiocpp 仓库根解析，须以源码目录为 cwd
        from .audiocpp import _backend_flag
        cmd = [binary, "--task", "asr", "--family", "sense_asr",
               "--model", model, "--backend", _backend_flag(cfg.device),
               "--audio", tmp_wav]
        logger.info("转写中（SenseVoice Q8，audiocpp sense_asr）…")
        out = run_asr(cmd, cfg.device, logger)
        text = _extract_text(out)
        logger.info("参考文本: %s", text)
        return text
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def run_asr(cmd: list[str], device: str, logger: logging.Logger) -> str:
    """执行 sense_asr 子进程（cwd=audiocpp 源码目录，供 silero_vad 定位）。"""
    from .audiocpp import run_cli
    return run_cli(cmd, device, logger, cwd=_src_dir())


def _extract_text(stdout: str) -> str:
    """从 audiocpp_cli stdout 提取转写文本（`text_output=` 行）。

    输出形如 "text_output=识别文本"；无该行时回退取最后一个非空行。
    """
    for line in stdout.splitlines():
        if line.startswith("text_output="):
            return line[len("text_output="):].strip()
    # 兜底：去掉 family/task/mode 摘要行后取剩余文本
    lines = [ln for ln in stdout.splitlines()
             if not ln.startswith(("family=", "task=", "mode=", "audio_out="))]
    return "\n".join(lines).strip()
