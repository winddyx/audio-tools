"""
OmniVoice 配音工具 — ASR 参考音频转写（SenseVoiceSmall-GGUF，FunASR llama.cpp runtime）

模型：FunAudioLLM/SenseVoiceSmall-GGUF 的 Q8_0 量化（~235 MB，与 f16 同精度；
SAN-M 编码器 + CTC，中/英等，CPU 上约 20× 实时），配套 FunAudioLLM/fsmn-vad-GGUF
做长音频切分。由官方预编译二进制 `llama-funasr-sensevoice`（whisper.cpp 风格，
零 Python 依赖）直接输出转写文本。

- 权重：sensevoice-small-q8.gguf + fsmn-vad.gguf 经 huggingface_hub 下载，
  落 HF 默认缓存（符合项目 HF 管理规则：本地优先 + 镜像兜底 + 进度条）。
- 二进制：FUNASR_LLAMACPP_BIN 显式指定；否则自动从 modelscope/FunASR
  GitHub Releases 下载预编译包（v1.4.3，macOS arm64 / Linux x64）解压到
  项目内 vendor/funasr-llamacpp/（gitignore），失败时给出手动安装提示。
- 转写：llama-funasr-sensevoice 默认输出纯文本（不含 <|lang|>/<|emo|> 标签，
  --keep-tags 才保留），直接作为语音克隆参考文本；--transcribe 子命令打印同样内容。
- 语言：模型自动检测，--language / --lang-sym 仅作记录，不强制作业。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request

from .config import Config
from .hf import _hf_download
from settings import (
    ASR_GGUF_BASE,
    ASR_GGUF_REPO,
    ASR_VAD_BASE,
    ASR_VAD_REPO,
    FUNASR_LLAMACPP_BIN,
)

_ASR_BIN_CACHE = None    # 进程内缓存：llama-funasr-sensevoice 路径
_ASR_GGUF_CACHE = None   # 进程内缓存：(q8 路径, vad 路径)


def _asr_binary(logger: logging.Logger) -> str:
    """定位 llama-funasr-sensevoice 二进制；不存在则自动下载预编译包到 vendor/。"""
    global _ASR_BIN_CACHE
    if _ASR_BIN_CACHE:
        return _ASR_BIN_CACHE
    if FUNASR_LLAMACPP_BIN and os.path.isfile(FUNASR_LLAMACPP_BIN):
        _ASR_BIN_CACHE = FUNASR_LLAMACPP_BIN
        return _ASR_BIN_CACHE

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor_dir = os.path.join(project_root, "vendor", "funasr-llamacpp")
    bin_path = os.path.join(vendor_dir, "llama-funasr-sensevoice")
    if os.path.isfile(bin_path):
        _ASR_BIN_CACHE = bin_path
        return _ASR_BIN_CACHE

    # 自动下载官方预编译包（v1.4.3）
    asset = _funasr_release_asset()
    if not asset:
        raise RuntimeError(
            "未找到 llama-funasr-sensevoice 二进制；可手动下载 FunASR "
            "GitHub Releases（tag v1.4.3，funasr-llamacpp-*）解压后，用 "
            "FUNASR_LLAMACPP_BIN 指向它")
    logger.info("⏳ 下载 FunASR llama.cpp runtime（%s）…", asset["name"])
    url = ("https://github.com/modelscope/FunASR/releases/download/"
           f"v1.4.3/{asset['name']}")
    os.makedirs(vendor_dir, exist_ok=True)
    tmp = os.path.join(vendor_dir, asset["name"])
    try:
        urllib.request.urlretrieve(url, tmp)
        if asset["name"].endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(tmp) as zf:
                zf.extractall(vendor_dir)
        else:
            shutil.unpack_archive(tmp, vendor_dir)
        os.remove(tmp)
    except Exception as e:
        raise RuntimeError(f"下载 FunASR llama.cpp runtime 失败: {e}") from e
    if not os.path.isfile(bin_path):
        raise RuntimeError(
            f"解压后未找到 {bin_path}；可手动设置 FUNASR_LLAMACPP_BIN 指定二进制")
    _ASR_BIN_CACHE = bin_path
    logger.info("✓ llama-funasr-sensevoice 就绪: %s", bin_path)
    return _ASR_BIN_CACHE


def _funasr_release_asset() -> dict | None:
    """按平台选择 v1.4.3 的预编译包资产名。"""
    import platform
    if sys.platform == "darwin":
        if platform.machine() == "arm64":
            return {"name": "funasr-llamacpp-macos-arm64.tar.gz"}
        return None  # Intel mac 无官方预编译
    if sys.platform.startswith("linux"):
        return {"name": "funasr-llamacpp-linux-x64-avx2.tar.gz"}
    if sys.platform == "win32":
        return {"name": "funasr-llamacpp-windows-x64.zip"}
    return None


def _ensure_gguf(logger: logging.Logger) -> tuple[str, str]:
    """定位/下载 q8 GGUF + fsmn-vad GGUF（HF 默认缓存；本地优先 + 镜像兜底）。"""
    global _ASR_GGUF_CACHE
    if _ASR_GGUF_CACHE:
        return _ASR_GGUF_CACHE
    logger.info("⏳ 定位 SenseVoiceSmall-GGUF（%s，本地优先）…", ASR_GGUF_REPO)
    t0 = time.time()
    model = _hf_download(ASR_GGUF_REPO, ASR_GGUF_BASE)
    vad = _hf_download(ASR_VAD_REPO, ASR_VAD_BASE)
    logger.info("✓ ASR GGUF 就绪: %.1fs\n  model: %s\n  vad  : %s",
                time.time() - t0, model, vad)
    _ASR_GGUF_CACHE = (model, vad)
    return _ASR_GGUF_CACHE


def _transcribe_ref(cfg: Config, logger: logging.Logger) -> str:
    """用 llama-funasr-sensevoice 转写参考音频，返回纯文本。

    - 供 --transcribe 子命令与 TTS 语音克隆路径共用；
    - 音频原路径直接交给二进制（内部加载/重采样）；长音频由 fsmn-vad 切分；
    - 不加载任何 TTS 模型（GGUF 推理在 C++ 侧）。
    """
    if not cfg.ref_audio or not os.path.isfile(cfg.ref_audio):
        # 抛异常而非 sys.exit：web.py 在进程内调用本函数，exit 会杀死
        # Gradio 服务器；CLI 侧由 cli.py 的 main() 捕获并退出
        raise ValueError(
            "请设置有效的参考音频路径（--transcribe <ref_audio>）")

    # --asr-model 传本地 .gguf 时直接使用（跳过下载）
    model, vad = _ensure_gguf(logger)
    if cfg.asr_model and os.path.isfile(cfg.asr_model):
        model = os.path.abspath(cfg.asr_model)

    binary = _asr_binary(logger)
    cmd = [binary, "-m", model, "--vad", vad, "-a", cfg.ref_audio]
    logger.info("⏳ 转写中（SenseVoice GGUF Q8_0）…")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "llama-funasr-sensevoice 转写失败（退出码 %d）\n%s"
            % (proc.returncode, (proc.stderr or proc.stdout).strip()[-800:]))
    text = (proc.stdout or "").strip()
    logger.info("✓ 转写完成: %.1fs", time.time() - t0)
    return text
