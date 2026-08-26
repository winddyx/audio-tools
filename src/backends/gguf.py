"""
GGUF 推理后端：C++/GGML 移植版（ServeurpersoCom/omnivoice.cpp）
+ Serveurperso/OmniVoice-GGUF Q8_0 量化权重。

与 transformers 后端（src/backends/transformers.py）接口完全一致，供
cli.py / web.py 的 _BACKEND 变量切换（默认 "gguf"）：

    _load_model(cfg, logger) → 句柄（.sampling_rate = 24000）
    generate(cfg, logger, text=…, language=…, ref_audio=…, ref_text=…,
             instruct=…, **gen_kwargs) → [音频数组]

模型管理（遵循项目规则）：
- 两个 GGUF（omnivoice-base-Q8_0.gguf + omnivoice-tokenizer-Q8_0.gguf）统一经
  huggingface_hub 下载，落 HF 默认缓存；本地优先 + hf-mirror 兜底（src/hf.py）。
- 推理二进制 omnivoice-tts：OMNIVOICE_CPP_BIN 显式指定；否则自动 clone +
  编译 omnivoice.cpp 到项目内 vendor/omnivoice.cpp/（gitignore，首次约
  10-20 分钟）；OMNIVOICE_CPP_SRC 可指向已有源码目录、
  OMNIVOICE_CPP_BUILD_ARGS 可追加 cmake 参数。
- 设备映射（GGML_BACKEND）：mps→Metal，cuda→CUDA0，cpu→CPU；xpu/留空
  交给运行时自动选择（SYCL 未在官方 backend 表）。设备推理失败自动回退
  CPU 重试一次（符合设备加速优先级规则）。
- 输出 24 kHz mono WAV（Q8_0 量化固定，DTYPE 不适用）。

参考音频转写（ref_text）仍复用 src/asr.py 的 SenseVoiceSmall，与 TTS 模型无关。
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time

from ..config import Config
from ..hf import _hf_download

# ── 可配置项（环境变量覆盖）──────────────────────────────

_GGUF_REPO = os.environ.get("OMNIVOICE_GGUF_REPO") or "Serveurperso/OmniVoice-GGUF"
_GGUF_BASE = os.environ.get("OMNIVOICE_GGUF_BASE") or "omnivoice-base-Q8_0.gguf"
_GGUF_CODEC = os.environ.get("OMNIVOICE_GGUF_CODEC") or "omnivoice-tokenizer-Q8_0.gguf"
_CPP_REPO = "https://github.com/ServeurpersoCom/omnivoice.cpp.git"

_SAMPLING_RATE = 24000  # omnivoice.cpp 输出固定 24 kHz mono

# 项目根 = src/backends/gguf.py → src/backends → src → 项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_VENDOR_DIR = os.path.join(_PROJECT_ROOT, "vendor")
_CPP_SRC = (os.environ.get("OMNIVOICE_CPP_SRC")
            or os.path.join(_VENDOR_DIR, "omnivoice.cpp"))
_TMP_DIR = os.path.join(_PROJECT_ROOT, ".tmp")


def _bin_name() -> str:
    return "omnivoice-tts.exe" if sys.platform == "win32" else "omnivoice-tts"


def _built_binary() -> str:
    return os.path.join(_CPP_SRC, "build", _bin_name())


def _run(cmd: list[str]) -> None:
    """流式运行子进程（clone/编译/推理进度直接透传终端）。"""
    subprocess.run(cmd, check=True)


def _ensure_tmp_dir() -> None:
    os.makedirs(_TMP_DIR, exist_ok=True)


# ── 模型权重（HF 下载）────────────────────────────────────


def _ensure_gguf(logger: logging.Logger) -> tuple[str, str]:
    """定位/下载两个 Q8_0 GGUF（HF 默认缓存；本地优先 + 镜像兜底）。"""
    logger.info("⏳ 定位 GGUF 权重（%s，Q8_0，本地优先）…", _GGUF_REPO)
    t0 = time.time()
    base = _hf_download(_GGUF_REPO, _GGUF_BASE)
    codec = _hf_download(_GGUF_REPO, _GGUF_CODEC)
    logger.info("✓ GGUF 就绪: %.1fs\n  base : %s\n  codec: %s",
                time.time() - t0, base, codec)
    return base, codec


# ── 推理二进制（clone + 编译）─────────────────────────────


def _clone_cpp(logger: logging.Logger) -> None:
    os.makedirs(_VENDOR_DIR, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "--recurse-submodules",
           "--shallow-submodules", _CPP_REPO, _CPP_SRC]
    logger.info("  git clone %s …", _CPP_REPO)
    _run(cmd)


def _build_cpp(logger: logging.Logger) -> None:
    args = ["-DCMAKE_BUILD_TYPE=Release"]
    if sys.platform == "darwin":
        args.append("-DGGML_METAL=ON")
    extra = os.environ.get("OMNIVOICE_CPP_BUILD_ARGS")
    if extra:
        args.extend(shlex.split(extra))
    build_dir = os.path.join(_CPP_SRC, "build")
    logger.info("  cmake 配置（%s）…", " ".join(args))
    _run(["cmake", "-B", build_dir, "-S", _CPP_SRC, *args])
    logger.info("  cmake 编译（多核，请耐心等待）…")
    _run(["cmake", "--build", build_dir, "-j"])


def _ensure_binary(logger: logging.Logger) -> str:
    """定位 omnivoice-tts 二进制；不存在则自动 clone + 编译到 vendor/。"""
    explicit = os.environ.get("OMNIVOICE_CPP_BIN")
    if explicit and os.path.isfile(explicit):
        logger.info("✓ 使用 OMNIVOICE_CPP_BIN: %s", explicit)
        return explicit
    built = _built_binary()
    if os.path.isfile(built):
        return built
    logger.info("⏳ 未找到 omnivoice-tts 二进制，clone + 编译 omnivoice.cpp"
                "（首次约 10-20 分钟，产物在项目内 vendor/，gitignore）…")
    if not os.path.isdir(_CPP_SRC):
        _clone_cpp(logger)
    _build_cpp(logger)
    if not os.path.isfile(built):
        raise RuntimeError(f"omnivoice.cpp 编译失败：未生成 {built}")
    logger.info("✓ omnivoice-tts 就绪: %s", built)
    return built


# ── 设备映射与参数映射 ────────────────────────────────────


def _ggml_backend(device: str) -> str | None:
    """设备 → GGML_BACKEND 环境变量。

    mps→Metal，cuda→CUDA0，cpu→CPU；xpu/未知不设置（GGML 运行时自动选择，
    SYCL 未出现在 omnivoice.cpp 官方 backend 表）。
    """
    return {"cuda": "CUDA0", "mps": "Metal", "cpu": "CPU"}.get(device or "")


def _gen_kwargs_to_cli(kwargs: dict, logger: logging.Logger) -> list[str]:
    """生成参数 → omnivoice-tts CLI 参数（只映射 C++ 支持的子集，其余忽略）。"""
    cli: list[str] = []
    if "num_step" in kwargs:
        cli += ["--steps", str(int(kwargs["num_step"]))]
    if "denoise" in kwargs and not kwargs["denoise"]:
        cli.append("--no-denoise")
    if "audio_chunk_duration" in kwargs:
        cli += ["--chunk-duration", str(float(kwargs["audio_chunk_duration"]))]
    if "audio_chunk_threshold" in kwargs:
        cli += ["--chunk-threshold", str(float(kwargs["audio_chunk_threshold"]))]
    if "duration" in kwargs:
        cli += ["--duration", str(float(kwargs["duration"]))]
    supported = {"num_step", "denoise", "audio_chunk_duration",
                 "audio_chunk_threshold", "duration"}
    unsupported = set(kwargs) - supported
    if unsupported:
        logger.info("ℹ️ GGUF 后端不支持以下生成参数，已忽略: %s",
                    ", ".join(sorted(unsupported)))
    return cli


# ── 对外接口（与 transformers 后端一致）──────────────────


class _ModelHandle:
    """GGUF 后端模型句柄：与 transformers 模型对象兼容的最小接口。"""

    sampling_rate = _SAMPLING_RATE

    def __init__(self, binary: str, base: str, codec: str) -> None:
        self.binary = binary
        self.base = base
        self.codec = codec


def _load_model(cfg: Config, logger: logging.Logger) -> _ModelHandle:
    """准备 GGUF 后端：确保二进制已编译、Q8_0 权重已下载，返回句柄。"""
    binary = _ensure_binary(logger)
    base, codec = _ensure_gguf(logger)
    logger.info("✓ GGUF 后端就绪（Q8_0，%d Hz，设备 %s）",
                _SAMPLING_RATE, cfg.device or "auto")
    return _ModelHandle(binary, base, codec)


def generate(cfg: Config, logger: logging.Logger, **kwargs):
    """调用 omnivoice-tts 生成音频，返回 [音频数组]（与 transformers 后端一致）。

    kwargs 支持：text / language / ref_audio / ref_text / instruct，
    以及 src/params.py 中 GGUF 支持的生成参数子集（num_step / denoise /
    audio_chunk_duration / audio_chunk_threshold / duration）。
    """
    text = kwargs.pop("text", "")
    language = kwargs.pop("language", None)
    ref_audio = kwargs.pop("ref_audio", None)
    ref_text = kwargs.pop("ref_text", None)
    instruct = kwargs.pop("instruct", None)

    if ref_audio and not ref_text:
        raise ValueError(
            "GGUF 后端语音克隆需要 ref_text（参考音频转写文本，"
            "CLI/web 会自动用 SenseVoiceSmall 转写）")

    handle = _load_model(cfg, logger)
    _ensure_tmp_dir()

    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="gguf-", dir=_TMP_DIR)
    os.close(fd)
    ref_tmp: str | None = None
    try:
        cmd = [handle.binary, "--model", handle.base, "--codec", handle.codec,
               "-o", out_wav]
        if language:
            cmd += ["--lang", str(language)]
        if ref_audio:
            cmd += ["--ref-wav", ref_audio]
            fd2, ref_tmp = tempfile.mkstemp(suffix=".txt", prefix="ref-",
                                            dir=_TMP_DIR)
            with os.fdopen(fd2, "w", encoding="utf-8") as f:
                f.write(ref_text)
            cmd += ["--ref-text", ref_tmp]
        elif instruct:
            cmd += ["--instruct", instruct]
        cmd += _gen_kwargs_to_cli(kwargs, logger)

        env = dict(os.environ)
        backend = _ggml_backend(cfg.device)
        if backend:
            env["GGML_BACKEND"] = backend

        logger.info("  ⏳ omnivoice-tts 生成中（%s）…", backend or "设备自动选择")
        t0 = time.time()
        payload = (text or "").encode("utf-8")
        rc = subprocess.run(cmd, input=payload, env=env)
        if rc.returncode != 0 and backend and backend != "CPU":
            # 设备（Metal/CUDA）推理失败 → 警告并回退 CPU 重试一次
            logger.warning("⚠️ %s 推理失败（退出码 %d），改用 CPU 重试 …",
                           backend, rc.returncode)
            env["GGML_BACKEND"] = "CPU"
            rc = subprocess.run(cmd, input=payload, env=env)
        if rc.returncode != 0:
            raise RuntimeError(
                f"omnivoice-tts 生成失败（退出码 {rc.returncode}），详见上方输出")

        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError("omnivoice-tts 未产出 WAV 文件")

        import soundfile as sf
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        logger.info("✓ 生成完成: %.1fs（%d Hz，%.1f s 音频）",
                    time.time() - t0, sr, len(data) / sr)
        return [data]
    finally:
        for p in (out_wav, ref_tmp):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
