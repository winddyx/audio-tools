"""
GGUF 推理后端：C++/GGML 移植版（ServeurpersoCom/omnivoice.cpp）
+ Serveurperso/OmniVoice-GGUF BF16 权重。

与 transformers 后端（src/backends/transformers.py）接口完全一致，由
settings.BACKEND 选择（默认 "gguf"）：

    _load_model(cfg, logger) → 句柄（.sampling_rate = 24000）
    generate(cfg, logger, text=…, language=…, ref_audio=…, ref_text=…,
             instruct=…, **gen_kwargs) → [音频数组]

模型管理（遵循项目规则）：
- 两个 GGUF（omnivoice-base-BF16.gguf + omnivoice-tokenizer-BF16.gguf）统一经
  huggingface_hub 下载，落 HF 默认缓存；本地优先 + hf-mirror 兜底（src/hf.py）。
- 推理二进制 omnivoice-tts：settings.CPP_BIN 显式指定；否则自动 clone + 编译
  omnivoice.cpp 到项目内 vendor/omnivoice.cpp/（gitignore，首次约 10-20 分钟）；
  settings.CPP_SRC 可指向已有源码目录、settings.CPP_BUILD_ARGS 可追加 cmake 参数。
- 设备映射（GGML_BACKEND）：mps→MTL0，cuda→CUDA0，cpu→CPU；xpu/留空
  交给运行时自动选择（SYCL 未在官方 backend 表；注意 ggml 的 Metal 设备名是
  MTL0 而非 "Metal"）。设备推理失败自动回退 CPU 重试一次（符合设备加速优先级规则）。
- 输出 24 kHz mono WAV（GGUF 量化随文件而定，DTYPE 不适用；默认 BF16，
  可在 settings.py 用 GGUF_BASE / GGUF_CODEC 切回 Q8_0 等变体）。

以上可调项统一在项目根 settings.py 中维护，同名环境变量可运行时覆盖。
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
from settings import (
    CPP_BIN,
    CPP_BUILD_ARGS,
    CPP_SRC,
    GGUF_BASE,
    GGUF_CODEC,
    GGUF_DEBUG,
    GGUF_REPO,
)

# 上游仓库固定地址（非可调设置，仅在自动 clone 时使用）
_CPP_REPO = "https://github.com/ServeurpersoCom/omnivoice.cpp.git"

_SAMPLING_RATE = 24000  # omnivoice.cpp 输出固定 24 kHz mono

# 项目根 = src/backends/gguf.py → src/backends → src → 项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_VENDOR_DIR = os.path.join(_PROJECT_ROOT, "vendor")
_CPP_SRC_DEFAULT = os.path.join(_VENDOR_DIR, "omnivoice.cpp")
_TMP_DIR = os.path.join(_PROJECT_ROOT, ".tmp")


def _bin_name() -> str:
    return "omnivoice-tts.exe" if sys.platform == "win32" else "omnivoice-tts"


_STDERR_TAIL_LINES = 60  # 生成失败时随报错打印的 stderr 行数


def _backend_init_failed(stderr: bytes) -> bool:
    """stderr 是否表明 GGML 后端初始化失败（设备名无效/无可用后端）。

    设备回退只针对这类启动期失败；输入错误（如非法 instruct）等其它
    退出不应触发 CPU 重试，避免误导性警告与重复加载。
    """
    text = stderr.decode("utf-8", "replace")
    return any(k in text for k in (
        "backend_init failed",
        "no backend available",
        "not found. Available:",
    ))


def _stderr_tail(stderr: bytes) -> str:
    """取 stderr 最后 N 行（失败诊断用），空则给提示。"""
    text = stderr.decode("utf-8", "replace").strip()
    if not text:
        return "（无 stderr 输出）"
    lines = text.splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def _built_binary() -> str:
    return os.path.join(_cpp_src_dir(), "build", _bin_name())


def _cpp_src_dir() -> str:
    return CPP_SRC or _CPP_SRC_DEFAULT


def _run(cmd: list[str]) -> None:
    """流式运行子进程（clone/编译/推理进度直接透传终端）。"""
    subprocess.run(cmd, check=True)


def _ensure_tmp_dir() -> None:
    os.makedirs(_TMP_DIR, exist_ok=True)


# ── 模型权重（HF 下载）────────────────────────────────────

# 进程内缓存：_load_model / generate 每次调用都会走到这里，缓存放过一次
# 就不再重复下载定位与重复打印日志（多轮抽卡只打一次）
_GGUF_CACHE: tuple[str, str] | None = None


def _ensure_gguf(logger: logging.Logger) -> tuple[str, str]:
    """定位/下载两个 GGUF 权重（HF 默认缓存；本地优先 + 镜像兜底，进程内缓存）。"""
    global _GGUF_CACHE
    if _GGUF_CACHE:
        return _GGUF_CACHE
    logger.info("⏳ 定位 GGUF 权重（%s，本地优先）…", GGUF_REPO)
    t0 = time.time()
    base = _hf_download(GGUF_REPO, GGUF_BASE)
    codec = _hf_download(GGUF_REPO, GGUF_CODEC)
    logger.info("✓ GGUF 就绪: %.1fs\n  base : %s\n  codec: %s",
                time.time() - t0, base, codec)
    _GGUF_CACHE = (base, codec)
    return _GGUF_CACHE


# ── 推理二进制（clone + 编译）─────────────────────────────


def _clone_cpp(logger: logging.Logger) -> None:
    os.makedirs(_VENDOR_DIR, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "--recurse-submodules",
           "--shallow-submodules", _CPP_REPO, _cpp_src_dir()]
    logger.info("  git clone %s …", _CPP_REPO)
    _run(cmd)


def _build_cpp(logger: logging.Logger) -> None:
    args = ["-DCMAKE_BUILD_TYPE=Release"]
    if sys.platform == "darwin":
        args.append("-DGGML_METAL=ON")
    if CPP_BUILD_ARGS:
        args.extend(shlex.split(CPP_BUILD_ARGS))
    src_dir = _cpp_src_dir()
    build_dir = os.path.join(src_dir, "build")
    logger.info("  cmake 配置（%s）…", " ".join(args))
    _run(["cmake", "-B", build_dir, "-S", src_dir, *args])
    logger.info("  cmake 编译（多核，请耐心等待）…")
    _run(["cmake", "--build", build_dir, "-j"])


def _ensure_binary(logger: logging.Logger) -> str:
    """定位 omnivoice-tts 二进制（进程内缓存）；不存在则自动 clone + 编译到 vendor/。"""
    global _BINARY_CACHE
    if _BINARY_CACHE:
        return _BINARY_CACHE
    if CPP_BIN and os.path.isfile(CPP_BIN):
        logger.info("✓ 使用 OMNIVOICE_CPP_BIN: %s", CPP_BIN)
        _BINARY_CACHE = CPP_BIN
        return _BINARY_CACHE
    built = _built_binary()
    if os.path.isfile(built):
        _BINARY_CACHE = built
        return _BINARY_CACHE
    logger.info("⏳ 未找到 omnivoice-tts 二进制，clone + 编译 omnivoice.cpp"
                "（首次约 10-20 分钟，产物在项目内 vendor/，gitignore）…")
    if not os.path.isdir(_cpp_src_dir()):
        _clone_cpp(logger)
    _build_cpp(logger)
    if not os.path.isfile(built):
        raise RuntimeError(f"omnivoice.cpp 编译失败：未生成 {built}")
    logger.info("✓ omnivoice-tts 就绪: %s", built)
    _BINARY_CACHE = built
    return _BINARY_CACHE


_BINARY_CACHE: str | None = None


# ── 设备映射与参数映射 ────────────────────────────────────


def _ggml_backend(device: str) -> str | None:
    """设备 → GGML_BACKEND 环境变量（ggml 索引式设备名）。

    实测本仓库 pin 的 ggml 0.17.0 中 Metal 设备注册名为 MTL0（并非 "Metal"），
    cuda→CUDA0，mps→MTL0，cpu→CPU；xpu/未知不设置（GGML 运行时自动选择，
    SYCL 未出现在官方 backend 表）。设备名不存在时二进制会硬失败，随后由
    generate() 的 CPU 回退重试兜底。
    """
    return {"cuda": "CUDA0", "mps": "MTL0", "cpu": "CPU"}.get(device or "")


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
    """准备 GGUF 后端：确保二进制已编译、权重已下载，返回句柄（进程内只打一次日志）。"""
    first_call = _BINARY_CACHE is None or _GGUF_CACHE is None
    binary = _ensure_binary(logger)
    base, codec = _ensure_gguf(logger)
    if first_call:
        logger.info("✓ GGUF 后端就绪（%s，%d Hz，设备 %s）",
                    os.path.basename(base).replace("omnivoice-base-", "").replace(".gguf", ""),
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
        # 捕获 stderr：默认静默（ggml 内核编译 / [MaskGIT-Step] 步进等噪音全部
        # 吞掉，Python 侧打摘要即可）；GGUF_DEBUG=1 时全量透传调试。
        # 失败时保留最后 N 行用于报错，避免"详见上方输出"却无迹可查。
        rc = subprocess.run(cmd, input=payload, env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
        if (rc.returncode != 0 and backend and backend != "CPU"
                and _backend_init_failed(rc.stderr)):
            # 后端初始化失败（设备名无效/无可用后端）→ 警告并回退 CPU 重试一次
            logger.warning("⚠️ %s 初始化失败（退出码 %d），改用 CPU 重试 …",
                           backend, rc.returncode)
            env["GGML_BACKEND"] = "CPU"
            rc = subprocess.run(cmd, input=payload, env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        if rc.returncode != 0:
            raise RuntimeError(
                "omnivoice-tts 生成失败（退出码 %d）\n%s"
                % (rc.returncode, _stderr_tail(rc.stderr)))
        if GGUF_DEBUG and rc.stderr:
            sys.stderr.write(rc.stderr.decode("utf-8", "replace"))

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
