"""
推理引擎运行器：audio.cpp（audiocpp_cli，ggml 框架）

audio.cpp 是单一 C++ 引擎二进制，通过 `--family <族>` 支持多种音频模型
（omnivoice / index_tts2 / sense_asr 等），本文件是模型无关的运行层：
- 定位/自动构建 `audiocpp_cli`（config.AUDIOCPP_BIN 显式指定，否则在
  vendor/audiocpp 下查找已构建产物，都没有才 clone + cmake 构建）
- 设备 → `--backend` 映射（cuda / metal / cpu），GPU 初始化失败自动回退 CPU
- 子进程封装：捕获 stderr、失败尾部诊断、AUDIOCPP_DEBUG 全量透传

各模型的具体 CLI 参数（task/family/选项）由对应模型核心构造，
本文件只负责执行与环境，不感知具体模型。
"""

from __future__ import annotations

import glob
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .config import (
    AUDIOCPP_BIN,
    AUDIOCPP_BUILD_ARGS,
    AUDIOCPP_DEBUG,
    AUDIOCPP_REPO,
    AUDIOCPP_SRC,
    TMP_DIR,
    VENDOR_DIR,
)

_STDERR_TAIL_LINES = 60  # 失败诊断随报错打印的 stderr 行数

# 进程内缓存
_BINARY_CACHE: str | None = None


@dataclass(frozen=True)
class ChunkInfo:
    """一段合成文本（协议保留字段；audiocpp 逐 chunk 输出未接入时恒空）。"""

    text: str


@dataclass
class AudioResult:
    """一次生成的完整结果（音频 + 采样率；采样率随模型/文件而定）。"""

    audio: np.ndarray  # mono float32 PCM
    sampling_rate: int
    chunks: list[ChunkInfo] = field(default_factory=list)


def _bin_name() -> str:
    return "audiocpp_cli.exe" if sys.platform == "win32" else "audiocpp_cli"


def _src_dir() -> str:
    return AUDIOCPP_SRC or os.path.join(VENDOR_DIR, "audiocpp")


def _find_built_binary() -> str | None:
    """在 vendor/audiocpp 的构建目录里找已编译产物（build/*/bin/audiocpp_cli）。"""
    pattern = os.path.join(_src_dir(), "build", "*", "bin", _bin_name())
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    return hits[-1] if hits else None


def _run_quiet(cmd: list[str], what: str, logger: logging.Logger,
               env: dict | None = None) -> None:
    """捕获运行子进程：成功静默；失败抛错并附输出尾部诊断。"""
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        out = (proc.stderr or proc.stdout).strip()
        raise RuntimeError("%s 失败（退出码 %d）\n%s"
                           % (what, proc.returncode,
                              out[-4000:] or "（无输出）"))


def _clone_and_build(logger: logging.Logger) -> None:
    """clone audio.cpp 并构建 audiocpp_cli（custom 模型集，仅本项目所需族）。"""
    src_dir = _src_dir()
    os.makedirs(VENDOR_DIR, exist_ok=True)
    env = dict(os.environ)
    if not os.path.isdir(src_dir):
        logger.info("  git clone %s …", AUDIOCPP_REPO)
        _run_quiet(["git", "clone", "--depth", "1", AUDIOCPP_REPO, src_dir],
                   "git clone audio.cpp", logger, env=env)

    # 只构建本项目的模型族，避免全套 62+ 族（耗时/体积）
    models = "omnivoice,index_tts2,sense_asr"
    args = [
        "-S", src_dir, "-B", os.path.join(src_dir, "build", "audiocpp"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DAUDIOCPP_MODEL_SET=custom", f"-DAUDIOCPP_MODELS={models}",
    ]
    # macOS：Apple clang 不带 OpenMP，须用 brew libomp。include 路径与
    # -fopenmp flag 直接拼进 OpenMP_C/CXX_FLAGS（cmake 探测与后续编译
    # 都依赖它找 omp.h / libomp.dylib）；C_INCLUDE_PATH 作双保险一并传入。
    if sys.platform == "darwin":
        for home in ("/opt/homebrew", "/usr/local"):
            inc = os.path.join(home, "opt", "libomp", "include")
            lib = os.path.join(home, "opt", "libomp", "lib", "libomp.dylib")
            if os.path.isdir(inc) and os.path.isfile(lib):
                env.setdefault("C_INCLUDE_PATH", inc)
                env.setdefault("CPLUS_INCLUDE_PATH", inc)
                args += [
                    f"-DOpenMP_C_FLAGS=-Xclang -fopenmp -I{inc}",
                    f"-DOpenMP_CXX_FLAGS=-Xclang -fopenmp -I{inc}",
                    "-DOpenMP_C_LIB_NAMES=libomp",
                    "-DOpenMP_CXX_LIB_NAMES=libomp",
                    f"-DOpenMP_libomp_LIBRARY={lib}",
                ]
                break
    if AUDIOCPP_BUILD_ARGS:
        args.extend(shlex.split(AUDIOCPP_BUILD_ARGS))
    logger.info("  cmake 配置（custom: %s）…", models)
    _run_quiet(["cmake", *args], "cmake 配置", logger, env=env)
    build_dir = os.path.join(src_dir, "build", "audiocpp")
    logger.info("  cmake 编译（多核，请耐心等待）…")
    _run_quiet(["cmake", "--build", build_dir, "--target", "audiocpp_cli",
                "-j", str(os.cpu_count() or 4)],
               "cmake 编译", logger, env=env)


def _ensure_binary(logger: logging.Logger) -> str:
    """定位 audiocpp_cli（进程内缓存）；不存在则自动 clone + 构建到 vendor/。"""
    global _BINARY_CACHE
    if _BINARY_CACHE:
        return _BINARY_CACHE
    if AUDIOCPP_BIN and os.path.isfile(AUDIOCPP_BIN):
        logger.info("使用 AUDIOCPP_BIN: %s", AUDIOCPP_BIN)
        _BINARY_CACHE = AUDIOCPP_BIN
        return _BINARY_CACHE
    built = _find_built_binary()
    if built:
        _BINARY_CACHE = built
        logger.info("audiocpp_cli 就绪: %s", built)
        return _BINARY_CACHE
    logger.info("未找到 audiocpp_cli，clone + 编译 audio.cpp（首次较久，"
                "产物在项目内 vendor/audiocpp，gitignore）…")
    _clone_and_build(logger)
    built = _find_built_binary()
    if not built:
        raise RuntimeError("audio.cpp 编译失败：未找到 audiocpp_cli")
    _BINARY_CACHE = built
    logger.info("audiocpp_cli 就绪: %s", built)
    return _BINARY_CACHE


# ── 设备 → --backend ──────────────────────────────────────

def _backend_flag(device: str) -> str:
    """cfg.device → audiocpp --backend 值。

    cuda→cuda，mps→metal，xpu→cpu（audio.cpp 无 SYCL 后端），cpu→cpu，
    留空→best（引擎自动选择）。
    """
    return {
        "cuda": "cuda",
        "mps": "metal",
        "xpu": "cpu",
        "cpu": "cpu",
        "": "best",
    }.get(device or "", "best")


def _backend_init_failed(stderr: bytes) -> bool:
    """stderr 是否表明后端初始化失败（设备名无效/无可用后端）。"""
    text = stderr.decode("utf-8", "replace")
    return any(k in text for k in (
        "backend_init failed",
        "no backend available",
        "not found. Available:",
        "ggml_backend_load_all",
        "failed to initialize",
    ))


def _stderr_tail(stderr: bytes) -> str:
    """取 stderr 最后 N 行（失败诊断用），空则给提示。"""
    text = stderr.decode("utf-8", "replace").strip()
    if not text:
        return "（无 stderr 输出）"
    lines = text.splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


# ── 子进程执行 ────────────────────────────────────────────

def run_cli(cmd: list[str], device: str, logger: logging.Logger,
            *, input_text: str | None = None, cwd: str | None = None) -> str:
    """执行 audiocpp_cli；失败抛 RuntimeError（带 stderr 尾部），成功返回 stdout。

    - cwd：audio.cpp 的资源文件（如 silero_vad）相对仓库根，调用方按需传入
      _src_dir()；None 表示继承当前工作目录。
    - GPU 后端初始化失败自动回退 CPU 重试一次（符合设备加速优先级规则）。
    """
    env = dict(os.environ)
    t0 = time.time()
    logger.info("  audiocpp_cli 生成中（%s）…", _backend_flag(device))
    rc = subprocess.run(cmd, env=env, cwd=cwd,
                        input=input_text.encode("utf-8") if input_text else None,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_fail: bytes | None = None
    backend = _backend_flag(device)
    if (rc.returncode != 0 and backend not in ("cpu", "best")
            and _backend_init_failed(rc.stderr)):
        first_fail = rc.stderr
        logger.warning("%s 初始化失败（退出码 %d），改用 CPU 重试 …",
                       backend, rc.returncode)
        cmd = _swap_backend(cmd, "cpu")
        rc = subprocess.run(cmd, env=env, cwd=cwd,
                            input=input_text.encode("utf-8")
                            if input_text else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if AUDIOCPP_DEBUG:
        for blob in (first_fail, rc.stderr):
            if blob:
                sys.stderr.write(blob.decode("utf-8", "replace"))
        out = (rc.stdout or b"").decode("utf-8", "replace").strip()
        if out:
            sys.stderr.write(out + "\n")
    if rc.returncode != 0:
        raise RuntimeError("audiocpp_cli 失败（退出码 %d）\n%s"
                           % (rc.returncode, _stderr_tail(rc.stderr)))
    # stdout（family/task/text_output 等）由调用方按需解析与展示，
    # 这里不逐行回显，避免 ASR 转写文本重复打印
    out = (rc.stdout or b"").decode("utf-8", "replace").strip()
    logger.info("  audiocpp_cli 完成: %.1fs", time.time() - t0)
    return out


def _swap_backend(cmd: list[str], backend: str) -> list[str]:
    """把 cmd 中的 --backend <旧值> 替换为 CPU（找不到则追加）。"""
    if "--backend" in cmd:
        i = cmd.index("--backend")
        cmd = list(cmd)
        cmd[i + 1] = backend
        return cmd
    return cmd + ["--backend", backend]


def ensure_tmp_dir() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)
