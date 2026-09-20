"""
audio-tools — llama.cpp 单次补全的共用运行器

按 llama-completion（llama.cpp 的单次补全 CLI）子进程做一次补全，Python 只
做编排。本模块只负责"定位引擎 / 定位模型 / 调一次补全"，不含任何业务提示词：

- 引擎：LLAMA_CLI 留空时自动 clone + cmake 编译 vendor/llama.cpp（与 audiocpp
  同模式，不依赖本机 brew 安装），产物
  vendor/llama.cpp/build/bin/llama-completion（vendor 已 gitignore）；
- 模型：本地手工放置优先（local），否则经 HF 默认缓存下载并生成 .gguf 硬链接
  别名（src/hf._ensure_gguf_file），不落工程目录；
- 失败一律抛 RuntimeError（带 stderr 尾部诊断），不在入口裸奔。

调用方（src/hymt2.py 粤语翻译、src/segment_llm.py 断句辅助）各自用配置里的
参数构造 LlmParams，本文件不散落业务默认值。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

from .config import (
    LLAMA_BUILD_ARGS,
    LLAMA_CLI,
    LLAMA_DEBUG,
    LLAMA_REF,
    LLAMA_REPO,
    VENDOR_DIR,
)
from .hf import _ensure_gguf_file

_BIN_NAME = "llama-completion"


@dataclass
class LlmParams:
    """一次补全的采样与运行参数（值统一来自 config.py 顶部常量）。"""

    max_tokens: int
    temperature: str = "0.0"
    top_p: str = "1.0"
    top_k: int = 0
    repeat_penalty: str = "1.0"
    timeout: int = 600
    device: str = ""
    seed: int | None = None
    label: str = "llama-completion"


def binary(logger: logging.Logger) -> str:
    """定位 llama-completion：LLAMA_CLI 显式设置优先，否则自动构建 vendor。

    - LLAMA_CLI 非空：当作可执行名（PATH 查找）或绝对路径，找不到即报错；
    - LLAMA_CLI 留空（默认）：复用 vendor/llama.cpp/build/bin 下已有产物，
      缺失则自动 clone + cmake 编译。
    """
    if LLAMA_CLI:
        if os.path.sep in LLAMA_CLI and os.path.isfile(LLAMA_CLI):
            return LLAMA_CLI
        found = shutil.which(LLAMA_CLI)
        if found:
            return found
        raise RuntimeError(f"LLAMA_CLI 指向的可执行未找到: {LLAMA_CLI}")
    built = os.path.join(VENDOR_DIR, "llama.cpp", "build", "bin", _BIN_NAME)
    if os.path.isfile(built):
        return built
    return _ensure_built(logger)


def model_path(repo: str, file: str, local: str,
               logger: logging.Logger, desc: str = "LLM") -> str:
    """返回可直接喂给 llama-completion 的 .gguf 路径（本地手工放置优先）。"""
    if local:
        path = os.path.abspath(local)
        if not os.path.isfile(path):
            raise RuntimeError(f"{desc}本地模型（local）指向的文件不存在: {path}")
        return path
    logger.info("%s模型: %s/%s（llama.cpp）", desc, repo, file)
    return _ensure_gguf_file(repo, file, logger)


def run_once(model: str, prompt: str, logger: logging.Logger,
             params: LlmParams) -> str:
    """跑一次补全，返回模型输出文本（已清掉结束标记与首尾空白）。

    seed 为 None 时交给 llama-completion 自选；指定则固定可复现。device
    为 "cpu" 时加 --device none 强制不卸载 GPU。
    """
    cmd = [
        binary(logger),
        "-m", model,
        "-p", prompt,
        "-n", str(params.max_tokens),
        "--temp", str(params.temperature),
        "--top-p", str(params.top_p),
        "--top-k", str(params.top_k),
        "--repeat-penalty", str(params.repeat_penalty),
        "--no-display-prompt",
        "--jinja",
        "-st",
    ]
    if params.seed is not None:
        cmd += ["--seed", str(params.seed)]
    if params.device == "cpu":
        cmd += ["--device", "none"]
    elif params.device:
        cmd += ["--device", params.device]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=params.timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{params.label} 超时（>{params.timeout}s）；可调大超时或改小 "
            "输出 token 上限 / 输入长度。")
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-600:]
        raise RuntimeError(
            f"{params.label} 失败（rc={r.returncode}，{dt:.0f}s）: {tail}")
    logger.info("%s 完成: %.1fs（输出 %d 字符）",
                params.label, dt, len(r.stdout or ""))
    out = (r.stdout or "").strip()
    # llama-completion 会把结束标记（[end of text] 等）也打到 stdout，清掉
    for marker in ("[end of text]", "[end of turn]", "[EOT]"):
        while out.endswith(marker):
            out = out[:-len(marker)].rstrip()
    return out


def _run_step(cmd: list[str], logger: logging.Logger, desc: str) -> None:
    """执行 clone/编译等一次性步骤；失败抛 RuntimeError（带输出尾部）。"""
    t0 = time.time()
    try:
        if LLAMA_DEBUG:
            r = subprocess.run(cmd)          # 透传原始输出（调试用）
        else:
            r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{desc}失败：缺少命令 {e.filename}（请检查 git/cmake 是否安装）")
    if r.returncode != 0:
        tail = "" if LLAMA_DEBUG else (r.stderr or r.stdout or "").strip()[-800:]
        raise RuntimeError(f"{desc}失败（rc={r.returncode}）: {tail}")
    logger.info("%s完成（%.0fs）", desc, time.time() - t0)


def _ensure_built(logger: logging.Logger) -> str:
    """自动 clone + 编译 llama.cpp，返回 llama-completion 绝对路径。"""
    src = os.path.join(VENDOR_DIR, "llama.cpp")
    build = os.path.join(src, "build")
    binary_path = os.path.join(build, "bin", _BIN_NAME)
    if shutil.which("cmake") is None:
        raise RuntimeError("未找到 cmake，请先安装（brew install cmake）后重试。")
    if not os.path.isfile(os.path.join(src, "CMakeLists.txt")):
        logger.info("未找到 llama.cpp 源码，clone %s（分支 %s）…",
                    LLAMA_REPO, LLAMA_REF)
        os.makedirs(VENDOR_DIR, exist_ok=True)
        _run_step([
            "git", "clone", "--depth", "1", "--branch", LLAMA_REF,
            "--single-branch", "--recurse-submodules",
            LLAMA_REPO, src,
        ], logger, "git clone llama.cpp")
    logger.info("cmake 配置 vendor/llama.cpp（首次较久）…")
    cfg = ["cmake", "-S", src, "-B", build, "-DCMAKE_BUILD_TYPE=Release"]
    cfg += (LLAMA_BUILD_ARGS or "").split()
    _run_step(cfg, logger, "cmake 配置")
    logger.info("cmake 编译 %s（多核，请耐心等待）…", _BIN_NAME)
    _run_step([
        "cmake", "--build", build, "--config", "Release",
        "--target", _BIN_NAME, "--parallel",
    ], logger, "cmake 编译")
    if not os.path.isfile(binary_path):
        raise RuntimeError(f"编译完成但未找到产物: {binary_path}")
    logger.info("%s 就绪: %s", _BIN_NAME, binary_path)
    return binary_path
