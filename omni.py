"""
OmniVoice 配音工具 — 文本转语音（零样本语音克隆 / 声音设计 / 自动音色）

基于 k2-fsa/OmniVoice（HuggingFace 仓库：k2-fsa/OmniVoice），支持 600+ 语言。

模型管理规则（必须遵守）：
- 所有模型统一由 HuggingFace（huggingface_hub）管理下载，不硬编码任何本地路径。
- 模型文件一律落在 HuggingFace 默认缓存（~/.cache/huggingface，或
  HF_HOME / HF_HUB_CACHE 指定的位置）；路径只取 hf_hub_download() /
  snapshot_download() 的返回值，代码里不写死缓存路径。
- 本地优先：模型已完整缓存在本地时直接复用，不发起任何网络请求（跳过
  revision 检查与文件列表，启动快且离线可用）；HF_LOCAL_FIRST=0 可关闭
  本地优先、强制联网校验更新。
- 直连 huggingface.co 失败（超时等）且未显式设置 HF_ENDPOINT 时，自动改用
  hf-mirror.com 镜像重试一次；可用 HF_NO_MIRROR_FALLBACK=1 关闭兜底。

三种生成模式（全部复用模型原生能力，本工具只做薄封装）：
- 语音克隆：传 ref_audio（+ 可选 ref_text；ref_text 省略时默认用
  FunASR/SenseVoiceSmall 转写参考音频，不依赖 OmniVoice 内部 Whisper ASR）；
- 声音设计：传 --instruct（如 "female, low pitch, british accent"，无需参考音频）；
- 自动音色：两者都不传，模型自动选择音色。

参考音频与文本两个位置参数一一对应（<ref_audio> <text_file>）；声音设计/
自动音色模式可用 --text 指定文本文件，避免与参考音频位置参数歧义。

长文本切段与拼接、参考音频加载/重采样/转单声道、生成参数默认值全部交由模型
generate() 自身逻辑处理；生成参数默认使用模型默认值，可用环境变量覆盖
（见 _GEN_PARAM_ENVS）。

ASR（FunASR / SenseVoiceSmall，阿里巴巴，MIT）：
- 模型：FunAudioLLM/SenseVoiceSmall（funasr AutoModel 加载，含 fsmn-vad 自动切分，
  可处理任意长度音频；小模型、CPU 即可实时，参考音频转写场景绰绰有余）。
- 独立子命令 --transcribe：转写参考音频并打印文本（校对/数据集/验证用）。
- TTS 语音克隆路径：ref_text 省略时默认用 SenseVoiceSmall 转写参考音频得到
  参考文本（语音克隆本身零样本，不需要参考文本，但提供准确的 ref_text 提升
  克隆质量；已禁用 OmniVoice 内部 Whisper ASR 兜底）。
- 支持 中/英/粤/日/韩 五种语言强制转写（语言代码映射见 _LANG_CODE_TO_ASR），
  其余语言交给模型自动检测；--language 代码自动映射为 SenseVoice 语言代码。
- 权重经 funasr + huggingface_hub 从 HuggingFace 下载（hub="hf"，落 HF 默认
  缓存，符合 HF 管理规则；SenseVoiceSmall 与 fsmn-vad 在 HF 均有官方镜像）。
- 音频传原路径，funasr 内部加载/重采样，无需系统 ffmpeg；超长音频由
  fsmn-vad 自动切段（单段上限 30 s，段间拼接后返回整段文本）。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


# ── 设备检测 ──────────────────────────────────────────────


def get_best_device() -> str:
    """自动检测最佳可用设备：CUDA > MPS > CPU。"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_THREAD_CONFIGURED = False  # CPU 线程池已配置（幂等标志，避免重复设置/重复日志）


def _configure_cpu_threads(cfg: Config, logger: logging.Logger) -> None:
    """CPU 设备下显式配置 PyTorch 线程池，用满所有可用核心（仅执行一次）。

    - 线程数取值优先级：THREADS 环境变量 > OMP_NUM_THREADS > os.cpu_count()。
      torch 默认只按物理核心建线程池，Windows 超线程机器上逻辑核心会闲置；
      显式设置后全部逻辑核心参与计算，对长文本配音的 batch 算子有明显增益。
    - 只调整 torch 的线程池（torch.set_num_threads），不改环境变量、不影响
      numpy 等其他 OpenMP 库；CUDA/MPS 设备跳过（GPU 推理与线程数无关）。
    - 在模型加载前调用（加载/推理是首次算子执行点），CLI 与 web 共用
      （web 复用 omni._load_model / _asr_model，自动生效）。
    """
    global _THREAD_CONFIGURED
    if _THREAD_CONFIGURED:
        return
    if (cfg.device or "cpu") != "cpu":
        _THREAD_CONFIGURED = True
        return

    import torch
    n = os.cpu_count() or 1
    for env in ("THREADS", "OMP_NUM_THREADS"):
        v = (os.environ.get(env) or "").strip()
        if v.isdigit() and int(v) >= 1:
            n = int(v)
            break
    torch.set_num_threads(n)
    _THREAD_CONFIGURED = True
    logger.info("🧵 CPU 线程池: %d 线程（可用核心 %d；THREADS 环境变量可覆盖）",
                n, os.cpu_count() or 1)


# ── 配置 ──────────────────────────────────────────────────

@dataclass
class Config:
    """全局默认配置。可通过环境变量或 CLI 参数覆盖。"""

    # ── 模型（统一由 HuggingFace 管理；路径取自下载接口返回值）──
    model_id: str = "k2-fsa/OmniVoice"  # HuggingFace 模型 ID
    model_path: str = ""      # 非空时优先于 model_id（本地 snapshot 目录）
    device: str = ""          # 留空则自动检测（CUDA > MPS > CPU）
    dtype: str = ""           # 留空自动：CUDA 用 float16，MPS/CPU 用 float32（MPS fp16 会崩溃）

    # ── 生成模式 ──
    language: str = ""        # 语言代码/名称（如 en / zh / English）；留空 = 语言无关（模型自动判断）
    ref_audio: str = ""
    ref_text: str = ""        # 参考音频转写文本；留空则用 FunASR/SenseVoiceSmall 转写（不依赖 Whisper）
    instruct: str = ""        # 声音设计指令（如 "female, low pitch, british accent"）

    # ── 长文本配音 ──
    text_path: str = ""
    draw_count: int = 2       # 抽卡次数
    output_dir: str = ""      # 留空则输出到文本文件所在目录

    # ── ASR 子命令（可选，FunASR/SenseVoiceSmall；仅用于校对/数据集/验证）──
    transcribe: bool = False  # --transcribe：转写 ref_audio 并打印文本
    asr_model: str = ""       # SenseVoice 模型 ID/本地目录；留空默认 FunAudioLLM/SenseVoiceSmall
    asr_hub: str = ""         # 模型下载源：hf（HuggingFace，默认）/ ms（ModelScope）
    asr_vad: str = ""         # VAD 切分模型；留空默认 fsmn-vad，设 "0" 关闭
    asr_lang_sym: str = ""    # ASR 语言代码（如 zh/en）；留空则跟随 --language，再留空自动检测
    asr_region_sym: str = ""  # 保留字段（SenseVoice 不支持地区强制，已废弃）

    # 生成参数不在此配置：全部交由模型 generate() 自身默认值，
    # 如需覆盖用环境变量（见 _GEN_PARAM_ENVS）。


# ── 模型下载与缓存 ────────────────────────────────────────
#
# 模型管理规则（必须遵守）：
#   1. 所有模型统一由 HuggingFace（huggingface_hub）管理下载，不硬编码任何本地路径。
#   2. 模型文件一律落在 HuggingFace 默认缓存（~/.cache/huggingface，或
#      HF_HOME / HF_HUB_CACHE 指定的位置）；路径只取 hf_hub_download() /
#      snapshot_download() 的返回值，代码里不写死缓存路径。
#   3. 本地优先：先尝试 local_files_only 定位缓存快照，命中则零网络请求直接
#      复用（跳过 revision 检查与文件列表）；未命中/快照不完整才联网下载。
#      HF_LOCAL_FIRST=0 可关闭本地优先，强制联网校验更新。
#   4. 直连 huggingface.co 失败（超时等）且未显式设置 HF_ENDPOINT 时，
#      自动改用 hf-mirror.com 镜像重试一次；HF_NO_MIRROR_FALLBACK=1 可关闭。

_OMNIVOICE_MODEL = None   # 全局 OmniVoice 模型缓存（单例）

# 直连 HuggingFace 失败时的镜像兜底（国内网络常用）
_HF_MIRROR = "https://hf-mirror.com"


def _switch_hf_endpoint(endpoint: str) -> None:
    """运行时把 huggingface_hub 的目标 endpoint 切换到镜像。

    huggingface_hub 在 import 时就把 endpoint 固化进 URL 模板（constants.ENDPOINT
    与 file_download.HUGGINGFACE_CO_URL_TEMPLATE），仅设置环境变量不生效，
    需同步更新这些常量；版本差异导致的异常忽略，环境变量仍对新进程生效。
    """
    endpoint = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import constants as hf_constants
        from huggingface_hub import file_download as hf_file_download

        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            endpoint + "/{repo_id}/resolve/{revision}/{filename}"
        )
        # 新版 huggingface_hub 已删除 file_download 里的模板属性，用 hasattr 保护
        if hasattr(hf_file_download, "HUGGINGFACE_CO_URL_TEMPLATE"):
            hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE = (
                hf_constants.HUGGINGFACE_CO_URL_TEMPLATE
            )
    except Exception:
        pass


def _hf_download(repo_id: str, filename: str = "") -> str:
    """HuggingFace 下载单个文件/快照（遵循 HF_ENDPOINT 镜像）。

    本地优先：模型已完整缓存在本地时直接复用快照，不发起 revision 检查/
    文件列表等任何网络请求（默认，HF_LOCAL_FIRST=0 可关闭）；未命中或快照
    不完整才联网下载。直连失败且用户未显式设置 HF_ENDPOINT（也未用
    HF_NO_MIRROR_FALLBACK=1 关闭兜底）时，自动切换到 hf-mirror.com 重试一次。
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    def _local_only() -> str:
        if filename:
            return hf_hub_download(repo_id, filename,
                                   local_files_only=True)
        return snapshot_download(repo_id, local_files_only=True)

    def _remote() -> str:
        if filename:
            return hf_hub_download(repo_id, filename)
        return snapshot_download(repo_id)

    # 判断：本地已有完整模型则直接复用，跳过一切联网
    # （含 revision 检查/文件列表；HF_LOCAL_FIRST=0 可关闭本地优先）
    if os.environ.get("HF_LOCAL_FIRST", "1") != "0":
        try:
            path = _local_only()
        except Exception:
            path = ""  # 未命中/快照不完整，走联网下载
        if path:
            logging.getLogger("omni").info(
                "✅ 本地已有模型，跳过联网: %s", repo_id)
            return path

    try:
        return _remote()
    except Exception:
        # 用户已显式指定 endpoint / 明确关闭兜底：尊重其选择，直接抛错
        if os.environ.get("HF_ENDPOINT") or os.environ.get("HF_NO_MIRROR_FALLBACK"):
            raise
        logger = logging.getLogger("omni")
        logger.warning("⚠️ 直连 HuggingFace 失败（%s），改用镜像 %s 重试 …",
                       repo_id, _HF_MIRROR)
        _switch_hf_endpoint(_HF_MIRROR)
        try:
            return _remote()
        except Exception as e2:
            logger.error("⚠️ 镜像 %s 下载 %s 也失败: %s", _HF_MIRROR, repo_id, e2)
            raise


def resolve_path(model_id: str = "", local_path: str = "") -> str:
    """解析主模型路径：优先 local_path，否则从 HuggingFace 自动下载。

    snapshot_download 不传 cache_dir，由 huggingface_hub 决定缓存位置
    （默认 ~/.cache/huggingface，或遵循 HF_HOME / HF_HUB_CACHE）。
    """
    if local_path:
        return os.path.abspath(local_path)
    return _hf_download(model_id or Config.model_id)


# ── 模型加载 ──────────────────────────────────────────────
#
# OmniVoice 是 transformers PreTrainedModel，自带 config.json / tokenizer /
# audio_tokenizer（均在 k2-fsa/OmniVoice 快照内），用 from_pretrained 直接加载，
# 无需生成任何临时 config；快照内没有的辅助资源由模型内部按需经 huggingface_hub
# 下载，同样受 HF_ENDPOINT 镜像约束（参考文本转写已改由 FunASR/SenseVoiceSmall
# 承担，不走 Whisper）。


def _default_dtype(device: str) -> str:
    """默认精度：CUDA 用 float16（快），MPS/CPU 用 float32。

    torch 2.13 在 MPS 上加载 float16 权重会 SIGTRAP 崩溃（已实测复现），
    MPS 上必须用 float32；CPU 同理。可用 DTYPE 环境变量覆盖。
    """
    return "float16" if device == "cuda" else "float32"


def _load_model(cfg: Config, logger: logging.Logger):
    """加载 OmniVoice 模型（带全局缓存，避免重复加载）。"""
    global _OMNIVOICE_MODEL
    if _OMNIVOICE_MODEL is not None:
        return _OMNIVOICE_MODEL

    # CPU 设备下先把线程池配满再加载模型（幂等；web 复用此路径，自动生效）
    _configure_cpu_threads(cfg, logger)

    import torch
    from omnivoice import OmniVoice

    # 先经 resolve_path 下载/定位快照，再交给 from_pretrained：
    # 模型内部对本地目录直接复用，全程不触发额外网络请求
    resolved = resolve_path(cfg.model_id, cfg.model_path)
    logger.info("📦 模型目录: %s", resolved)

    # 默认 dtype：CUDA 用 float16（快），MPS/CPU 用 float32——
    # torch 2.13 在 MPS 上加载 float16 权重会 SIGTRAP 崩溃（已验证），
    # 且 HiggAudio tokenizer 不支持 MPS 会自动落 CPU，float32 更稳。
    # 可用 DTYPE 环境变量显式覆盖（如 float16/float32/bfloat16）。
    dtype = getattr(torch, cfg.dtype) if cfg.dtype else getattr(
        torch, _default_dtype(cfg.device)
    )

    logger.info("⏳ 加载 OmniVoice 模型（%s, %s）…", cfg.device, dtype)
    t0 = time.time()
    _OMNIVOICE_MODEL = OmniVoice.from_pretrained(
        resolved, device_map=cfg.device, dtype=dtype,
    )
    logger.info("✓ 模型加载: %.1fs (%s)", time.time() - t0, cfg.device)
    return _OMNIVOICE_MODEL


# ── ASR（FunASR / SenseVoiceSmall）───────────────────────
#
# 目标模型：SenseVoiceSmall（FunAudioLLM/SenseVoiceSmall，MIT）——多语言语音识别，
# 官方支持 中/英/粤/日/韩 五种语言强制识别 + 自动检测（auto），并输出情感/
# 事件/标点标签；小模型（234M）CPU 即可实时，参考音频转写场景绰绰有余。
# - 权重：funasr AutoModel(hub="hf") 经 huggingface_hub 从 HuggingFace 下载
#   （FunAudioLLM/SenseVoiceSmall + funasr/fsmn-vad 官方镜像），落 HF 默认缓存，
#   符合本项目的 HF 管理规则；本地优先：缓存命中直接复用本地快照，零网络请求。
# - 不依赖 transformers 的 ASR 架构（funasr 自带模型实现，仅复用 torch），
#   与 omnivoice 需要的 transformers 5.x 无版本冲突（已实测 funasr 1.4.x）。
# - VAD：默认附带 fsmn-vad 自动切分长音频（单段上限 30 s，段间合并返回整段
#   文本），参考音频多长都能处理；ASR_VAD=0 可关闭（省内存，仅适合短音频）。
# - 语言：SenseVoice 用代码（zh/en/yue/ja/ko/auto）；本工具把 --language 代码
#   映射为 SenseVoice 语言代码并强制转写（避免自动检测误判，如英语被判成韩语），
#   映射不到的语言交给模型自动检测（auto）。
# - 输出：SenseVoice 文本带 <|lang|><|emo|><|event|><|woitn|> 元数据 token，
#   返回前统一剥离，只留纯文本。
# - 接入形态：独立子命令 --transcribe；同时作为 TTS 语音克隆路径的默认参考
#   文本来源（ref_text 省略时先用 SenseVoiceSmall 转写参考音频）。

_ASR_MODEL = None  # 全局 ASR 模型缓存（懒加载单例，CLI/web 共用）

# OmniVoice 语言代码 → SenseVoice 语言代码（SenseVoice 仅支持这几种强制语言；
# 未列出的语言代码不强制语言，交给模型自动检测 auto）
_LANG_CODE_TO_ASR = {
    "zh": "zh", "yue": "yue", "en": "en", "ja": "ja", "ko": "ko",
}

# SenseVoice 输出中的元数据 token：<|zh|> <|NEUTRAL|> <|Speech|> <|woitn|> …
_ASR_TOKEN_RE = re.compile(r"<\|[^>]*\|>")


def _asr_model_id(cfg: Config) -> str:
    """ASR 模型 ID：默认 FunAudioLLM/SenseVoiceSmall（官方 HF 镜像），可用 ASR_MODEL 覆盖。"""
    return cfg.asr_model or "FunAudioLLM/SenseVoiceSmall"


def _asr_hub(cfg: Config) -> str:
    """模型下载源：默认 hf（HuggingFace，符合本项目 HF 管理规则）；ASR_HUB=ms 切 ModelScope。"""
    return (cfg.asr_hub or os.environ.get("ASR_HUB") or "hf").strip().lower()


def _asr_vad_id(cfg: Config) -> str:
    """VAD 切分模型：默认 fsmn-vad（hub=hf 时 funasr 自动映射 funasr/fsmn-vad）；
    ASR_VAD=0 关闭（仅适合短音频，省内存）。返回 "" 表示关闭。"""
    v = (cfg.asr_vad or os.environ.get("ASR_VAD") or "fsmn-vad").strip().lower()
    return "" if v in ("", "0", "none", "off", "false") else v


def _asr_language(cfg: Config) -> str:
    """把 TTS 的 --language / LANGUAGE 映射为 SenseVoice 语言代码。

    - cfg.asr_lang_sym 显式指定时（--lang-sym / ASR_LANG_SYM）优先；
    - 否则用 cfg.language 的代码映射；映射不到则返回 ""（自动检测 auto）。
    """
    sym = (cfg.asr_lang_sym or "").strip().lower()
    if sym in _LANG_CODE_TO_ASR:
        return _LANG_CODE_TO_ASR[sym]
    code = (cfg.language or "").strip().lower()
    return _LANG_CODE_TO_ASR.get(code, "")


def _asr_model(cfg: Config, logger: logging.Logger):
    """加载 FunASR/SenseVoiceSmall（含 VAD，全局缓存单例，懒加载）。

    - hub="hf"：模型经 huggingface_hub 从 HuggingFace 下载，落 HF 默认缓存；
    - 本地优先：先尝试仅从 HF 缓存定位快照（零网络请求），未命中才联网下载；
      HF_LOCAL_FIRST=0 可关闭本地优先、强制联网校验更新。
    """
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL

    # --transcribe / 参考音频转写路径不加载 TTS 模型，ASR 走 CPU 时同样配满线程
    _configure_cpu_threads(cfg, logger)

    from funasr import AutoModel

    asr_id = _asr_model_id(cfg)
    hub = _asr_hub(cfg)
    vad = _asr_vad_id(cfg)
    device = cfg.device or "cpu"

    logger.info("⏳ 加载 SenseVoiceSmall（%s, %s%s%s）…", asr_id, device,
                f", VAD={vad}" if vad else ", 无 VAD",
                f", hub={hub}" if hub != "hf" else "")
    t0 = time.time()

    # 本地优先：ASR_MODEL 是本地目录时直接复用（跳过缓存检查与下载）；
    # 否则仅从 HF 缓存定位快照（零网络请求，跳过 revision 检查），未命中/
    # 快照不完整再交给 funasr 联网下载（HF_LOCAL_FIRST=0 可关闭本地优先）
    model_path = None
    local_hit = False
    if os.path.isdir(asr_id):
        model_path = os.path.abspath(asr_id)
        local_hit = True
    elif hub == "hf" and os.environ.get("HF_LOCAL_FIRST", "1") != "0":
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(asr_id, local_files_only=True)
            local_hit = True
        except Exception:
            pass  # 未命中/快照不完整，走 funasr 联网下载

    kwargs = dict(
        model=model_path or asr_id,
        device=device,
        hub=hub,
        disable_update=True,   # 跳过 funasr 的启动版本检查（避免无谓联网）
        disable_pbar=True,     # 关闭 tqdm 进度条（长音频静默处理）
        log_level="WARNING",   # 压低 funasr 内部日志，不刷屏
    )
    if vad:
        kwargs["vad_model"] = vad
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    try:
        _ASR_MODEL = AutoModel(**kwargs)
    except Exception as e:
        # funasr 在模型缺失/下载失败时抛的异常信息不含 ASR 上下文，包一层明确报错
        logger.error("❌ SenseVoiceSmall 加载失败（%s）: %s", asr_id, e)
        logger.error("   提示: 首次运行需联网下载模型（约 900MB）；可检查网络、")
        logger.error("   HF_ENDPOINT 镜像，或设 ASR_MODEL 指向本地模型目录")
        raise
    logger.info("✓ ASR 模型加载: %.1fs (%s%s)", time.time() - t0, device,
                "，本地缓存命中，跳过联网" if local_hit else "")
    return _ASR_MODEL


def _clean_asr_text(text: str) -> str:
    """剥离 SenseVoice 输出的元数据 token（<|zh|> <|NEUTRAL|> <|Speech|> <|woitn|> …），
    只保留纯文本。"""
    return _ASR_TOKEN_RE.sub("", text or "").strip()


def _transcribe_ref(cfg: Config, logger: logging.Logger) -> str:
    """用 FunASR/SenseVoiceSmall 转写参考音频，返回纯文本。

    - 供 --transcribe 子命令与 TTS 语音克隆路径共用；
    - 音频原路径交给模型（内部加载/重采样，无需 ffmpeg）；超长音频由
      fsmn-vad 自动切段，返回整段合并文本；
    - 不加载 OmniVoice TTS 模型。
    """
    if not cfg.ref_audio or not os.path.isfile(cfg.ref_audio):
        logger.error("❌ 请设置有效的参考音频路径（--transcribe <ref_audio>）")
        sys.exit(1)

    asr = _asr_model(cfg, logger)
    lang = _asr_language(cfg)
    logger.info("⏳ 转写中（语言: %s）…", lang or "auto 自动检测")
    t0 = time.time()
    results = asr.generate(
        input=cfg.ref_audio,
        cache={},
        language=lang or "auto",
        use_itn=True,        # 逆文本正则化：数字/单位转中文汉字等
        batch_size_s=60,     # 动态批量：按音频总时长合批
        merge_vad=True,      # 合并 VAD 切段，返回整段文本
        merge_length_s=15,
    )
    text = _clean_asr_text(results[0]["text"] if results else "")
    logger.info("✓ 转写完成: %.1fs（语言: %s）", time.time() - t0, lang or "auto 自动检测")
    return text


def _run_transcribe(cfg: Config, logger: logging.Logger) -> None:
    """ASR 子命令：用 SenseVoiceSmall 转写参考音频并打印文本，然后退出。

    不加载 OmniVoice TTS 模型，独立于主流程运行。
    """
    print(_transcribe_ref(cfg, logger))




# ── CLI 入口 ──────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="OmniVoice 配音：600+ 语言零样本语音克隆 / 声音设计 / 自动音色"
    )
    parser.add_argument(
        "ref_audio", nargs="?", default=None,
        help="参考音频文件（语音克隆模式；省略时用 --instruct 声音设计或自动音色）",
    )
    parser.add_argument(
        "text_file", nargs="?", default=None,
        help="文本文件路径（声音设计/自动音色模式可用 --text 代替）",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="文本文件路径（声音设计/自动音色模式使用，避免与参考音频位置参数歧义）",
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None,
        help="合成语言代码/名称（如 en/zh/English；默认自动判断，可用 LANGUAGE 覆盖）",
    )
    parser.add_argument(
        "--ref-text", type=str, default=None,
        help="参考音频转写文本（省略时默认用 FunASR/SenseVoiceSmall 转写参考音频）",
    )
    parser.add_argument(
        "--instruct", type=str, default=None,
        help="声音设计指令，如 'female, low pitch, british accent'（无需参考音频）",
    )
    parser.add_argument(
        "--draw-count", "-n", type=int, default=None,
        help="生成次数（默认 2，可用 DRAW_COUNT 环境变量覆盖）",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="设备：cuda/mps/cpu（默认自动检测）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（默认文本文件所在目录，可用 OUTPUT_DIR 覆盖）",
    )
    parser.add_argument(
        "--transcribe", action="store_true",
        help="ASR 子命令：用 FunASR/SenseVoiceSmall 转写参考音频并打印文本（不生成 TTS）",
    )
    parser.add_argument(
        "--asr-model", type=str, default=None,
        help="SenseVoice 模型 ID 或本地目录（默认 FunAudioLLM/SenseVoiceSmall，可用 ASR_MODEL 覆盖）",
    )
    parser.add_argument(
        "--lang-sym", type=str, default=None,
        help="ASR 语言代码（如 en/zh/yue/ja/ko）；留空则跟随 --language，再留空自动检测",
    )
    parser.add_argument(
        "--region-sym", type=str, default=None,
        help="（已废弃，SenseVoice 不支持地区强制）仅保留兼容参数",
    )
    return parser.parse_args(argv)


def _pick(cli_val, env_name: str, default, cast=None):
    """按 CLI 参数 > 环境变量 > 默认值的优先级取值（空字符串视为未设置）。"""
    v = cli_val if cli_val is not None else os.environ.get(env_name)
    if v is None or v == "":
        return default
    return cast(v) if cast else v


def _resolve_config(args: argparse.Namespace,
                    defaults: Optional[Config] = None) -> Config:
    """合并 CLI 参数 → 环境变量 → 默认值，返回有效的运行配置。"""
    d = defaults or Config()
    cfg = Config(
        ref_audio=_pick(args.ref_audio, "REF_AUDIO", d.ref_audio),
        text_path=_pick(args.text_file or getattr(args, "text", None),
                        "TEXT_PATH", d.text_path),
        language=_pick(args.language, "LANGUAGE", d.language),
        ref_text=_pick(args.ref_text, "REF_TEXT", d.ref_text),
        instruct=_pick(args.instruct, "INSTRUCT", d.instruct),
        draw_count=_pick(args.draw_count, "DRAW_COUNT", d.draw_count, int),
        output_dir=_pick(args.output_dir, "OUTPUT_DIR", d.output_dir),
        device=_pick(args.device, "DEVICE", d.device),
        dtype=_pick(None, "DTYPE", d.dtype),
        model_path=_pick(None, "MODEL_PATH", d.model_path),
        model_id=_pick(None, "OMNIVOICE_MODEL_ID", d.model_id),
        transcribe=bool(getattr(args, "transcribe", False))
                   or _to_bool(os.environ.get("TRANSCRIBE", "")),
        asr_model=_pick(getattr(args, "asr_model", None), "ASR_MODEL", d.asr_model),
        asr_hub=_pick(None, "ASR_HUB", d.asr_hub),
        asr_vad=_pick(None, "ASR_VAD", d.asr_vad),
        asr_lang_sym=_pick(getattr(args, "lang_sym", None), "ASR_LANG_SYM", d.asr_lang_sym),
        asr_region_sym=_pick(getattr(args, "region_sym", None), "ASR_REGION_SYM", d.asr_region_sym),
    )
    if not cfg.device:
        cfg.device = get_best_device()
    return cfg


def _to_bool(v: str) -> bool:
    """把字符串解析为布尔（1/true/yes/on → True，其余 → False）。"""
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# 生成参数 → model.generate() 入参名：只透传环境变量显式设置的项，
# 其余交给模型自身默认值（与上游 generate()/OmniVoiceGenerationConfig 默认一致）
_GEN_PARAM_ENVS = {
    "NUM_STEP": ("num_step", int),
    "GUIDANCE_SCALE": ("guidance_scale", float),
    "T_SHIFT": ("t_shift", float),
    "DENOISE": ("denoise", _to_bool),
    "POSTPROCESS_OUTPUT": ("postprocess_output", _to_bool),
    "LAYER_PENALTY_FACTOR": ("layer_penalty_factor", float),
    "POSITION_TEMPERATURE": ("position_temperature", float),
    "CLASS_TEMPERATURE": ("class_temperature", float),
    "AUDIO_CHUNK_DURATION": ("audio_chunk_duration", float),
    "AUDIO_CHUNK_THRESHOLD": ("audio_chunk_threshold", float),
    "PAD_DURATION": ("pad_duration", float),
    "FADE_DURATION": ("fade_duration", float),
    "SPEED": ("speed", float),
    "DURATION": ("duration", float),
    "NORMALIZE_TEXT": ("normalize_text", _to_bool),
}


def _gen_kwargs() -> dict:
    """生成参数：从环境变量构建，未设置的交给模型默认值。"""
    return {
        param: cast(os.environ[env])
        for env, (param, cast) in _GEN_PARAM_ENVS.items()
        if env in os.environ
    }


def _validate_inputs(cfg: Config, logger: logging.Logger) -> None:
    """验证输入文件是否存在，不通过则退出进程。"""
    if cfg.draw_count < 1:
        logger.error("❌ DRAW_COUNT/--draw-count 必须 >= 1（当前 %d）", cfg.draw_count)
        sys.exit(1)
    if not cfg.text_path or not os.path.isfile(cfg.text_path):
        logger.error("❌ 请设置有效的文本文件路径")
        sys.exit(1)
    if cfg.ref_audio and not os.path.isfile(cfg.ref_audio):
        logger.error("❌ 参考音频不存在: %s", cfg.ref_audio)
        sys.exit(1)
    if not cfg.ref_audio and not cfg.instruct:
        logger.info("ℹ️ 未提供参考音频与指令，使用自动音色模式")


def main(argv: Optional[list[str]] = None) -> None:
    """
    OmniVoice 配音入口（语音克隆 / 声音设计 / 自动音色）。

    用法:
      uv run python omni.py <ref_audio> <text_file>
      uv run python omni.py <ref_audio> <text_file> --language en
      uv run python omni.py --text <text_file> --instruct "female, low pitch, british accent"
      uv run python omni.py --transcribe <ref_audio>                 # ASR（FunASR/SenseVoiceSmall）
      uv run python omni.py --transcribe <ref_audio> --lang-sym en
      DRAW_COUNT=3 LANGUAGE=yue uv run python omni.py /path/to/ref.wav /path/to/text.txt
    """
    logger = logging.getLogger("omni")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        args = _parse_args(argv)
        cfg = _resolve_config(args)

        # ── ASR 子命令：独立于 TTS 主流程 ──
        if cfg.transcribe:
            _run_transcribe(cfg, logger)
            return

        mode = ("语音克隆" if cfg.ref_audio
                else "声音设计" if cfg.instruct
                else "自动音色")
        logger.info("🌐 模式: %s  语言: %s  设备: %s",
                    mode, cfg.language or "自动", cfg.device)

        _validate_inputs(cfg, logger)

        with open(cfg.text_path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            logger.error("❌ 文本文件为空")
            sys.exit(1)

        # 参考音频直接交给模型处理（模型内部自行加载/重采样/转单声道，
        # 无需 ffmpeg 预转换）；ref_text 省略时默认用 SenseVoiceSmall 转写参考音频
        # （不依赖 OmniVoice 内部 Whisper ASR）
        ref_text = cfg.ref_text
        if cfg.ref_audio and not ref_text:
            logger.info("ℹ️ ref_text 未提供，用 SenseVoiceSmall 转写参考音频 …")
            ref_text = _transcribe_ref(cfg, logger)
            logger.info("📝 参考文本: %s", ref_text)
            if not ref_text:
                logger.error("❌ ASR 转写结果为空（参考音频可能为静音）")
                sys.exit(1)
        model = _load_model(cfg, logger)

        # ── 多轮生成 ──
        out_dir = (os.path.abspath(cfg.output_dir)
                   if cfg.output_dir
                   else os.path.dirname(os.path.abspath(cfg.text_path)))
        os.makedirs(out_dir, exist_ok=True)
        out_base = os.path.join(out_dir, os.path.basename(cfg.text_path))
        gen_kwargs = _gen_kwargs()

        import soundfile as sf  # 与上游 CLI 一致，用 soundfile 保存 WAV
        from tqdm import tqdm  # 进度条（transformers 已带该依赖）

        for draw in range(1, cfg.draw_count + 1):
            # 文件名：<文本名>.<unix 秒时间戳>.wav（不含抽卡序号）。
            # 同秒内多轮时递增秒数，避免覆盖已有文件
            ts = int(time.time())
            out_path = f"{out_base}.{ts}.wav"
            while os.path.exists(out_path):
                ts += 1
                out_path = f"{out_base}.{ts}.wav"
            logger.info("  [%d/%d] 生成中 …", draw, cfg.draw_count)

            # 上游 generate() 无内部进度回调：起一个后台线程实时刷新耗时进度条，
            # generate 返回后关闭（仅显示 elapsed，不伪造步进百分比）
            pbar = tqdm(total=None, bar_format="{desc}", leave=False)
            stop = threading.Event()

            def _tick() -> None:
                t0 = time.time()
                while not stop.is_set():
                    pbar.set_description_str(
                        f"  ⏳ 生成中 [{draw}/{cfg.draw_count}] "
                        f"{time.time() - t0:.1f}s")
                    pbar.refresh()
                    time.sleep(0.1)

            spinner = threading.Thread(target=_tick, daemon=True)
            spinner.start()
            t1 = time.time()
            try:
                audios = model.generate(
                    text=text,
                    language=cfg.language or None,
                    ref_audio=cfg.ref_audio or None,
                    # 克隆模式下 ref_text 已由用户提供或 SenseVoiceSmall 转写（保证非空非 None，
                    # 模型内部不会走 Whisper 兜底）；声音设计/自动音色模式传 None
                    ref_text=ref_text if cfg.ref_audio else None,
                    instruct=cfg.instruct or None,
                    **gen_kwargs,
                )
            finally:
                stop.set()
                spinner.join(timeout=1.0)
                pbar.close()
            sf.write(out_path, audios[0], model.sampling_rate)
            kb = os.path.getsize(out_path) / 1024
            logger.info("  %s  (%.0f KB, %.1f s)",
                        os.path.basename(out_path), kb, time.time() - t1)
    except Exception:
        # cfg 可能尚未赋值（_parse_args/_resolve_config 阶段抛错）——用局部变量兜底
        cfg = locals().get("cfg")
        logger.exception("❌ %s失败",
                         "转写" if getattr(cfg, "transcribe", False) else "生成")
        sys.exit(1)


if __name__ == "__main__":
    main()
