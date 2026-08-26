"""
OmniVoice 配音工具 — ASR 参考音频转写（FunASR / SenseVoiceSmall）

目标模型：SenseVoiceSmall（FunAudioLLM/SenseVoiceSmall，MIT）——多语言语音识别，
官方支持 中/英/粤/日/韩 五种语言强制识别 + 自动检测（auto），并输出情感/
事件/标点标签；小模型（234M）CPU 即可实时，参考音频转写场景绰绰有余。
- 权重：funasr AutoModel(hub="hf") 经 huggingface_hub 从 HuggingFace 下载
  （FunAudioLLM/SenseVoiceSmall + funasr/fsmn-vad 官方镜像），落 HF 默认缓存，
  符合本项目 HF 管理规则；本地优先：缓存命中直接复用本地快照，零网络请求。
- 不依赖 transformers 的 ASR 架构（funasr 自带模型实现，仅复用 torch），
  与两种 TTS 后端（transformers / GGUF）均无冲突。
- VAD：默认附带 fsmn-vad 自动切分长音频（单段上限 30 s，段间合并返回整段
  文本），参考音频多长都能处理；ASR_VAD=0 可关闭（省内存，仅适合短音频）。
- 语言：SenseVoice 用代码（zh/en/yue/ja/ko/auto）；把 --language 代码映射为
  SenseVoice 语言代码并强制转写（避免自动检测误判，如英语被判成韩语），
  映射不到的语言交给模型自动检测（auto）。
- 输出：SenseVoice 文本带 <|lang|><|emo|><|event|><|woitn|> 元数据 token，
  返回前统一剥离，只留纯文本。
"""

from __future__ import annotations

import logging
import os
import re
import time

from .config import Config
from .device import _apply_mps_memory_settings, _configure_cpu_threads, _should_fallback_to_cpu


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

    # MPS 设备先解除分配器内存上限（幂等）
    _apply_mps_memory_settings(cfg.device or "cpu", logger)

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
    except RuntimeError as e:
        if not _should_fallback_to_cpu(e, device):
            raise
        # MPS 内存不足 / XPU 运行时失败：SenseVoice 改用 CPU 重试（小模型，CPU 足够实时）
        failed_device = device.upper()
        device = "cpu"
        cfg.device = "cpu"
        kwargs["device"] = "cpu"
        logger.warning("⚠️ %s 加载失败（%s），SenseVoice 改用 CPU 加载 …",
                       failed_device, e)
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
    - 不加载任何 TTS 模型（transformers / GGUF 均不加载）。
    """
    if not cfg.ref_audio or not os.path.isfile(cfg.ref_audio):
        # 抛异常而非 sys.exit：web.py 在进程内调用本函数，exit 会杀死
        # Gradio 服务器；CLI 侧由 cli.py 的 main() 捕获并退出
        raise ValueError(
            "请设置有效的参考音频路径（--transcribe <ref_audio>）")

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
