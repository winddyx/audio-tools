"""
audio-tools — 核心配置（唯一设置源）

包含：Config 数据类 + 全局可调设置（推理引擎 audiocpp / TTS 与 ASR 模型 /
Web 选项）+ 设备检测。原 config.py 的 torch 依赖逻辑（transformers 后端、
MPS 内存设置、PyTorch 线程池）已随 torch 后端移除而删除——推理全部在
audio.cpp C++ 子进程完成，Python 侧不再 import torch。

规则：
- 直接改本文件里的顶部变量即可生效；
- 同名环境变量在运行时覆盖文件默认值（如 `TTS_MODEL=indextts2`）；
- vc.py / web.py 与核心模块都从这里取值，不再各自散落默认值。

目录规划（业务入口在根，核心在 src/）：
- 根目录：vc.py（CLI 入口）、web.py（Gradio 入口）
- src/：config.py（本文件，全局设置）、audiocpp.py（推理引擎运行器）、
  各 TTS 模型核心（omnivoice.py / indextts2.py / fireredtts3.py /
  cosyvoice3.py / moss_tts_local.py / qwen3_tts.py / fish_audio.py）、
  sensevoice.py（ASR 核心）、hf.py（HuggingFace 下载）、pipeline.py（统一编排）
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    """环境变量取值：未设置或为空时用文件默认值。"""
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    """环境变量整数取值：未设置、为空或解析失败时用文件默认值。"""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _to_bool(v: str) -> bool:
    """把字符串解析为布尔（1/true/yes/on → True，其余 → False）。"""
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool) -> bool:
    """环境变量布尔取值：未设置或为空时用文件默认值。"""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return _to_bool(v)


@dataclass
class Config:
    """全局默认配置。可通过环境变量覆盖。"""

    # ── 模型（GGUF_LOCAL 手工放置优先；缺失自动经 HF 下载，文件留在 HF 默认缓存，
    #    不落工程目录；引擎需要真实 .gguf 路径，见 hf._ensure_gguf_file）──
    tts_model: str = ""       # "omnivoice" | "indextts2" | "fireredtts3" | "cosyvoice3" | "moss_tts_local" | "qwen3_tts" | "fish_audio"；留空用 TTS_MODEL
    device: str = ""          # 留空则自动检测（cuda > xpu > mps > cpu）

    # ── 生成模式（本期只做语音克隆）──
    language: str = ""        # 语言代码（如 en / zh / yue）；留空 = 自动判断
    ref_audio: str = ""
    ref_text: str = ""        # 参考音频转写文本；留空则用 SenseVoice 转写

    # ── 长文本配音 ──
    text_path: str = ""
    draw_count: int = 2       # 抽卡次数
    output_dir: str = ""      # 留空则输出到文本文件所在目录

    # ── ASR 子命令（可选，SenseVoice；用于校对/数据集/验证）──
    transcribe: bool = False  # --transcribe：转写 ref_audio 并打印文本
    asr_model: str = ""       # 本地 SenseVoice GGUF 文件路径（默认用 ASR_GGUF_*）


# ── 项目内固定路径（模型/引擎可换，目录本身不可调）────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = os.path.join(_PROJECT_ROOT, "vendor")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")   # 本地模型目录（gitignore）
TMP_DIR = os.path.join(_PROJECT_ROOT, ".tmp")        # 运行期临时目录


# ── 推理引擎：audio.cpp（audiocpp_cli，ggml 框架）──────────
# 引擎源仓库固定（自动 clone 时使用；已有源码目录可用 AUDIOCPP_SRC 指向）
AUDIOCPP_REPO = "https://github.com/0xShug0/audio.cpp.git"
# clone/构建所用的引擎分支或提交（AUDIOCPP_REF）。注意：cosyvoice3 族目前
# 只在 dev 分支实现（main 尚未合并），故默认 dev；等 main 合并后可改回 main。
AUDIOCPP_REF = _env("AUDIOCPP_REF", "dev")
AUDIOCPP_BIN = _env("AUDIOCPP_BIN", "")     # 已编译二进制绝对路径（留空自动定位/构建）
AUDIOCPP_SRC = _env("AUDIOCPP_SRC", "")     # 已有源码目录（默认 vendor/audiocpp）
# 追加 cmake 参数（如 "-DGGML_CUDA=ON"）；构建默认参数见 src/audiocpp.py
AUDIOCPP_BUILD_ARGS = _env("AUDIOCPP_BUILD_ARGS", "")
# True = 透传子进程原始输出（构建/推理全部直通终端，调试用）
AUDIOCPP_DEBUG = _env_bool("AUDIOCPP_DEBUG", False)


# ── TTS 模型（audiocpp 族）────────────────────────────────
# TTS_MODEL 切换模型（弱化单一模型绑定）：omnivoice / indextts2 / fireredtts3 /
# cosyvoice3 / moss_tts_local / qwen3_tts / fish_audio（简写亦可，如 fish）。
# 各模型的 GGUF 文件与 HF 兜底仓库定义在对应模型核心
# （src/omnivoice.py、src/indextts2.py、src/fireredtts3.py、src/cosyvoice3.py、
# src/moss_tts_local.py、src/qwen3_tts.py、src/fish_audio.py），
# 本文件只放默认选择与本地目录。
TTS_MODEL = _env("TTS_MODEL", "omnivoice")

# ── 生成参数（各模型核心在拼 CLI 时消费；默认 = 官方基准，env 可覆盖）──
# 默认值即各模型的官方基准（不改引擎质量/速度取舍，只是显式落在 config）：
# - OmniVoice：去噪步数 32 / CFG 引导 2.0 + 随机种子
# - IndexTTS-2.5：gpt 层 top-k 30 / top-p 0.8 / temperature 0.8 + 随机种子
# - FireRedTTS-3 Base（零样本克隆）：flow 4 步 / CFG 2.0 / 停止阈值 0.5 +
#   随机种子（不传种子时引擎固定 1234，可复现）
# - CosyVoice-3（零样本克隆）：AR top-k 25 / flow 10 步 + 随机种子（不传时
#   引擎固定 1986，可复现）
# - MOSS-TTS-Local v1.5（零样本克隆，模型自动多语言）：音频 token 采样
#   temperature 1.7 / top-p 0.8 / top-k 25 / repetition penalty 1.0（文本
#   门控与分块走引擎默认；该族未暴露 seed，采样随机不可复现）
# - Qwen3-TTS 12Hz 1.7B Base（零样本克隆）：主 talker 采样 temperature 0.9 /
#   top-k 50 / top-p 1.0 / repetition penalty 1.05 + 随机种子（不传时引擎
#   随机，同值可复现）
# - Fish Audio S2-Pro（零样本克隆，引擎自动处理语言）：采样 temperature 0.8 /
#   top-k 30 / top-p 0.8 / 单 chunk 上限 max_new_tokens 1024 + 随机种子
#   （不传时引擎随机，同值可复现）
# 设 0 / 空 / -1 可回到"不传 flag = 引擎默认"。
OMNI_INFERENCE_STEPS = _env_int("OMNI_INFERENCE_STEPS", 32)  # 0 = 引擎默认
OMNI_GUIDANCE_SCALE = _env("OMNI_GUIDANCE_SCALE", "2.0")     # 空 = 引擎默认
INDEXTTS_TOP_K = _env_int("INDEXTTS_TOP_K", 30)              # 0 = 引擎默认
INDEXTTS_TOP_P = _env("INDEXTTS_TOP_P", "0.8")               # 空 = 引擎默认
INDEXTTS_TEMPERATURE = _env("INDEXTTS_TEMPERATURE", "0.8")   # 空 = 引擎默认
FIREREDTTS3_INFERENCE_STEPS = _env_int("FIREREDTTS3_INFERENCE_STEPS", 4)  # 0 = 引擎默认
FIREREDTTS3_GUIDANCE_SCALE = _env("FIREREDTTS3_GUIDANCE_SCALE", "2.0")    # 空 = 引擎默认
FIREREDTTS3_STOP_THRESHOLD = _env("FIREREDTTS3_STOP_THRESHOLD", "0.5")    # 空 = 引擎默认
COSYVOICE3_TOP_K = _env_int("COSYVOICE3_TOP_K", 25)          # 0 = 引擎默认
COSYVOICE3_INFERENCE_STEPS = _env_int("COSYVOICE3_INFERENCE_STEPS", 10)   # 0 = 引擎默认
MOSS_TEMPERATURE = _env("MOSS_TEMPERATURE", "1.7")            # 空 = 引擎默认
MOSS_TOP_P = _env("MOSS_TOP_P", "0.8")                        # 空 = 引擎默认
MOSS_TOP_K = _env_int("MOSS_TOP_K", 25)                       # 0 = 引擎默认
MOSS_REPETITION_PENALTY = _env("MOSS_REPETITION_PENALTY", "1.0")  # 空 = 引擎默认
QWEN3TTS_TEMPERATURE = _env("QWEN3TTS_TEMPERATURE", "0.9")      # 空 = 引擎默认
QWEN3TTS_TOP_P = _env("QWEN3TTS_TOP_P", "1.0")                  # 空 = 引擎默认
QWEN3TTS_TOP_K = _env_int("QWEN3TTS_TOP_K", 50)                 # 0 = 引擎默认
QWEN3TTS_REPETITION_PENALTY = _env("QWEN3TTS_REPETITION_PENALTY", "1.05")  # 空 = 引擎默认
FISH_AUDIO_TEMPERATURE = _env("FISH_AUDIO_TEMPERATURE", "0.8")   # 空 = 引擎默认
FISH_AUDIO_TOP_P = _env("FISH_AUDIO_TOP_P", "0.8")               # 空 = 引擎默认
FISH_AUDIO_TOP_K = _env_int("FISH_AUDIO_TOP_K", 30)              # 0 = 引擎默认
FISH_AUDIO_MAX_NEW_TOKENS = _env_int("FISH_AUDIO_MAX_NEW_TOKENS", 1024)  # 0 = 引擎默认
GEN_SEED = _env_int("GEN_SEED", -1)                          # -1 = 随机（不传 seed）

# ── 长文本分块（引擎 --text-chunk-size / --text-chunk-mode）─────────
# 超长文本一次性合成会产生吞字/x/啊等乱码：实测 OmniVoice 1027 字整段合成
# 字符相似度仅 0.877，160 字分块后 0.982。vc 与 web 共用 src 分块逻辑
# （src/audiocpp._chunk_flags），无需各自处理。
TEXT_CHUNK_SIZE = _env_int("TEXT_CHUNK_SIZE", 160)  # 每块上限（字）；0 = 不分块
# 分块模式：空 = 自动——输入按换行分段且每段 ≤ 上限时用 endline（以换行为
# 界、段落完整），否则用 default（按标点/CJK 断句）。可显式设
# endline / tag_aware / japanese / default。
TEXT_CHUNK_MODE = _env("TEXT_CHUNK_MODE", "")

# ── ASR（参考音频转写）：SenseVoice-Small（audiocpp sense_asr 族）──
# 权重经 HF 下载（本地优先 + 镜像兜底）；注意该 GGUF 是 audiocpp 专用包
# （FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp），与旧 llama-funasr 包不同。
ASR_GGUF_REPO = _env("ASR_GGUF_REPO", "FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp")
ASR_GGUF_BASE = _env("ASR_GGUF_BASE", "sensevoice-small-q8-audiocpp-v1.gguf")


# ── Web 界面 ──────────────────────────────────────────────
WEB_IP = _env("AUDIOTOOLS_WEB_IP", "0.0.0.0")
WEB_PORT = _env_int("AUDIOTOOLS_WEB_PORT", 38001)
# True = 启动后自动用默认浏览器打开界面
WEB_AUTO_OPEN_BROWSER = _env_bool("AUDIOTOOLS_WEB_OPEN_BROWSER", False)


# ── LLM 文案翻译：Hy-MT2-1.8B（腾讯开源翻译模型，llama.cpp 子进程）──
# 用途：普通话/中文文案 → 粤语口播文案（作为 TTS 输入前的文案处理）。
# Hy-MT2 官方支持粤语（yue）互译；本模块用本机 llama-cli 子进程推理，
# Python 只做编排。模型经 HF 下载只留默认缓存（见 src/hf），工程目录
# 不落盘；HYMT2_LOCAL 可手工放置本地 .gguf（本地优先）。
HYMT2_REPO = _env("HYMT2_REPO", "tencent/Hy-MT2-1.8B-GGUF")
HYMT2_FILE = _env("HYMT2_FILE", "Hy-MT2-1.8B-Q8_0.gguf")  # 仓库另备 Q4_K_M / Q6_K
HYMT2_LOCAL = _env("HYMT2_LOCAL", "")     # 手工放置的 .gguf 绝对路径（优先）
# llama.cpp 单次补全可执行名或绝对路径。新版 llama.cpp（ggml >= 0.10）把
# 单次补全拆到 llama-completion（llama-cli 只做对话，不支持 --no-conversation）；
# 旧版可用 llama-cli / llama-simple。brew install llama.cpp 自带。
LLAMA_CLI = _env("LLAMA_CLI", "llama-completion")
# 采样参数默认 = 腾讯官方推荐（temp 0.7 / top-p 0.6 / top-k 20 / rep 1.05）
HYMT2_TEMPERATURE = _env("HYMT2_TEMPERATURE", "0.7")
HYMT2_TOP_P = _env("HYMT2_TOP_P", "0.6")
HYMT2_TOP_K = _env_int("HYMT2_TOP_K", 20)              # 0 = 关闭 top-k
HYMT2_REPETITION_PENALTY = _env("HYMT2_REPETITION_PENALTY", "1.05")
HYMT2_MAX_TOKENS = _env_int("HYMT2_MAX_TOKENS", 4096)  # 每段输出 token 上限
HYMT2_CHUNK_CHARS = _env_int("HYMT2_CHUNK_CHARS", 1500)  # 长文按段分块上限（字）；0 = 不分块
HYMT2_TIMEOUT = _env_int("HYMT2_TIMEOUT", 600)         # 单次推理超时（秒）
# 设备：留空 = 交给 llama-completion 自行选择（当前 brew 版本在 M4 上 Metal
# 张量 API 未启用，实际走 CPU，速度已够）；cpu = 强制不卸载 GPU（--device none）。
HYMT2_DEVICE = _env("HYMT2_DEVICE", "")
# True = 译文统一转成香港繁体（zhconv zh-hk）。audiocpp 粤语前端对简体字
# 容易按普通话读音处理（如"消费者"读错），转繁体后读音更稳，同时消除
# 模型输出里常见的简繁混排。
HYMT2_TRAD = _env_bool("HYMT2_TRAD", True)
# 粤语翻译默认提示词模板（web「粤语翻译」页右侧提示词框的初始内容，
# 可编辑；运行时把 {text} 替换为输入源文案；不含 {text} 时直接拼接在末尾）。
# 采用 Hy-MT2 官方"参考翻译"指令格式 + 防照抄约束：示例只示范粤语用字与
# 句式（否则 1.8B 会在输入与示例相近时直接照搬示例答案），并要求逐句
# 翻译、禁止回显原文。对抽象品牌文案与室内口播文案均已实测可用。
HYMT2_PROMPT = (
    "请将下面的普通话文案翻译成地道的香港粤语口播稿。要求：1) 逐句翻译，"
    "用粤语口语书面字（我哋、你哋、佢、係、唔係、冇、喺、喺度、嘅、咗、同、"
    "同埋、啲、呢啲、嗰啲、乜嘢、點、幾、好、仲、而家、睇、畀）；"
    "2) 唔好照抄原文，亦都唔好照搬下面嘅示例——示例只係示範用字同句式，"
    "唔係答案，就算原文同示例好似，都要按原文意思自己重新寫；"
    "3) 品牌名、人名、地名、英文保留不译；4) 只输出译文，不加解释；"
    "5) 全文统一用香港繁体字输出，唔好用简体字——简体字会令粤语配音读错音"
    "（例如消费者要写消費者、体验要写體驗、这要写這、个要写個、里要写裏/裡）。\n\n"
    "参考下面的翻译（示範粵語點寫同用繁體，唔好照搬）：\n"
    "未来的消费者需要什么 翻译成 未來嘅消費者需要啲乜嘢\n"
    "品牌还能给消费者怎样的体验 翻译成 品牌仲可以畀到消費者啲乜嘢體驗\n"
    "把新的品牌策略和产品变化变成能被看见、被体验的空间场景 翻译成 "
    "將新嘅品牌策略同產品變化，變成睇得到、體驗得到嘅空間場景\n"
    "我们一直在思考这个空间要怎么设计 翻译成 我哋一路喺度諗呢個空間要點樣設計\n"
    "业主希望空间安静舒服 翻译成 戶主希望空間靜靜哋、舒舒服服\n\n"
    "将以下文本翻译为 粤语，注意只需要输出翻译后的结果，不要额外解释：\n\n"
    "{text}"
)


# ── 设备检测（无 torch：纯平台探测 + 引擎能力）─────────────
def get_best_device() -> str:
    """自动检测最佳可用设备（cuda > xpu > mps > cpu）。

    audio.cpp 是独立 C++ 引擎，Python 侧不做 torch 探测；这里按平台给
    默认后端，用户可在 DEVICE / 各入口顶部变量显式覆盖（cuda / cpu）。
    - darwin + Apple Silicon → mps（audiocpp 映射 --backend metal）
    - 其余平台 → cpu（NVIDIA GPU 用户请显式设 DEVICE=cuda）
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    if os.environ.get("DEVICE") in ("cuda", "xpu"):
        return os.environ["DEVICE"]
    return "cpu"


def _should_fallback_to_cpu(e: Exception, device: str) -> bool:
    """设备（GPU 后端）初始化失败是否应自动回退 CPU。"""
    if not device:
        return False
    if device in ("cuda", "mps", "xpu"):
        return isinstance(e, RuntimeError)
    return False


# ── 日志噪音控制 ──────────────────────────────────────────
# 把 HuggingFace 相关第三方库（httpx / huggingface_hub / urllib3 等）的
# INFO 日志压到 WARNING，保留 WARNING 及以上提示（如缺 HF_TOKEN）与业务日志。
# CLI / web 入口在 logging.basicConfig 之后调用 _quiet_hf_logs()。


def _quiet_hf_logs() -> None:
    """把 HuggingFace 相关第三方库的 INFO 日志压到 WARNING。"""
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                 "filelock", "fsspec"):
        logging.getLogger(name).setLevel(logging.WARNING)
