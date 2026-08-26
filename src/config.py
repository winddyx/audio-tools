"""
OmniVoice 配音工具 — 核心配置（Config 数据类 + 通用解析助手）

Config 为 CLI / web / 各推理后端共用的运行配置，字段与旧 omni.py 完全一致；
可调设置统一在项目根 settings.py（BACKEND / GGUF 权重 / Web 选项等）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """全局默认配置。可通过环境变量或 CLI 参数覆盖。"""

    # ── 模型（统一由 HuggingFace 管理；路径取自下载接口返回值）──
    model_id: str = "k2-fsa/OmniVoice"  # transformers 后端模型 ID（GGUF 后端忽略）
    model_path: str = ""      # 非空时优先于 model_id（本地 snapshot 目录）
    device: str = ""          # 留空则自动检测（CUDA > XPU > MPS > CPU）
    dtype: str = ""           # 仅 transformers 后端生效；GGUF 量化随文件而定（默认 BF16）

    # ── 生成模式 ──
    language: str = ""        # 语言代码/名称（如 en / zh / English）；留空 = 自动判断
    ref_audio: str = ""
    ref_text: str = ""        # 参考音频转写文本；留空则用 SenseVoiceSmall-GGUF 转写
    instruct: str = ""        # 声音设计指令（如 "female, low pitch, british accent"）

    # ── 长文本配音 ──
    text_path: str = ""
    draw_count: int = 2       # 抽卡次数
    output_dir: str = ""      # 留空则输出到文本文件所在目录

    # ── ASR 子命令（可选，SenseVoiceSmall-GGUF；用于校对/数据集/验证）──
    transcribe: bool = False  # --transcribe：转写 ref_audio 并打印文本
    asr_model: str = ""       # 本地 SenseVoice GGUF 文件路径（默认经 HF 下载 q8，见 settings）
    asr_hub: str = ""         # 保留字段（GGUF 经 huggingface_hub 下载，无 hub 切换）
    asr_vad: str = ""         # 保留字段（VAD 由 fsmn-vad.gguf 承担，见 settings）
    asr_lang_sym: str = ""    # 保留字段（SenseVoice 自动检测语言）
    asr_region_sym: str = ""  # 保留字段（SenseVoice 不支持地区强制，已废弃）

    # 生成参数不在此配置：全部交由后端自身默认值，如需覆盖用环境变量
    # （见 src/params.py 的 _GEN_PARAM_ENVS）。


def _to_bool(v: str) -> bool:
    """把字符串解析为布尔（1/true/yes/on → True，其余 → False）。"""
    return str(v).strip().lower() in ("1", "true", "yes", "on")
