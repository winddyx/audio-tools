"""
OmniVoice 配音工具 — 引擎总出口（生成参数 / 语言表 / 兼容导出面）

本文件聚合三部分，供 cli.py / web.py 与各后端共用：
- 生成参数环境变量映射（_GEN_PARAM_ENVS / _gen_kwargs，原 src/params.py）
- Web 语言下拉表（LANG_NAME_TO_ID / lang_display_name，原 src/lang_map.py）
- 兼容导出（原根 omni.py shim：Config / 设备 / HF / 日志 / ASR / GGUF 后端
  `generate` / `_load_model` 等旧 `from omni import ...` 调用方的聚合点）

生成参数不设默认值，全部交由后端自身默认值；只有显式设置的环境变量才
透传。参数名（generate() 关键字）与旧 omni.py 完全一致，cli.py / web.py
与各后端共用（后端只取自己支持的子集）。
"""

from __future__ import annotations

import os

from .config import _to_bool


# ── 生成参数 → generate() 入参名：只透传环境变量显式设置的项 ──
# 其余交给后端自身默认值（与上游默认配置一致）
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
    """生成参数：从环境变量构建，未设置的交给后端默认值。"""
    return {
        param: cast(os.environ[env])
        for env, (param, cast) in _GEN_PARAM_ENVS.items()
        if env in os.environ
    }


# ── Web 界面语言下拉：显示名 ↔ ISO 代码（内置常用语言表）──
# GGUF 后端（omnivoice.cpp）的 --lang 接受 ISO 代码或语言名（C++ lang-map.h
# 内置 600+ 语言表，未命中则自动识别）；这里只给 UI 常用集合，覆盖日常
# 中/英/粤/日/韩等即可。
LANG_NAME_TO_ID = {
    "english": "en",
    "chinese": "zh",
    "cantonese": "yue",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
    "vietnamese": "vi",
    "thai": "th",
    "indonesian": "id",
    "hindi": "hi",
}


def lang_display_name(name: str) -> str:
    """小写语言名 → UI 显示名（'english' → 'English'）。"""
    return name.capitalize()