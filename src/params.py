"""
OmniVoice 配音工具 — 生成参数（环境变量 → 后端入参）

生成参数不设默认值，全部交由后端自身默认值；只有显式设置的环境变量才
透传。参数名（generate() 关键字）与旧 omni.py 完全一致，供 cli.py /
web.py 与各后端共用（后端只取自己支持的子集）。
"""

from __future__ import annotations

import os

from .config import _to_bool


# 生成参数 → generate() 入参名：只透传环境变量显式设置的项，
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
