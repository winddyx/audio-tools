"""音频工具：wav 读写 / 简单拼接（编排层与测试用，引擎产物一律 float32 mono）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读 wav → (float32 mono 数组, 采样率)。多声道自动取第一声道。"""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    return np.ascontiguousarray(data), int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sampling_rate: int) -> str:
    """写 float32 数组为 wav（自动转 16-bit PCM），返回绝对路径。"""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sampling_rate)
    return str(path.resolve())


def concat(items: list[np.ndarray], gap_zeros: int = 0) -> np.ndarray:
    """顺序拼接音频列表，段间可插 gap_zeros 个零采样。"""
    parts: list[np.ndarray] = []
    for i, a in enumerate(items):
        if i > 0 and gap_zeros > 0:
            parts.append(np.zeros(gap_zeros, dtype=np.float32))
        parts.append(np.asarray(a, dtype=np.float32))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)
