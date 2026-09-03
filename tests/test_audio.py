"""音频工具：wav 读写回环与拼接。"""

from __future__ import annotations

import numpy as np

from ov.audio import concat, read_wav, write_wav


def test_write_read_roundtrip(tmp_path):
    sr = 24000
    audio = (np.random.default_rng(0).random(sr // 10) * 2 - 1).astype(
        np.float32)
    path = write_wav(tmp_path / "a.wav", audio, sr)
    data, got_sr = read_wav(path)
    assert got_sr == sr
    assert data.shape == audio.shape
    assert np.allclose(data, audio, atol=1e-3)   # 16-bit PCM 往返误差


def test_write_wav_creates_parent(tmp_path):
    path = write_wav(tmp_path / "sub" / "b.wav",
                     np.zeros(100, dtype=np.float32), 8000)
    assert path.endswith(".wav")


def test_concat_gap_zeros():
    a = np.ones(3, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    out = concat([a, b], gap_zeros=2)
    assert out.shape == (9,)
    assert out[3] == 0.0 and out[4] == 0.0


def test_concat_empty():
    assert concat([]).shape == (0,)
