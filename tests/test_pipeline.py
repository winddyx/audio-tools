"""src.pipeline：统一编排层（synthesize 命名 / ASR / 分块透传），用 stub 后端。"""

import os
import types

import numpy as np
import pytest

from src.backends.protocol import AudioResult, ChunkInfo
from src.config import Config
from src.pipeline import _unique_out_path, synthesize


def _stub_backend(monkeypatch, audio=None, chunks=None):
    """把 pipeline.get_backend 换成 stub 模块，记录 generate 调用参数。"""
    calls = []

    class _Model:
        sampling_rate = 24000

    def _load_model(cfg, logger):
        return _Model()

    def generate(cfg, logger, **kwargs):
        calls.append(kwargs)
        return AudioResult(
            audio=audio if audio is not None else np.zeros(24000, dtype=np.float32),
            sampling_rate=24000,
            chunks=chunks or [],
        )

    mod = types.ModuleType("stub")
    mod._load_model = _load_model
    mod.generate = generate
    monkeypatch.setattr("src.pipeline.get_backend", lambda name: mod)
    return calls


def _cfg(tmp_path) -> Config:
    return Config(output_dir=str(tmp_path))


def test_unique_out_path_collision(tmp_path):
    p1 = _unique_out_path(str(tmp_path), "demo")
    assert p1.endswith(".wav")
    # 已存在同名时间戳文件 → 秒数递增避免覆盖
    open(p1, "w").close()
    p2 = _unique_out_path(str(tmp_path), "demo")
    assert p1 != p2
    assert os.path.exists(p1) and not os.path.exists(p2)


def test_synthesize_writes_wav_and_passes_chunks(tmp_path, logger, monkeypatch):
    calls = _stub_backend(monkeypatch, chunks=[ChunkInfo("甲。"), ChunkInfo("乙。")])
    result = synthesize(
        _cfg(tmp_path), logger, text="甲。\n乙。", language="zh",
        out_dir=str(tmp_path), out_name="demo", gen_kwargs={"num_step": 16},
    )
    assert result.out_path.endswith(".wav")
    assert result.out_path.startswith(str(tmp_path))
    assert result.duration_sec == pytest.approx(1.0)
    assert [c.text for c in result.chunks] == ["甲。", "乙。"]
    # generate 收到的参数
    assert calls[0]["text"] == "甲。\n乙。"
    assert calls[0]["language"] == "zh"
    assert calls[0]["ref_audio"] is None
    assert calls[0]["num_step"] == 16


def test_synthesize_transcribes_when_ref_text_missing(tmp_path, logger, monkeypatch):
    calls = _stub_backend(monkeypatch)
    monkeypatch.setattr("src.pipeline._transcribe_ref",
                        lambda cfg, logger: "转写结果")
    result = synthesize(
        _cfg(tmp_path), logger, text="你好",
        ref_audio="/tmp/ref.wav", ref_text=None,
        out_dir=str(tmp_path), out_name="demo",
    )
    assert result.ref_text == "转写结果"
    assert calls[0]["ref_text"] == "转写结果"
    assert calls[0]["ref_audio"] == "/tmp/ref.wav"


def test_synthesize_uses_provided_ref_text_without_asr(tmp_path, logger, monkeypatch):
    calls = _stub_backend(monkeypatch)
    called = {"n": 0}

    def fake_asr(cfg, logger):
        called["n"] += 1
        return "不应调用"

    monkeypatch.setattr("src.pipeline._transcribe_ref", fake_asr)
    synthesize(
        _cfg(tmp_path), logger, text="你好",
        ref_audio="/tmp/ref.wav", ref_text="用户提供",
        out_dir=str(tmp_path), out_name="demo",
    )
    assert called["n"] == 0
    assert calls[0]["ref_text"] == "用户提供"
