"""注册表：真实模型（导入即注册）与未知 id 行为。"""

from __future__ import annotations

import pytest

import ov.models  # noqa: F401  真实模型注册（导入一次）
from ov.model import ModelSpec, get_model, models, register

import stubs  # noqa: E402   stub 注册


def test_real_models_registered():
    assert get_model("omnivoice").kind == "tts"
    caps = get_model("omnivoice").capabilities
    assert {"clone", "design", "auto", "native_longform"} <= caps
    assert get_model("sensevoice").kind == "asr"
    assert "transcribe" in get_model("sensevoice").capabilities


def test_stub_registered():
    assert get_model("stub-tts").engine.calls == 0
    assert "native_longform" not in get_model("stub-tts").capabilities


def test_duplicate_register_rejected():
    with pytest.raises(ValueError):
        register(ModelSpec(id="stub-tts", kind="tts", description="dup",
                           capabilities=frozenset(),
                           supported_params=frozenset(),
                           engine=stubs.StubTTS()))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model("no-such-model")


def test_models_list_nonempty():
    ids = {m.id for m in models()}
    assert {"omnivoice", "sensevoice", "stub-tts"} <= ids
