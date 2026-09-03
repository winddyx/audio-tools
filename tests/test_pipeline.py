"""编排层：命名唯一、落盘、长文本策略（stub 引擎驱动，不加载真实模型）。"""

from __future__ import annotations

import os

import numpy as np

from ov import pipeline
from ov.model import get_model
from ov.types import GenParams, SynthesizeRequest
from ov.audio import read_wav


def test_unique_out_path_never_collides(tmp_path):
    name = "demo"
    p1 = pipeline.unique_out_path(str(tmp_path), name)
    open(p1, "w").close()          # 模拟已有文件
    p2 = pipeline.unique_out_path(str(tmp_path), name)
    assert p1 != p2
    assert p1.endswith(".wav") and p2.endswith(".wav")
    assert p1.startswith(str(tmp_path)) and p2.startswith(str(tmp_path))


def test_complete_short_text_writes_wav(tmp_path, stub_tts):
    spec = get_model("stub-tts")
    before = stub_tts.calls
    outcome = pipeline.complete(
        spec, spec.engine,
        SynthesizeRequest(text="短文本。"),
        out_dir=str(tmp_path), out_name="out", logger=None)
    assert stub_tts.calls == before + 1        # 单次合成
    assert os.path.isfile(outcome.out_path)
    data, sr = read_wav(outcome.out_path)
    assert sr == 8000 and len(data) > 0
    assert outcome.segments and outcome.segments[0].text == "短文本。"
    assert outcome.ref_text == ""


def test_complete_long_text_uses_python_fallback(tmp_path, stub_tts):
    spec = get_model("stub-tts")              # fallback_chunk_chars=20，无 native
    before = stub_tts.calls
    long_text = "第一句很长的内容。" * 12      # 远超 20 字 → 分段
    outcome = pipeline.complete(
        spec, spec.engine,
        SynthesizeRequest(text=long_text),
        out_dir=str(tmp_path), out_name="long", logger=None)
    assert stub_tts.calls > before + 1        # 多次调用引擎
    assert len(outcome.segments) > 1
    assert "".join(s.text for s in outcome.segments) == long_text
    data, sr = read_wav(outcome.out_path)
    assert sr == 8000 and len(data) > 0


def test_complete_native_longform_single_call(tmp_path, stub_native):
    spec = get_model("stub-native")
    before = stub_native.calls
    long_text = "超长文本。超长文本。超长文本。" * 20
    outcome = pipeline.complete(
        spec, spec.engine,
        SynthesizeRequest(text=long_text),
        out_dir=str(tmp_path), out_name="native", logger=None)
    assert stub_native.calls == before + 1    # 引擎原生处理，不兜底分段
    assert len(outcome.segments) == 1


def test_complete_passes_asr_text(tmp_path, stub_tts):
    spec = get_model("stub-tts")
    outcome = pipeline.complete(
        spec, spec.engine,
        SynthesizeRequest(text="hi"),
        out_dir=str(tmp_path), out_name="x", logger=None, asr_text="转录文本")
    assert outcome.ref_text == "转录文本"


def test_params_are_forwarded(tmp_path, stub_tts):
    spec = get_model("stub-tts")
    req = SynthesizeRequest(text="hello", params=GenParams(num_step=5))
    assert req.params.num_step == 5
