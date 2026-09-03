"""src.device：纯逻辑助手（回退判定 / MPS OOM 识别）。get_best_device 依赖
torch 实机探测，不在单测覆盖范围。"""

import pytest

from src.device import _is_mps_oom, _should_fallback_to_cpu


def test_is_mps_oom():
    assert _is_mps_oom(RuntimeError("MPS backend out of memory (other allocations)"))
    assert not _is_mps_oom(RuntimeError("mps something else"))
    assert not _is_mps_oom(ValueError("MPS out of memory"))


@pytest.mark.parametrize("device,exc,expected", [
    ("mps", RuntimeError("MPS backend out of memory"), True),
    ("mps", RuntimeError("unrelated"), False),
    ("xpu", RuntimeError("any runtime error"), True),
    ("xpu", ValueError("not a runtime error"), False),
    ("cuda", RuntimeError("whatever"), False),
    ("cpu", RuntimeError("whatever"), False),
])
def test_should_fallback_to_cpu(device, exc, expected):
    assert _should_fallback_to_cpu(exc, device) is expected
