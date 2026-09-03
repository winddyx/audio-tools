"""pytest 公共夹具（stub 引擎见 tests/stubs.py）。"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证 repo 根目录可 import（ov 包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
import stubs  # noqa: E402,F401  注册 stub 引擎（幂等）


@pytest.fixture
def stub_tts():
    return stubs._stub_tts


@pytest.fixture
def stub_native():
    return stubs._stub_native
