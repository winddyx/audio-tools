"""pytest 根配置：确保项目根可导入（src 为普通包，非 src-layout）。"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test-omni")
