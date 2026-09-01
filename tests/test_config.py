"""src.config：环境变量覆盖解析（_env / _env_int / _env_bool）。"""

import pytest

from src.config import _env, _env_bool, _env_int


@pytest.mark.parametrize("name,default,set_to,expected", [
    ("TEST_ENV_A", "dflt", "custom", "custom"),
    ("TEST_ENV_A", "dflt", "", "dflt"),      # 空串视为未设置
    ("TEST_ENV_A", "dflt", None, "dflt"),    # 未设置
])
def test_env(name, default, set_to, expected, monkeypatch):
    if set_to is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, set_to)
    assert _env(name, default) == expected


@pytest.mark.parametrize("set_to,expected", [
    ("42", 42), ("", 7), ("abc", 7), (None, 7),
])
def test_env_int(set_to, expected, monkeypatch):
    if set_to is None:
        monkeypatch.delenv("TEST_ENV_B", raising=False)
    else:
        monkeypatch.setenv("TEST_ENV_B", set_to)
    assert _env_int("TEST_ENV_B", 7) == expected


@pytest.mark.parametrize("set_to,expected", [
    ("1", True), ("true", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("", False), (None, False),
])
def test_env_bool(set_to, expected, monkeypatch):
    if set_to is None:
        monkeypatch.delenv("TEST_ENV_C", raising=False)
    else:
        monkeypatch.setenv("TEST_ENV_C", set_to)
    assert _env_bool("TEST_ENV_C", False) == expected


def test_to_bool():
    from src.config import _to_bool
    assert _to_bool("1") and _to_bool("True") and _to_bool("YES")
    assert not _to_bool("0") and not _to_bool("no") and not _to_bool("x")
