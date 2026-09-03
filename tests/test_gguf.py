"""src.params：环境变量 → generate() 入参映射；src/backends/gguf 参数/解析助手。"""

import pytest

from src.backends.gguf import (
    _backend_init_failed,
    _gen_kwargs_to_cli,
    _ggml_backend,
    _read_chunks_sidecar,
)
from src.params import _gen_kwargs


def test_gen_kwargs_only_passes_explicit_env(monkeypatch):
    monkeypatch.delenv("NUM_STEP", raising=False)
    monkeypatch.delenv("DENOISE", raising=False)
    monkeypatch.setenv("AUDIO_CHUNK_DURATION", "12.5")
    kw = _gen_kwargs()
    assert kw == {"audio_chunk_duration": 12.5}


def test_gen_kwargs_typecasts(monkeypatch):
    monkeypatch.setenv("NUM_STEP", "16")
    monkeypatch.setenv("DENOISE", "0")
    kw = _gen_kwargs()
    assert kw["num_step"] == 16
    assert kw["denoise"] is False


def test_ggml_backend_mapping():
    assert _ggml_backend("cuda") == "CUDA0"
    assert _ggml_backend("mps") == "MTL0"
    assert _ggml_backend("cpu") == "CPU"
    assert _ggml_backend("") is None
    assert _ggml_backend("xpu") is None  # 交给运行时自动选择


def test_gen_kwargs_to_cli_subset(logger):
    cli = _gen_kwargs_to_cli(
        {"num_step": 24, "denoise": False, "duration": 8.0,
         "guidance_scale": 2.0}, logger)
    assert "--steps" in cli and "24" in cli
    assert "--no-denoise" in cli
    assert "--duration" in cli and "8.0" in cli
    assert "guidance_scale" not in " ".join(cli)  # 不支持的被忽略


def test_read_chunks_sidecar(tmp_path):
    p = tmp_path / "chunks.json"
    p.write_text('["甲。","乙。"]', encoding="utf-8")
    chunks = _read_chunks_sidecar(str(p))
    assert [c.text for c in chunks] == ["甲。", "乙。"]

    p.write_text("not json", encoding="utf-8")
    assert _read_chunks_sidecar(str(p)) == []
    assert _read_chunks_sidecar(str(tmp_path / "missing.json")) == []


def test_backend_init_failed_detection():
    assert _backend_init_failed(b"backend_init failed: no backend available")
    assert not _backend_init_failed(b"unknown option --foo")
    assert not _backend_init_failed(b"")
