"""文本工具：断句与按长度兜底分块。"""

from __future__ import annotations

from ov.text import chunk_by_length, split_sentences


def test_split_sentences_keeps_punctuation():
    text = "第一句。第二句！\n第三句？第四句"
    parts = split_sentences(text)
    assert parts == ["第一句。", "第二句！", "第三句？", "第四句"]


def test_chunk_by_length_groups_short_sentences():
    text = "一二三四五六七八九十。" * 3   # 30 字，含标点
    chunks = chunk_by_length(text, max_chars=25)
    # 每块不超过 25 字；块数 = ceil(30/25) = 2
    assert 1 < len(chunks) <= 3
    assert all(len(c) <= 25 for c in chunks)


def test_chunk_by_length_joins_without_loss():
    text = "甲乙丙丁。戊己庚辛。壬癸。"
    chunks = chunk_by_length(text, max_chars=100)
    assert "".join(chunks) == text
