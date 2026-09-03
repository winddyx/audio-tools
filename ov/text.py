"""文本工具：断句兜底分块（用于无原生长文本能力的引擎）。

有原生分块（native_longform）的模型不走这里；本模块仅作为模板给
未来模型复用：按句末标点切句，按最大长度聚块。
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;：:])\s*|\n+")


def split_sentences(text: str) -> list[str]:
    """按句末标点切句（保留标点本身），空串剔除。"""
    parts = [p for p in _SENTENCE_END.split(text) if p.strip()]
    return parts


def chunk_by_length(text: str, max_chars: int = 200) -> list[str]:
    """把文本切成 <= max_chars 的块：优先按句，长句内部硬切。"""
    chunks: list[str] = []
    buf = ""
    for sent in split_sentences(text):
        if len(buf) + len(sent) <= max_chars:
            buf += sent
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        # 单句超长：按字符硬切
        while len(sent) > max_chars:
            chunks.append(sent[:max_chars])
            sent = sent[max_chars:]
        buf = sent
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]
