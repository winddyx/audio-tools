"""
Web 界面语言下拉：显示名 ↔ ISO 代码（内置常用语言表）。

GGUF 后端（omnivoice.cpp）的 --lang 接受 ISO 代码或语言名（C++ lang-map.h
内置 600+ 语言表，未命中则自动识别）；这里只给 UI 常用集合，覆盖日常
中/英/粤/日/韩等即可。
"""

from __future__ import annotations

# 小写语言名 → ISO 639-3 代码
LANG_NAME_TO_ID = {
    "english": "en",
    "chinese": "zh",
    "cantonese": "yue",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
    "vietnamese": "vi",
    "thai": "th",
    "indonesian": "id",
    "hindi": "hi",
}


def lang_display_name(name: str) -> str:
    """小写语言名 → UI 显示名（'english' → 'English'）。"""
    return name.capitalize()
