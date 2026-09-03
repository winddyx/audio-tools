"""日志：纯文本 + 第三方库降噪。

业务日志统一 logger "ov"。不在日志中使用 emoji / 特殊符号。
"""

from __future__ import annotations

import logging

_LOUD_LIBS = ("httpx", "httpcore", "huggingface_hub", "urllib3",
              "filelock", "fsspec")


def setup(level: int = logging.INFO) -> logging.Logger:
    """配置根业务日志（格式：纯文本消息）并返回 ov logger。"""
    logger = logging.getLogger("ov")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    # 第三方库 INFO 压到 WARNING，保留 WARNING 以上提示（如缺 HF_TOKEN）
    for name in _LOUD_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger
