"""
OmniVoice 配音工具 — 日志噪音控制

把 HuggingFace 相关第三方库（httpx / huggingface_hub / urllib3 等）的
INFO 日志压到 WARNING，保留 WARNING 及以上提示（如缺 HF_TOKEN）与业务日志。
CLI / web 入口在 logging.basicConfig 之后调用 _quiet_hf_logs()。
"""

from __future__ import annotations

import logging


def _quiet_hf_logs() -> None:
    """把 HuggingFace 相关第三方库的 INFO 日志压到 WARNING。

    符合模型管理规则：HF 相关 INFO 日志（httpx/huggingface_hub/urllib3 等）
    压到 WARNING，保留 WARNING 及以上提示（如缺 HF_TOKEN）与业务日志。
    """
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                 "filelock", "fsspec", "funasr"):
        logging.getLogger(name).setLevel(logging.WARNING)
