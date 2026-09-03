"""HuggingFace 资产定位与下载：本地优先 + hf-mirror 兜底。

规则（所有模型的 GGUF 一律经此模块获取路径）：
1. 模型文件落在 HuggingFace 默认缓存（HF_HOME / HF_HUB_CACHE 指定则遵循）；
   代码只取 hf_hub_download() / snapshot_download() 返回值，不写死缓存路径。
2. 本地优先：命中完整缓存快照则零网络复用；HF_LOCAL_FIRST=0 可关闭。
3. 直连失败且未显式设置 HF_ENDPOINT / HF_NO_MIRROR_FALLBACK 时，
   自动切 hf-mirror.com 重试一次。
"""

from __future__ import annotations

import logging
import os

_HF_MIRROR = "https://hf-mirror.com"


def _switch_hf_endpoint(endpoint: str) -> None:
    """运行时切换 huggingface_hub endpoint（import 时已固化的常量一并更新）。"""
    endpoint = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import constants as hf_constants
        from huggingface_hub import file_download as hf_file_download

        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            endpoint + "/{repo_id}/resolve/{revision}/{filename}"
        )
        if hasattr(hf_file_download, "HUGGINGFACE_CO_URL_TEMPLATE"):
            hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE = (
                hf_constants.HUGGINGFACE_CO_URL_TEMPLATE
            )
    except Exception:
        pass  # 版本差异导致异常时忽略，环境变量对新进程仍生效


def resolve(repo_id: str, filename: str = "") -> str:
    """下载/定位 repo 内单个文件（filename 非空）或整个快照，返回本地路径。

    调用方进程内自行缓存返回值；本地命中零网络请求。
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    def _local() -> str:
        if filename:
            return hf_hub_download(repo_id, filename, local_files_only=True)
        return snapshot_download(repo_id, local_files_only=True)

    def _remote() -> str:
        if filename:
            return hf_hub_download(repo_id, filename)
        return snapshot_download(repo_id)

    if os.environ.get("HF_LOCAL_FIRST", "1") != "0":
        try:
            path = _local()
        except Exception:
            path = ""
        if path:
            logging.getLogger("ov").info("本地已有模型，跳过联网: %s", repo_id)
            return path

    try:
        return _remote()
    except Exception:
        if os.environ.get("HF_ENDPOINT") or os.environ.get("HF_NO_MIRROR_FALLBACK"):
            raise
        logger = logging.getLogger("ov")
        logger.warning("直连 HuggingFace 失败（%s），改用镜像 %s 重试 …",
                       repo_id, _HF_MIRROR)
        _switch_hf_endpoint(_HF_MIRROR)
        try:
            return _remote()
        except Exception as e2:
            logger.error("镜像 %s 下载 %s 也失败: %s", _HF_MIRROR, repo_id, e2)
            raise
