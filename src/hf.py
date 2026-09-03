"""
OmniVoice 配音工具 — HuggingFace 下载与缓存管理

模型管理规则（必须遵守）：
1. 所有模型统一由 HuggingFace（huggingface_hub）管理下载，不硬编码任何本地路径。
2. 模型文件一律落在 HuggingFace 默认缓存（~/.cache/huggingface，或
   HF_HOME / HF_HUB_CACHE 指定的位置）；路径只取 hf_hub_download() /
   snapshot_download() 的返回值，代码里不写死缓存路径。
3. 本地优先：先尝试 local_files_only 定位缓存快照，命中则零网络请求直接
   复用（跳过 revision 检查与文件列表）；未命中/快照不完整才联网下载。
   HF_LOCAL_FIRST=0 可关闭本地优先，强制联网校验更新。
4. 直连 huggingface.co 失败（超时等）且未显式设置 HF_ENDPOINT 时，
   自动改用 hf-mirror.com 镜像重试一次；HF_NO_MIRROR_FALLBACK=1 可关闭。
5. 下载保留 huggingface_hub 默认进度条（不设 HF_HUB_DISABLE_PROGRESS_BARS）。
"""

from __future__ import annotations

import logging
import os


# 直连 HuggingFace 失败时的镜像兜底（国内网络常用）
_HF_MIRROR = "https://hf-mirror.com"


def _switch_hf_endpoint(endpoint: str) -> None:
    """运行时把 huggingface_hub 的目标 endpoint 切换到镜像。

    huggingface_hub 在 import 时就把 endpoint 固化进 URL 模板（constants.ENDPOINT
    与 file_download.HUGGINGFACE_CO_URL_TEMPLATE），仅设置环境变量不生效，
    需同步更新这些常量；版本差异导致的异常忽略，环境变量仍对新进程生效。
    """
    endpoint = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import constants as hf_constants
        from huggingface_hub import file_download as hf_file_download

        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            endpoint + "/{repo_id}/resolve/{revision}/{filename}"
        )
        # 新版 huggingface_hub 已删除 file_download 里的模板属性，用 hasattr 保护
        if hasattr(hf_file_download, "HUGGINGFACE_CO_URL_TEMPLATE"):
            hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE = (
                hf_constants.HUGGINGFACE_CO_URL_TEMPLATE
            )
    except Exception:
        pass


def _hf_download(repo_id: str, filename: str = "") -> str:
    """HuggingFace 下载单个文件/快照（遵循 HF_ENDPOINT 镜像）。

    本地优先：模型已完整缓存在本地时直接复用快照，不发起 revision 检查/
    文件列表等任何网络请求（默认，HF_LOCAL_FIRST=0 可关闭）；未命中或快照
    不完整才联网下载。直连失败且用户未显式设置 HF_ENDPOINT（也未用
    HF_NO_MIRROR_FALLBACK=1 关闭兜底）时，自动切换到 hf-mirror.com 重试一次。
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    def _local_only() -> str:
        if filename:
            return hf_hub_download(repo_id, filename,
                                   local_files_only=True)
        return snapshot_download(repo_id, local_files_only=True)

    def _remote() -> str:
        if filename:
            return hf_hub_download(repo_id, filename)
        return snapshot_download(repo_id)

    # 判断：本地已有完整模型则直接复用，跳过一切联网
    # （含 revision 检查/文件列表；HF_LOCAL_FIRST=0 可关闭本地优先）
    if os.environ.get("HF_LOCAL_FIRST", "1") != "0":
        try:
            path = _local_only()
        except Exception:
            path = ""  # 未命中/快照不完整，走联网下载
        if path:
            logging.getLogger("omni").info(
                "本地已有模型，跳过联网: %s", repo_id)
            return path

    try:
        return _remote()
    except Exception:
        # 用户已显式指定 endpoint / 明确关闭兜底：尊重其选择，直接抛错
        if os.environ.get("HF_ENDPOINT") or os.environ.get("HF_NO_MIRROR_FALLBACK"):
            raise
        logger = logging.getLogger("omni")
        logger.warning("直连 HuggingFace 失败（%s），改用镜像 %s 重试 …",
                       repo_id, _HF_MIRROR)
        _switch_hf_endpoint(_HF_MIRROR)
        try:
            return _remote()
        except Exception as e2:
            logger.error("镜像 %s 下载 %s 也失败: %s", _HF_MIRROR, repo_id, e2)
            raise


def resolve_path(model_id: str = "", local_path: str = "") -> str:
    """解析主模型路径：优先 local_path，否则从 HuggingFace 自动下载。

    snapshot_download 不传 cache_dir，由 huggingface_hub 决定缓存位置
    （默认 ~/.cache/huggingface，或遵循 HF_HOME / HF_HUB_CACHE）。
    """
    if local_path:
        return os.path.abspath(local_path)
    return _hf_download(model_id)
