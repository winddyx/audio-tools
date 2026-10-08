"""
audio-tools — Web 守护（macOS launchd 用户级单元）

`web.py service <install|uninstall|start|stop|restart|status|logs>` 的实现：
按 src/config.py 顶部的 SERVICE_* 常量生成 launchd 单元（plist）写到
~/Library/LaunchAgents/，并用 launchctl 加载/卸载/重启/查状态。Python 侧
不常驻任何东西，这里只是把 plist 与 launchctl 的细节收在一处。

动作语义（SERVICE_KEEPALIVE 常开时，"停"只能靠卸载单元——kill 掉的进程
会被 launchd 立刻拉起）：
- install   写单元文件 + bootstrap；装完按 RunAtLoad/KeepAlive 即在运行
- start     未加载则 bootstrap，已加载则 kickstart -k（立即重启进程）
- stop      bootout（卸载单元，plist 与日志保留；再 start 即可加载回来）
- restart   已加载则 kickstart -k，未加载则等同 start
- status    launchctl print 摘要（状态/pid/上次退出码）；未运行返回 1
- uninstall bootout + 删除 plist（日志文件保留）

守护进程用 SERVICE_PYTHON（留空自动取工程 .venv/bin/python）跑
`<工程根>/web.py`，工作目录为工程根；监听地址/端口仍是 WEB_IP / WEB_PORT
（写进 config.py 后重启守护生效：service restart）。
"""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
import sys
import time
from typing import Any, Callable, Optional

from .config import (
    _PROJECT_ROOT,
    SERVICE_KEEPALIVE,
    SERVICE_LABEL,
    SERVICE_LOG_DIR,
    SERVICE_LOG_LINES,
    SERVICE_PATH,
    SERVICE_PLIST_DIR,
    SERVICE_PYTHON,
    SERVICE_RUN_AT_LOAD,
    WEB_PORT,
)

# 支持的动作（web.py 的 service 子命令 choices 用同一份列表）
SERVICE_ACTIONS = ("install", "uninstall", "start", "stop", "restart",
                   "status", "logs")

_STDOUT_NAME = "web.stdout.log"
_STDERR_NAME = "web.stderr.log"


# ── 路径与单元内容 ────────────────────────────────────────


def _require_macos() -> None:
    """service 子命令依赖 launchd，非 macOS 直接报错。"""
    if sys.platform != "darwin":
        raise RuntimeError("service 子命令仅支持 macOS（launchd 用户级单元）")


def _domain() -> str:
    """当前用户的 launchd 域（gui/<uid>，服务在图形登录会话内运行）。"""
    return f"gui/{os.getuid()}"


def _target() -> str:
    return f"{_domain()}/{SERVICE_LABEL}"


def _plist_path() -> str:
    return os.path.join(SERVICE_PLIST_DIR, f"{SERVICE_LABEL}.plist")


def _stdout_path() -> str:
    return os.path.join(SERVICE_LOG_DIR, _STDOUT_NAME)


def _stderr_path() -> str:
    return os.path.join(SERVICE_LOG_DIR, _STDERR_NAME)


def _python_bin() -> str:
    """守护进程所用解释器：SERVICE_PYTHON 优先，其次工程 .venv，最后当前。"""
    if SERVICE_PYTHON:
        return SERVICE_PYTHON
    venv_py = os.path.join(_PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    return os.path.abspath(sys.executable)


def _plist_data() -> dict:
    """launchd 单元内容（键顺序即写入顺序，便于人读）。"""
    return {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [_python_bin(), os.path.join(_PROJECT_ROOT, "web.py")],
        "WorkingDirectory": _PROJECT_ROOT,
        "RunAtLoad": bool(SERVICE_RUN_AT_LOAD),
        "KeepAlive": bool(SERVICE_KEEPALIVE),
        "StandardOutPath": _stdout_path(),
        "StandardErrorPath": _stderr_path(),
        "EnvironmentVariables": {
            # launchd 的 PATH 默认极简（不含 homebrew）；首次运行要 git clone +
            # cmake 编译引擎。PYTHONUNBUFFERED 让日志文件即时可见。
            "PATH": SERVICE_PATH,
            "PYTHONUNBUFFERED": "1",
        },
    }


def _plist_text() -> str:
    return plistlib.dumps(
        _plist_data(), fmt=plistlib.FMT_XML, sort_keys=False).decode("utf-8")


# ── launchctl 调用 ────────────────────────────────────────


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _failed(what: str, p: subprocess.CompletedProcess) -> RuntimeError:
    tail = (p.stderr or p.stdout or "").strip()
    return RuntimeError(f"{what} 失败（exit {p.returncode}）: {tail}")


def _field(text: str, key: str) -> str:
    """从 launchctl print 输出里取 `key = value` 的值（取不到返回空串）。"""
    prefix = f"{key} ="
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _info() -> Optional[dict]:
    """单元状态摘要；未加载（无此单元）返回 None。"""
    p = _launchctl("print", _target())
    if p.returncode != 0:
        return None
    return {k: _field(p.stdout, k) for k in ("state", "pid", "last exit code")}


def _is_loaded() -> bool:
    return _info() is not None


def _bootout() -> None:
    """卸载单元（幂等：未加载时 launchctl 报错，忽略）。"""
    _launchctl("bootout", _target())


def _bootstrap() -> None:
    """加载单元文件（隐含按 RunAtLoad 启动）。"""
    p = _launchctl("bootstrap", _domain(), _plist_path())
    if p.returncode != 0:
        raise _failed("launchctl bootstrap", p)


def _kickstart() -> None:
    """重启单元内的进程（-k = 已在跑就先杀掉）。"""
    p = _launchctl("kickstart", "-k", _target())
    if p.returncode != 0:
        raise _failed("launchctl kickstart", p)


def _report(logger: logging.Logger, wait: float = 1.0) -> None:
    """等 launchd 拉起进程后打印一行结果（未起来时提示看日志）。"""
    if wait:
        time.sleep(wait)
    info = _info()
    if info is None:
        logger.warning("单元未加载，请查看: %s", _plist_path())
        return
    state = info.get("state") or "unknown"
    pid = info.get("pid") or ""
    if state == "running":
        logger.info("状态: running%s（http://localhost:%s）",
                    f"，pid {pid}" if pid else "", WEB_PORT)
    else:
        logger.warning("状态: %s（未在运行，请查看 %s）", state, _stderr_path())


# ── 动作实现 ──────────────────────────────────────────────


def install(logger: logging.Logger, **_: Any) -> int:
    """写单元文件并加载：之后随登录自启、退出自动拉起。"""
    _require_macos()
    os.makedirs(SERVICE_PLIST_DIR, exist_ok=True)
    os.makedirs(SERVICE_LOG_DIR, exist_ok=True)

    plist = _plist_path()
    existed = os.path.exists(plist)
    with open(plist, "w", encoding="utf-8") as f:
        f.write(_plist_text())
    logger.info("%s守护单元: %s", "更新" if existed else "写入", plist)
    logger.info("解释器: %s", _python_bin())
    logger.info("日志: %s / %s", _stdout_path(), _stderr_path())

    if _is_loaded():
        _bootout()          # 先卸旧实例，bootstrap 才会认新单元文件
        logger.info("已卸载旧实例")
    _bootstrap()
    _launchctl("enable", _target())   # 清掉上一次 disable 留下的状态
    _report(logger)
    if (SERVICE_KEEPALIVE or SERVICE_RUN_AT_LOAD) and not _is_running():
        return 1
    return 0


def uninstall(logger: logging.Logger, **_: Any) -> int:
    """卸载单元并删除单元文件（日志文件保留）。"""
    _require_macos()
    plist = _plist_path()
    if _is_loaded():
        _bootout()
        if _is_loaded():
            raise RuntimeError(f"卸载单元失败: {_target()}")
        logger.info("已卸载单元: %s", SERVICE_LABEL)
    else:
        logger.info("单元未加载（无需卸载）")
    if os.path.exists(plist):
        os.remove(plist)
        logger.info("已删除单元文件: %s", plist)
    else:
        logger.info("单元文件不存在: %s", plist)
    logger.info("日志文件保留在: %s", SERVICE_LOG_DIR)
    return 0


def start(logger: logging.Logger, **_: Any) -> int:
    """启动守护：未加载则加载单元，已加载则重启进程。"""
    _require_macos()
    if not os.path.exists(_plist_path()):
        raise RuntimeError(f"守护单元未安装: {_plist_path()}（先运行 service install）")
    if _is_loaded():
        _kickstart()
    else:
        _bootstrap()
    _report(logger)
    return 0 if _is_running() else 1


def stop(logger: logging.Logger, **_: Any) -> int:
    """停止守护：卸载单元（plist 与日志保留，start 可再加载回来）。"""
    _require_macos()
    if not _is_loaded():
        logger.info("单元未加载（已是停止状态）")
        return 0
    _bootout()
    if _is_loaded():
        raise RuntimeError(f"停止单元失败: {_target()}")
    logger.info("已停止守护: %s", SERVICE_LABEL)
    return 0


def restart(logger: logging.Logger, **_: Any) -> int:
    """重启守护（改完 config.py 后用这个让设置生效）。"""
    _require_macos()
    if not os.path.exists(_plist_path()):
        raise RuntimeError(f"守护单元未安装: {_plist_path()}（先运行 service install）")
    if _is_loaded():
        _kickstart()
    else:
        _bootstrap()
    _report(logger)
    return 0 if _is_running() else 1


def status(logger: logging.Logger, **_: Any) -> int:
    """打印守护状态；进程在运行返回 0，否则返回 1。"""
    _require_macos()
    plist = _plist_path()
    info = _info()
    print(f"单元: {SERVICE_LABEL}")
    print(f"单元文件: {plist}" + ("" if os.path.exists(plist) else "（不存在）"))
    if info is None:
        print("状态: stopped（单元未加载）")
        print(f"日志: {SERVICE_LOG_DIR}")
        return 1
    state = info.get("state") or "unknown"
    pid = info.get("pid") or ""
    line = f"状态: {state}"
    if state == "running" and pid:
        line += f"（pid {pid}）"
    print(line)
    exit_code = info.get("last exit code") or ""
    if exit_code:
        print(f"上次退出码: {exit_code}")
    print(f"地址: http://localhost:{WEB_PORT}")
    print(f"日志: {SERVICE_LOG_DIR}")
    return 0 if state == "running" else 1


def logs(logger: logging.Logger, lines: int = SERVICE_LOG_LINES,
         follow: bool = False, **_: Any) -> int:
    """tail 守护日志（stdout + stderr）；follow=True 相当于 tail -f。"""
    _require_macos()
    files = [p for p in (_stdout_path(), _stderr_path()) if os.path.exists(p)]
    if not files:
        raise RuntimeError(f"日志文件不存在: {_stdout_path()}（守护还没运行过？）")
    cmd = ["tail", "-n", str(max(int(lines), 0))]
    if follow:
        cmd.append("-f")
    cmd += files
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:      # tail -f 的常规退出方式
        return 0


_ACTIONS: dict[str, Callable[..., int]] = {
    "install": install,
    "uninstall": uninstall,
    "start": start,
    "stop": stop,
    "restart": restart,
    "status": status,
    "logs": logs,
}


def _is_running() -> bool:
    info = _info()
    return bool(info) and info.get("state") == "running"


def run_service(action: str, logger: logging.Logger, **kwargs: Any) -> int:
    """执行 service 动作并返回进程退出码（未知动作抛 RuntimeError）。"""
    fn = _ACTIONS.get(action)
    if fn is None:
        raise RuntimeError(
            f"未知 service 动作: {action}（可选: {'/'.join(SERVICE_ACTIONS)}）")
    return fn(logger, **kwargs)
