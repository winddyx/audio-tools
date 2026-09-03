"""Web 薄入口（实现见 tools/web_main.py，设置见 ov/settings.py 顶部变量）。

启动: uv run python web.py   （默认 http://localhost:38001）
"""

import sys

from tools.web_main import main

if __name__ == "__main__":
    sys.exit(main())
