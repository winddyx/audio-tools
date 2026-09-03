"""CLI 薄入口（实现见 tools/cli_main.py，设置见 ov/settings.py 顶部变量）。

用法:
  uv run python cli.py <text.txt> [<ref.wav>]   # 合成
  uv run python cli.py --asr <ref.wav>          # ASR 转写
"""

import sys

from tools.cli_main import main

if __name__ == "__main__":
    sys.exit(main())
