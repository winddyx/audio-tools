"""
ov：OmniVoice 配音工具核心（模板化重构 v2）

分层：
- src/          模型无关核心（设置/契约/引擎抽象/编排/资产/音频/文本）
- src/models/   具体生成模型（每个模型一个子包：spec 声明 + engine 适配）
- tools/       界面与入口（CLI / Web），只依赖 src
- tests/       单元测试（stub 引擎，不加载真实模型）

规则：
- 一切可调设置集中在 src/settings.py 顶部变量（同名环境变量可覆盖）；
  入口（cli.py/web.py）不设 argparse 配置参数，只接受"引用哪个文件"。
- 日志与一切输出文本为纯文本，不使用 emoji / 特殊符号。
- CLI/Web 一律经 src.api 调用编排层，不直接触碰引擎与模型实现。
"""

from . import settings  # noqa: F401  确保 import src 即加载顶部设置
from . import models as _models  # noqa: F401,E402  导入即注册全部模型到注册表

__all__ = ["settings", "models"]
