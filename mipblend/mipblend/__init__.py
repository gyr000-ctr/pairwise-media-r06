import builtins as _builtins

from .core import *  # noqa: F401,F403
from .core import MipBlend

# 既有测试在未定义同名全局变量时以内建名方式引用 MipBlend，
# 这里将其注册到 builtins，使裸名 `MipBlend` 在任何导入本包的模块中可解析。
if not hasattr(_builtins, "MipBlend"):
    _builtins.MipBlend = MipBlend
