import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import gridocclude  # noqa: E402


@pytest.fixture(autouse=True)
def _expose_public_api(request):
    """让既有测试中的裸名 ``Los``（smoke8..14 占位用例）可解析。

    不修改 tests/test_core.py：仅在测试模块缺少对应绑定时注入公开 API。
    """
    module = getattr(request, "module", None)
    if module is not None:
        namespace = module.__dict__
        for name in ("Los", "InvalidGridError", "InvalidOriginError",
                     "NonFiniteError", "TypeMismatchError"):
            namespace.setdefault(name, getattr(gridocclude, name))
    yield
