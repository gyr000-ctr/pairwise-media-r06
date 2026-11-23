"""mipblend 核心 API：mip 三线性混合。

- LOD 按各向异性比取：lod = log2(max(sx, sy) / min(sx, sy))，钳制到 [0, levels)。
- blend 返回 (level, frac)：level 为整数层号，frac 为级间权重，0 <= frac < 1，
  两级权重 (1 - frac) 与 frac 和恒为 1，且 lod 连续变化时过渡连续无跳变。
- sample 做两级线性混合，权重和为 1。
- 非法缩放（非正数/类型不符）与非有限导数（NaN/Inf）使用可区分的异常类型。
- 实例初始化后无可变状态，blend/sample 为纯函数，并发安全。
"""

import math

__all__ = [
    "MipBlend",
    "MipBlendError",
    "InvalidScaleError",
    "NonFiniteScaleError",
]


class MipBlendError(ValueError):
    """mipblend 参数错误基类（同时是 ValueError）。"""


class InvalidScaleError(MipBlendError):
    """非法缩放：非正数或类型不是实数。"""


class NonFiniteScaleError(MipBlendError):
    """非有限导数：NaN 或 ±Inf。"""


class MipBlend:
    """按各向异性比计算 mip LOD 并做两级三线性混合。

    无内部可变状态，方法可安全并发调用。
    """

    __slots__ = ("_levels", "_max_lod")

    def __init__(self, levels=8):
        if isinstance(levels, bool) or not isinstance(levels, int):
            raise InvalidScaleError(
                "levels 必须为正整数，得到 %r" % (levels,)
            )
        if levels < 1:
            raise InvalidScaleError("levels 必须 >= 1，得到 %r" % (levels,))
        self._levels = levels
        self._max_lod = float(levels - 1)

    @property
    def levels(self):
        return self._levels

    def blend(self, scale_x, scale_y):
        """返回 (level, frac)：level ∈ [0, levels)，frac ∈ [0, 1)。"""
        try:
            sx = float(scale_x)
            sy = float(scale_y)
        except (TypeError, ValueError):
            raise InvalidScaleError(
                "scale_x/scale_y 必须为实数，得到 %r, %r" % (scale_x, scale_y)
            ) from None
        if not (math.isfinite(sx) and math.isfinite(sy)):
            raise NonFiniteScaleError(
                "scale_x/scale_y 非有限（NaN/Inf）：%r, %r" % (scale_x, scale_y)
            )
        if sx <= 0.0 or sy <= 0.0:
            raise InvalidScaleError(
                "scale_x/scale_y 必须为正数，得到 %r, %r" % (scale_x, scale_y)
            )
        # 各向异性比 >= 1，lod >= 0，只需上钳制到 [0, levels)
        ratio = sx / sy if sx >= sy else sy / sx
        lod = math.log2(ratio)
        max_lod = self._max_lod
        if lod > max_lod:
            lod = max_lod
        level = int(lod)
        return (level, lod - level)

    def sample(self, l0, l1, f):
        """两级混合：l0 * (1 - f) + l1 * f，权重和恒为 1。"""
        try:
            a = float(l0)
            b = float(l1)
            w = float(f)
        except (TypeError, ValueError):
            raise InvalidScaleError(
                "l0/l1/f 必须为实数，得到 %r, %r, %r" % (l0, l1, f)
            ) from None
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(w)):
            raise NonFiniteScaleError(
                "l0/l1/f 非有限（NaN/Inf）：%r, %r, %r" % (l0, l1, f)
            )
        if not 0.0 <= w <= 1.0:
            raise InvalidScaleError("f 必须在 [0, 1]，得到 %r" % (f,))
        return a * (1.0 - w) + b * w
