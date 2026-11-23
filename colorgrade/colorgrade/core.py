"""colorgrade 核心 API：白平衡、曲线与三线性插值 3D LUT。

设计要点：
- 白平衡增益按三通道均值归一，灰轴（r=g=b）在等值增益下保持不动。
- 曲线为 v**gamma，单调递增且端点固定在 (0,0)/(1,1)。
- LUT3D 使用扁平不可变 tuple 存储，采样路径只读不写，天然并发安全。
- 非法尺寸抛 LUTSizeError，非有限值抛 LUTValueError，二者均可单独捕获，
  且同为 ValueError 子类以兼容通用异常处理。
"""

import math

__all__ = [
    "Grade",
    "LUT3D",
    "LUTError",
    "LUTSizeError",
    "LUTValueError",
]


class LUTError(ValueError):
    """3D LUT 相关错误基类。"""


class LUTSizeError(LUTError):
    """非法 LUT 尺寸或数据长度与尺寸不匹配。"""


class LUTValueError(LUTError):
    """LUT 数据或像素中出现 NaN/Inf 等非有限值。"""


def _finite_float(value, name, exc=LUTValueError):
    """转为 float 并校验有限性。"""
    try:
        fv = float(value)
    except (TypeError, ValueError) as err:
        raise exc(f"{name} 必须为数值") from err
    if not math.isfinite(fv):
        raise exc(f"{name} 必须为有限值")
    return fv


def _check_lut_size(size):
    if isinstance(size, bool) or not isinstance(size, int):
        raise LUTSizeError("LUT 尺寸必须为不小于 2 的整数")
    if size < 2:
        raise LUTSizeError("LUT 尺寸必须不小于 2")
    return size


class Grade:
    """调色：白平衡增益 + gamma 曲线。"""

    def white_balance(self, rgb, gains):
        """白平衡：逐通道乘以归一化后的增益。

        增益按均值归一（sum/3），等值增益作用于灰轴时输出不变。
        返回 (r, g, b) float 三元组。
        """
        try:
            r, g, b = rgb
        except (TypeError, ValueError) as err:
            raise ValueError("rgb 必须为三通道序列") from err
        try:
            gr, gg, gb = gains
        except (TypeError, ValueError) as err:
            raise ValueError("gains 必须为三通道序列") from err

        r = _finite_float(r, "rgb", ValueError)
        g = _finite_float(g, "rgb", ValueError)
        b = _finite_float(b, "rgb", ValueError)
        gr = _finite_float(gr, "gains", ValueError)
        gg = _finite_float(gg, "gains", ValueError)
        gb = _finite_float(gb, "gains", ValueError)
        if gr <= 0.0 or gg <= 0.0 or gb <= 0.0:
            raise ValueError("白平衡增益必须为正数")

        norm = (gr + gg + gb) / 3.0
        return (r * gr / norm, g * gg / norm, b * gb / norm)

    def curve(self, v, gamma):
        """gamma 曲线：v ** gamma，定义域 [0, 1]，gamma 必须为正数。"""
        fv = _finite_float(v, "v", ValueError)
        fg = _finite_float(gamma, "gamma", ValueError)
        if fg <= 0.0:
            raise ValueError("gamma 必须为正数")
        if not 0.0 <= fv <= 1.0:
            raise ValueError("v 必须位于 [0, 1] 区间")
        return fv ** fg

    def grade_pixel(self, rgb, gains, gamma):
        """逐像素解析式调色：先白平衡再逐通道 gamma。

        白平衡结果可能超过 1.0（增益放大），故此处直接做幂运算而不做
        定义域裁剪；它是 3D LUT 采样结果的解析对照基准。
        """
        r, g, b = self.white_balance(rgb, gains)
        fg = _finite_float(gamma, "gamma", ValueError)
        if fg <= 0.0:
            raise ValueError("gamma 必须为正数")
        return (r ** fg, g ** fg, b ** fg)


class LUT3D:
    """三线性插值 3D LUT。

    节点排列：每轴 size 个点，节点 (i, j, k) 对应归一化坐标
    (i, j, k) / (size - 1)，扁平下标为 (((i*n)+j)*n+k)*3 + 通道。
    采样坐标按 [0, 1] 钳制后做三线性插值。
    """

    def __init__(self, size, values):
        n = _check_lut_size(size)
        try:
            vals = list(values)
        except TypeError as err:
            raise LUTSizeError("values 必须为可迭代的扁平数值序列") from err
        expected = n * n * n * 3
        if len(vals) != expected:
            raise LUTSizeError(
                f"LUT 数据长度应为 {expected}（{n}^3*3），实际为 {len(vals)}"
            )
        table = [0.0] * expected
        for idx, item in enumerate(vals):
            table[idx] = _finite_float(item, f"values[{idx}]")
        self._n = n
        self._table = tuple(table)

    @classmethod
    def from_transform(cls, size, transform):
        """在规则网格上采样逐像素变换 transform(rgb)->rgb 构建 LUT。"""
        n = _check_lut_size(size)
        if not callable(transform):
            raise TypeError("transform 必须是可调用对象")
        scale = n - 1
        table = [0.0] * (n * n * n * 3)
        idx = 0
        for i in range(n):
            ri = i / scale
            for j in range(n):
                gj = j / scale
                for k in range(n):
                    try:
                        out = transform((ri, gj, k / scale))
                        cr, cg, cb = out
                    except (TypeError, ValueError) as err:
                        raise LUTValueError(
                            "transform 必须返回三通道序列"
                        ) from err
                    table[idx] = _finite_float(cr, f"transform[{i},{j},{k}].r")
                    table[idx + 1] = _finite_float(cg, f"transform[{i},{j},{k}].g")
                    table[idx + 2] = _finite_float(cb, f"transform[{i},{j},{k}].b")
                    idx += 3
        return cls(n, table)

    @property
    def size(self):
        return self._n

    @property
    def table(self):
        """扁平不可变节点数据（通道在后）。"""
        return self._table

    def apply_one(self, pixel):
        """单像素三线性插值，返回 (r, g, b)；越界钳制，非有限值报错。"""
        try:
            rf, gf, bf = pixel
        except (TypeError, ValueError) as err:
            raise LUTValueError("像素必须为三通道序列") from err
        rf = _finite_float(rf, "pixel.r")
        gf = _finite_float(gf, "pixel.g")
        bf = _finite_float(bf, "pixel.b")

        if rf < 0.0:
            rf = 0.0
        elif rf > 1.0:
            rf = 1.0
        if gf < 0.0:
            gf = 0.0
        elif gf > 1.0:
            gf = 1.0
        if bf < 0.0:
            bf = 0.0
        elif bf > 1.0:
            bf = 1.0

        n = self._n
        last = n - 2
        x = rf * (n - 1)
        y = gf * (n - 1)
        z = bf * (n - 1)
        i = int(x)
        j = int(y)
        k = int(z)
        if i > last:
            i = last
        if j > last:
            j = last
        if k > last:
            k = last
        tx = x - i
        ty = y - j
        tz = z - k

        table = self._table
        # i 轴（r）步长 n*n*3，j 轴（g）步长 n*3，k 轴（b）步长 3
        step_n = n * 3
        step_nn = n * n * 3
        base = (((i * n) + j) * n + k) * 3

        out = [0.0, 0.0, 0.0]
        for ch in range(3):
            o = base + ch
            c000 = table[o]
            c001 = table[o + 3]
            c010 = table[o + step_n]
            c011 = table[o + step_n + 3]
            c100 = table[o + step_nn]
            c101 = table[o + step_nn + 3]
            c110 = table[o + step_nn + step_n]
            c111 = table[o + step_nn + step_n + 3]

            # 先沿 r（i 轴，tx），再沿 g（j 轴，ty），最后沿 b（k 轴，tz）
            x00 = c000 + (c100 - c000) * tx
            x01 = c001 + (c101 - c001) * tx
            x10 = c010 + (c110 - c010) * tx
            x11 = c011 + (c111 - c011) * tx
            y0 = x00 + (x10 - x00) * ty
            y1 = x01 + (x11 - x01) * ty
            out[ch] = y0 + (y1 - y0) * tz
        return (out[0], out[1], out[2])

    def apply(self, pixels):
        """批量采样：可迭代的三通道像素 -> list[(r, g, b)]。"""
        one = self.apply_one
        return [one(pixel) for pixel in pixels]
