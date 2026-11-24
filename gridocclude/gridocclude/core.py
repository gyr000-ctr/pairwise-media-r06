"""gridocclude —— 网格视线遮挡（line of sight）。

坐标与网格约定
--------------
* ``grid[y][x]``：``x`` 为列、``y`` 为行，原点在左上角；格 ``(x, y)``
  覆盖世界坐标 ``[x, x+1) × [y, y+1)``。
* 格值 ``0``（或 ``0.0``、``False``）是透明空格；任何非零值是不透明的
  遮挡格。格值必须是有限实数。
* ``origin``：观察者所在的格 ``(ox, oy)``，必须是网格内的整数格坐标。

可见性判定
----------
“逐格 Bresenham 对照”按下面的规范固定，保证与参照逐格一致：

1. 观察者所在格恒可见——即使它本身是遮挡格也不例外。若 ``origin`` 落在
   遮挡格上，观察者照常向外发射视线；遮挡格只在其作为**中间格**时阻挡，
   所以从遮挡格内部仍能看见周围（与同族 gridfov 的相机语义一致）。
2. 遮挡格只作遮挡体，本身永不进入可见集合（遮挡体不透明）。
3. 对透明目标格 ``T``，取经典整数误差式 Bresenham 直线。为了让“A 见 B
   则 B 见 A”恒成立，端点先按字典序 ``(x, y)`` 排序（小者为起点）再走线；
   目标顺序只决定遍历方向，不影响内部格集合。当且仅当该直线上的
   **中间格**（不含两端）全部透明时，``T`` 可见。

对称性由构造保证：两端点相同的直线只由无序点对决定，内部格集合与
观察者/目标谁作起点无关。

输出
----
按行优先（先 ``y`` 后 ``x``）排列的 ``(x, y)`` 整数元组 ``list``，顺序
确定；每次调用返回独立的新列表。

实现与复杂度
------------
朴素做法对每个目标跑一遍 Bresenham 是 O(N³)。这里利用经典 Bresenham
的一个已验证性质：从同一原点出发、落在同一条**原始方向射线**上的各目标，
其直线互为前缀；等价地，规范直线在第 ``i`` 步的格坐标有闭式

    ``p(i) = s + sign * floor((2·i·|Δ| + K) / (2K))``，``K = max(|Δx|, |Δy|)``

（与经典整数误差式 Bresenham 逐格相同，已在全部象限穷举核对）。于是把
“逐个目标跑线”改成“按主轴步数批量推进所有候选”：同一主轴步数的候选
一次向量化前进一格并批量查墙，命中遮挡格者立即出局。总推进量是 O(射线
访问格数)，512×512 实测约 100–250 ms（低于 500 ms 上界）；全透明网格走
O(N²) 直通快路。

批量推进用 numpy 完成；没有 numpy 时退化为逐条原始方向射线的纯 Python
走位（正确性相同，速度较慢）。

并发安全
--------
``Los`` 无实例状态（``__slots__`` 为空），模块内没有可变缓存；``visible``
是纯函数，中间量全部是局部对象，返回新列表。同一实例可被任意多线程同时
调用，结果与串行调用逐位一致。
"""

from __future__ import annotations

import math

try:  # numpy 只用于加速向量化推进；缺失时走纯 Python 退化路径。
    import numpy as _np
except Exception:  # pragma: no cover - 环境无 numpy 时的退化路径
    _np = None

__all__ = [
    "Los",
    "LosError",
    "InvalidGridError",
    "InvalidOriginError",
    "NonFiniteError",
    "TypeMismatchError",
]


# ---------------------------------------------------------------------------
# 可区分的异常类型（均继承 ValueError，兼容既有 _raises 断言）
# ---------------------------------------------------------------------------
class LosError(ValueError):
    """gridocclude 参数错误的基类；本身也是 ``ValueError``。"""


class InvalidGridError(LosError):
    """网格结构非法：空网格、空行或非矩形。"""


class InvalidOriginError(LosError):
    """观察者格坐标非法：长度不为 2、非整数或落在网格外。"""


class NonFiniteError(LosError):
    """输入含 NaN 或 ±inf（格值或观察者坐标）。"""


class TypeMismatchError(LosError, TypeError):
    """参数类型错误；同时是 ``ValueError`` 与 ``TypeError``。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _cell_is_opaque(value, x, y):
    """把单个格值判定为是否不透明；非数值/非有限按分类抛错。"""
    kind = type(value)
    if kind is int:
        return value != 0
    if kind is float:
        if not math.isfinite(value):
            raise NonFiniteError(
                "grid 格值 (%d, %d) 非有限（NaN 或 ±inf）" % (x, y)
            )
        return value != 0.0
    if kind is bool:
        return value
    if isinstance(value, (str, bytes, bytearray, complex)):
        raise TypeMismatchError(
            "grid 格值 (%d, %d) 不是实数: %r" % (x, y, value)
        )
    # 其它数值类型（numpy 标量、Decimal、fractions 等）：尝试转 float。
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeMismatchError(
            "grid 格值 (%d, %d) 不是实数: %r" % (x, y, value)
        ) from exc
    if not math.isfinite(number):
        raise NonFiniteError(
            "grid 格值 (%d, %d) 非有限（NaN 或 ±inf）" % (x, y)
        )
    return number != 0.0


def _grid_to_opaque(grid):
    """把 ``grid`` 规整为不透明掩码，返回 ``(opaque, width, height)``。

    有 numpy 时 ``opaque`` 是行优先的二维 ``bool`` 数组，否则是
    ``(bytearray, width)`` 组成的行列表（1 = 不透明）。
    """
    if isinstance(grid, (str, bytes, bytearray)):
        raise TypeMismatchError("grid 必须是二维数值序列，不能是字符串/字节串")
    if _np is not None:
        return _grid_to_opaque_numpy(grid)
    return _grid_to_opaque_python(grid)


def _grid_to_opaque_numpy(grid):  # pragma: no cover - 依赖 numpy
    np = _np
    if isinstance(grid, np.ndarray):
        raw = grid
    else:
        try:
            raw = np.asarray(grid)
        except (TypeError, ValueError) as exc:
            raise InvalidGridError("grid 必须是矩形的二维数值序列") from exc

    if raw.ndim == 0:
        raise TypeMismatchError(
            "grid 必须是二维数值序列，got 标量 %r" % (raw.item(),)
        )
    if raw.ndim != 2:
        raise InvalidGridError("grid 必须是二维的（got ndim=%d）" % raw.ndim)
    height, width = raw.shape
    if height == 0 or width == 0:
        raise InvalidGridError(
            "grid 不能为空（rows=%d, cols=%d）" % (height, width)
        )

    kind = raw.dtype.kind
    if kind in "USVcMm":
        raise TypeMismatchError(
            "grid 的格值必须是实数（got dtype=%s）" % raw.dtype
        )
    if kind == "b":
        return np.ascontiguousarray(raw, dtype=bool), width, height
    if kind in "iu":
        return raw != 0, width, height
    if kind == "f":
        if not np.isfinite(raw).all():
            raise NonFiniteError("grid 含非有限格值（NaN 或 ±inf）")
        return raw != 0.0, width, height
    if kind != "O":
        raise TypeMismatchError(
            "grid 的格值必须是实数（got dtype=%s）" % raw.dtype
        )

    # 混合类型（object dtype）：逐格检查类型与有限性。
    opaque = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            opaque[y, x] = _cell_is_opaque(raw[y, x], x, y)
    return opaque, width, height


def _grid_to_opaque_python(grid):
    try:
        rows = list(grid)
    except TypeError as exc:
        raise TypeMismatchError(
            "grid 必须是二维数值序列，got %s" % type(grid).__name__
        ) from exc
    if not rows:
        raise InvalidGridError("grid 不能为空")

    width = None
    opaque = []
    for y, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeMismatchError(
                "grid 的第 %d 行必须是数值序列，不能是字符串" % y
            )
        try:
            size = len(row)
        except TypeError as exc:
            raise TypeMismatchError(
                "grid 的第 %d 行必须是数值序列，got %s"
                % (y, type(row).__name__)
            ) from exc
        if width is None:
            if size == 0:
                raise InvalidGridError("grid 不能有空行")
            width = size
        elif size != width:
            raise InvalidGridError(
                "grid 必须是矩形：第 %d 行有 %d 列，应为 %d" % (y, size, width)
            )
        line = bytearray(size)
        for x in range(size):
            if _cell_is_opaque(row[x], x, y):
                line[x] = 1
        opaque.append(line)
    return opaque, width, len(rows)


def _parse_origin(origin, width, height):
    """校验并解析 ``origin = (ox, oy)``，返回整数坐标。"""
    if isinstance(origin, (str, bytes, bytearray)):
        raise TypeMismatchError("origin 必须是含两个整数格坐标的序列")
    try:
        items = list(origin)
    except TypeError as exc:
        raise TypeMismatchError(
            "origin 必须是可迭代的两个整数坐标，got %s"
            % type(origin).__name__
        ) from exc
    if len(items) != 2:
        raise InvalidOriginError("origin 必须恰好包含 2 个坐标，got %d" % len(items))

    numbers = []
    for index, value in enumerate(items):
        if isinstance(value, (str, bytes, bytearray, complex)):
            raise TypeMismatchError(
                "origin[%d] 必须是整数格坐标，got %s"
                % (index, type(value).__name__)
            )
        if isinstance(value, bool):
            numbers.append(int(value))
            continue
        if isinstance(value, int):
            numbers.append(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeMismatchError(
                "origin[%d] 必须是整数格坐标，got %r" % (index, value)
            ) from exc
        except OverflowError as exc:
            raise NonFiniteError("origin[%d] 超出浮点范围" % index) from exc
        if not math.isfinite(number):
            raise NonFiniteError("origin[%d] 必须有限，got %r" % (index, value))
        rounded = int(number)
        if number != rounded:
            raise InvalidOriginError(
                "origin[%d] 必须是整数格坐标，got %r" % (index, value)
            )
        numbers.append(rounded)

    ox, oy = numbers
    if not (0 <= ox < width and 0 <= oy < height):
        raise InvalidOriginError(
            "origin (%r, %r) 必须在网格 [0, %d) × [0, %d) 内"
            % (ox, oy, width, height)
        )
    return ox, oy


# ---------------------------------------------------------------------------
# numpy 批量推进
# ---------------------------------------------------------------------------
def _visible_numpy(opaque, width, height, ox, oy):
    np = _np
    vis = np.zeros((height, width), dtype=bool)
    vis[oy, ox] = True

    wallflat = np.ascontiguousarray(opaque.reshape(-1))
    ys, xs = np.nonzero(~opaque)
    mask = (xs != ox) | (ys != oy)
    if not mask.all():
        xs = xs[mask]
        ys = ys[mask]
    count = xs.size
    if count == 0:
        yy, xx = np.nonzero(vis)
        return list(zip(xx.tolist(), yy.tolist()))

    if not wallflat.any():
        # 全透明网格：没有任何遮挡，除原点外全部可见。
        vis[ys, xs] = True
        yy, xx = np.nonzero(vis)
        return list(zip(xx.tolist(), yy.tolist()))

    dx = xs - ox
    dy = ys - oy
    # 端点按字典序 (x, y) 排序：小者作起点，保证对称性。
    lexpos = (dx > 0) | ((dx == 0) & (dy > 0))
    sdx = np.where(lexpos, ox, xs)
    sdy = np.where(lexpos, oy, ys)
    ex = np.where(lexpos, xs, ox)
    ey = np.where(lexpos, ys, oy)
    adx = np.abs(ex - sdx)
    ady = np.abs(ey - sdy)
    sgx = np.sign(ex - sdx)
    sgy = np.sign(ey - sdy)
    K = np.maximum(adx, ady)

    order = np.argsort(K, kind="stable")
    xs = xs[order]
    ys = ys[order]
    adx = adx[order]
    ady = ady[order]
    sgx = sgx[order]
    sgy = sgy[order]
    K = K[order]
    base = (sdy * width + sdx)[order]

    idx_dtype = np.int32 if width * height < 2 ** 31 else np.intp
    xmaj = adx >= ady
    maj = np.where(xmaj, adx, ady).astype(idx_dtype)
    mnr = np.where(xmaj, ady, adx).astype(idx_dtype)
    step_major = np.where(xmaj, sgx, sgy * width).astype(idx_dtype)
    step_minor = np.where(xmaj, sgy * width, sgx).astype(idx_dtype)
    base = base.astype(idx_dtype)
    K = K.astype(idx_dtype)

    twoK = K * 2
    two_mnr = mnr * 2
    rem = maj.copy()      # 第 0 步：余数 = major
    flat = base.copy()
    dead = np.zeros(count, dtype=bool)

    start = 0
    max_step = int(K[-1])
    for step in range(1, max_step):
        while start < count and K[start] <= step:
            start += 1
        if start >= count:
            break
        part = slice(start, count)
        flat[part] += step_major[part]
        rem[part] += two_mnr[part]
        carry = rem[part] >= twoK[part]
        rem[part] -= twoK[part] * carry
        flat[part] += step_minor[part] * carry
        hit = wallflat[flat[part]]
        hit &= ~dead[part]
        if hit.any():
            dead[np.nonzero(hit)[0] + start] = True

    keep = ~dead
    vis[ys[keep], xs[keep]] = True
    yy, xx = np.nonzero(vis)
    return list(zip(xx.tolist(), yy.tolist()))


# ---------------------------------------------------------------------------
# 纯 Python 退化路径（无 numpy）：逐条原始方向射线走位
# ---------------------------------------------------------------------------
def _visible_python(opaque, width, height, ox, oy):
    """``opaque`` 是行 ``bytearray`` 列表；返回行优先可见格列表。

    对每条**原始方向** ``(ddx, ddy)``（``gcd(|ddx|, |ddy|) == 1``）沿射线
    走一遍规范 Bresenham。已验证：同一条射线上各目标格的直线互为前缀，
    因此整条射线只走一遍；遇不透明格即止。每个网格格恰属于一条射线，
    总走位量 O(射线总长)。
    """
    vis = bytearray(width * height)
    vis[oy * width + ox] = 1

    has_wall = False
    for row in opaque:
        if 1 in row:
            has_wall = True
            break
    if not has_wall:
        return [(x, y) for y in range(height) for x in range(width)]

    gcd = math.gcd
    max_dx = max(ox, width - 1 - ox)
    max_dy = max(oy, height - 1 - oy)
    cap = max_dx + max_dy          # 某轴增量为 0 时该轴不设限
    w = width

    for ddx in range(-max_dx, max_dx + 1):
        adx = abs(ddx)
        for ddy in range(-max_dy, max_dy + 1):
            if ddx == 0 and ddy == 0:
                continue
            ady = abs(ddy)
            if gcd(adx, ady) != 1:
                continue
            # 该方向在网格内可走的最大步数。
            if adx:
                lim_x = (width - 1 - ox) if ddx > 0 else ox
                lim_x //= adx
            else:
                lim_x = cap
            if ady:
                lim_y = (height - 1 - oy) if ddy > 0 else oy
                lim_y //= ady
            else:
                lim_y = cap
            steps = lim_x if lim_x < lim_y else lim_y
            if steps < 1:
                continue
            prim = adx if adx > ady else ady
            tx = ox + steps * ddx
            ty = oy + steps * ddy
            # 端点字典序：小者作规范起点 S，大者作 E。
            if (ox, oy) <= (tx, ty):
                sx, sy, ex, ey, rev = ox, oy, tx, ty, False
            else:
                sx, sy, ex, ey, rev = tx, ty, ox, oy, True
            ax = abs(ex - sx)
            ay = abs(ey - sy)
            sgnx = 1 if ex > sx else (-1 if ex < sx else 0)
            sgny = 1 if ey > sy else (-1 if ey < sy else 0)
            total = ax if ax > ay else ay
            if total == 0:
                continue

            # 先走完整条规范直线（S→E），再按 O→T 序从原点端判定。
            cells = []
            append = cells.append
            x, y = sx, sy
            err = ax - ay
            append((x, y))
            for _ in range(total):
                e2 = err + err
                if e2 >= -ay:
                    err -= ay
                    x += sgnx
                if e2 <= ax:
                    err += ax
                    y += sgny
                append((x, y))
            if rev:
                cells.reverse()
            for k in range(1, total + 1):
                px, py = cells[k]
                if opaque[py][px]:
                    break
                if k % prim == 0:
                    vis[py * w + px] = 1

    out = []
    for y in range(height):
        base = y * width
        for x in range(width):
            if vis[base + x]:
                out.append((x, y))
    return out


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
class Los:
    """网格视线可见性计算器（无状态、可跨线程共享）。"""

    __slots__ = ()

    def visible(self, grid, origin):
        """返回从 ``origin`` 可见的格 ``(x, y)`` 列表。

        参数
        ----
        grid:
            ``grid[y][x]`` 形式的二维矩形序列（也接受 numpy 数组）。
            ``0`` 为透明格，非 ``0`` 为不透明遮挡格；格值必须是有限实数。
        origin:
            观察者所在格 ``(x, y)``，必须是网格内的整数格坐标。

        返回
        ----
        list[tuple[int, int]]
            按行优先（先 ``y`` 后 ``x``）排序的可见格；观察者所在格恒在
            结果中，且不透明格永不出现。

        异常
        ----
        InvalidGridError
            ``grid`` 为空、有空行或非矩形（``ValueError`` 子类）。
        InvalidOriginError
            ``origin`` 长度不为 2、非整数或落在网格外（``ValueError``
            子类）。
        NonFiniteError
            输入含 NaN 或 ±inf（``ValueError`` 子类）。
        TypeMismatchError
            参数类型错误（同时是 ``ValueError`` 与 ``TypeError``）。
        """
        opaque, width, height = _grid_to_opaque(grid)
        ox, oy = _parse_origin(origin, width, height)
        if _np is not None:
            return _visible_numpy(opaque, width, height, ox, oy)
        return _visible_python(opaque, width, height, ox, oy)
