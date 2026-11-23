"""3D LUT（三线性插值）及调色解析式的验收测试。

注意：本文件为新增测试，不修改 tests/test_core.py 的任何既有用例。
"""

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import colorgrade
from colorgrade import Grade, LUT3D, LUTSizeError, LUTValueError


# ---------- 独立标量三线性参考实现（与被测代码结构刻意不同） ----------

def _scalar_trilinear(table, n, point):
    """扁平 LUT 的独立三线性参考：逐层嵌套 lerp。"""
    r = min(max(float(point[0]), 0.0), 1.0)
    g = min(max(float(point[1]), 0.0), 1.0)
    b = min(max(float(point[2]), 0.0), 1.0)
    span = n - 1
    rj = r * span
    gj = g * span
    bj = b * span
    ri = min(int(rj), n - 2)
    gi = min(int(gj), n - 2)
    bi = min(int(bj), n - 2)
    tr, tg, tb = rj - ri, gj - gi, bj - bi

    def node(i, j, k, ch):
        return table[(((i * n) + j) * n + k) * 3 + ch]

    def lerp(a, bb, t):
        return a + (bb - a) * t

    out = []
    for ch in range(3):
        along_r = [
            lerp(node(ri, j, k, ch), node(ri + 1, j, k, ch), tr)
            for j in (gi, gi + 1)
            for k in (bi, bi + 1)
        ]
        # along_r 顺序：(gi,bi),(gi,bi+1),(gi+1,bi),(gi+1,bi+1)
        y0 = lerp(along_r[0], along_r[1], tb)
        y1 = lerp(along_r[2], along_r[3], tb)
        out.append(lerp(y0, y1, tg))
    return tuple(out)


def _rng():
    rnd = random.Random(20261124)
    return rnd


def _make_lut(n, rnd):
    return LUT3D(n, [rnd.uniform(-0.5, 1.5) for _ in range(n ** 3 * 3)])


# ---------- 三线性插值与标量解析式一致性（≤1e-9） ----------

@pytest.mark.parametrize("n", [2, 3, 8, 17, 33])
def test_trilinear_matches_scalar_reference(n):
    rnd = _rng()
    lut = _make_lut(n, rnd)
    points = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(500)]
    for p in points:
        got = lut.apply_one(p)
        ref = _scalar_trilinear(lut.table, n, p)
        for a, b in zip(got, ref):
            assert math.isfinite(a)
            assert abs(a - b) <= 1e-9

    # 批量接口与单像素接口结果一致
    batch = lut.apply(points)
    assert batch == [lut.apply_one(p) for p in points]


def test_identity_lut_is_exact():
    """恒等变换的 LUT 对任意采样点应还原输入（线性插值精确）。"""
    lut = LUT3D.from_transform(16, lambda p: (p[0], p[1], p[2]))
    rnd = _rng()
    for _ in range(200):
        p = (rnd.random(), rnd.random(), rnd.random())
        out = lut.apply_one(p)
        assert out == pytest.approx(p, abs=1e-12)


def test_linear_transform_matches_analytic_off_grid():
    """可分离线性变换下，任意点三线性插值必须与解析式一致（≤1e-9）。"""
    slope, offset = (0.7, 1.3, 0.4), (0.05, -0.1, 0.2)

    def analytic(p):
        return (
            slope[0] * p[0] + offset[0],
            slope[1] * p[1] + offset[1],
            slope[2] * p[2] + offset[2],
        )

    lut = LUT3D.from_transform(10, analytic)
    rnd = _rng()
    for _ in range(500):
        p = (rnd.random(), rnd.random(), rnd.random())
        got, ref = lut.apply_one(p), analytic(p)
        for a, b in zip(got, ref):
            assert abs(a - b) <= 1e-9


def test_lut_nodes_equal_analytic_transform():
    """网格节点上 LUT 输出必须与逐像素解析式逐位一致。"""
    grade = Grade()
    gains, gamma = (1.6, 0.8, 1.2), 2.2
    lut = LUT3D.from_transform(
        9, lambda p: grade.grade_pixel(p, gains, gamma)
    )
    n = lut.size
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = (i / (n - 1), j / (n - 1), k / (n - 1))
                assert lut.apply_one(p) == pytest.approx(
                    grade.grade_pixel(p, gains, gamma), abs=1e-12
                )


def test_lut_approximation_converges_to_analytic():
    """LUT 是解析式的网格近似：尺寸越大，任意点最大误差越小。"""
    grade = Grade()
    gains, gamma = (1.6, 0.8, 1.2), 2.2
    rnd = _rng()
    points = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(2000)]

    def max_error(n):
        lut = LUT3D.from_transform(
            n, lambda p: grade.grade_pixel(p, gains, gamma)
        )
        worst = 0.0
        for p in points:
            ref = grade.grade_pixel(p, gains, gamma)
            got = lut.apply_one(p)
            worst = max(worst, max(abs(a - b) for a, b in zip(got, ref)))
        return worst

    err_coarse = max_error(17)
    err_fine = max_error(65)
    assert err_fine < err_coarse
    assert err_fine < 1e-3


# ---------- 越界钳制与可配尺寸 ----------

@pytest.mark.parametrize("n", [2, 5, 32])
def test_clamps_out_of_range(n):
    rnd = _rng()
    lut = _make_lut(n, rnd)
    corners = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ]
    for p in [(-1.0, -2.0, -3.0), (1.5, 2.0, 9.0), (-0.1, 0.5, 1.2)]:
        clamped = (min(max(p[0], 0.0), 1.0),
                   min(max(p[1], 0.0), 1.0),
                   min(max(p[2], 0.0), 1.0))
        assert lut.apply_one(p) == pytest.approx(
            lut.apply_one(clamped), abs=1e-12
        )
    # 角点直接取节点值
    assert lut.apply_one((-5.0, -5.0, -5.0)) == pytest.approx(
        tuple(lut.table[0:3]), abs=1e-15
    )
    top = lut.table[-3:]
    assert lut.apply_one((5.0, 5.0, 5.0)) == pytest.approx(
        tuple(top), abs=1e-15
    )
    for c in corners:
        assert all(math.isfinite(v) for v in lut.apply_one(c))


# ---------- 非法尺寸与非有限值：可区分报错 ----------

@pytest.mark.parametrize("bad", [0, 1, -3])
def test_rejects_bad_size(bad):
    with pytest.raises(LUTSizeError):
        LUT3D(bad, [])
    with pytest.raises(LUTSizeError):
        LUT3D.from_transform(bad, lambda p: p)


def test_rejects_non_integer_size():
    with pytest.raises(LUTSizeError):
        LUT3D(8.0, [0.0] * (8 ** 3 * 3))
    with pytest.raises(LUTSizeError):
        LUT3D(True, [0.0] * 8)  # bool 不算合法尺寸


def test_rejects_wrong_length():
    with pytest.raises(LUTSizeError):
        LUT3D(4, [0.0] * (4 ** 3 * 3 - 1))
    with pytest.raises(LUTSizeError):
        LUT3D(4, [0.0] * (4 ** 3 * 3 + 1))


def test_rejects_non_finite_table_values():
    with pytest.raises(LUTValueError):
        LUT3D(2, [float("nan")] + [0.0] * 23)
    with pytest.raises(LUTValueError):
        LUT3D(2, [float("inf")] + [0.0] * 23)


def test_rejects_non_finite_pixel():
    lut = _make_lut(4, _rng())
    with pytest.raises(LUTValueError):
        lut.apply_one((float("nan"), 0.1, 0.1))
    with pytest.raises(LUTValueError):
        lut.apply_one((0.1, float("inf"), 0.1))


def test_error_types_are_distinct_and_catchable():
    assert issubclass(LUTSizeError, ValueError)
    assert issubclass(LUTValueError, ValueError)
    assert LUTSizeError is not LUTValueError
    # 通用 ValueError 捕获仍可兜住两类错误
    for fn in [
        lambda: LUT3D(1, []),
        lambda: LUT3D(2, [float("nan")] + [0.0] * 23),
    ]:
        with pytest.raises(ValueError):
            fn()


def test_from_transform_propagates_non_finite_output():
    def bad_transform(p):
        if p == (0.0, 0.0, 0.0):
            return (float("nan"), 0.0, 0.0)
        return p

    with pytest.raises(LUTValueError):
        LUT3D.from_transform(3, bad_transform)


# ---------- 白平衡：增益归一、灰轴不动 ----------

def test_white_balance_gray_axis_unchanged():
    grade = Grade()
    for gains in [(1.0, 1.0, 1.0), (3.0, 3.0, 3.0), (0.25, 0.25, 0.25)]:
        for v in (0.0, 0.25, 0.5, 0.9, 1.0):
            out = grade.white_balance((v, v, v), gains)
            for ch in out:
                assert abs(ch - v) <= 1e-6


def test_white_balance_gain_direction_and_normalization():
    grade = Grade()
    out = grade.white_balance((0.5, 0.5, 0.5), (2.0, 1.0, 1.0))
    # 归一化增益 = (1.5, 0.75, 0.75)
    assert out == pytest.approx((0.75, 0.375, 0.375), abs=1e-12)
    assert all(isinstance(v, float) for v in out)


def test_white_balance_rejects_bad_gains():
    grade = Grade()
    with pytest.raises(ValueError):
        grade.white_balance((0.5, 0.5, 0.5), (1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        grade.white_balance((0.5, 0.5, 0.5), (1.0, -1.0, 1.0))
    with pytest.raises(ValueError):
        grade.white_balance((float("nan"), 0.5, 0.5), (1.0, 1.0, 1.0))


# ---------- 曲线：单调、端点固定、确定性 ----------

def test_curve_endpoints_and_monotonicity():
    grade = Grade()
    for gamma in (0.5, 1.0, 2.2, 4.0):
        assert grade.curve(0.0, gamma) == 0.0
        assert abs(grade.curve(1.0, gamma) - 1.0) < 1e-9
        xs = [i / 16 for i in range(17)]
        ys = [grade.curve(x, gamma) for x in xs]
        assert all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
        assert isinstance(grade.curve(0.5, gamma), float)


def test_curve_deterministic_and_validated():
    grade = Grade()
    assert grade.curve(0.4, 2.2) == grade.curve(0.4, 2.2)
    with pytest.raises(ValueError):
        grade.curve(0.5, 0.0)
    with pytest.raises(ValueError):
        grade.curve(0.5, -1.0)
    with pytest.raises(ValueError):
        grade.curve(1.5, 2.2)
    with pytest.raises(ValueError):
        grade.curve(float("inf"), 2.2)


# ---------- 性能：10 万像素 ≤ 400 ms ----------

def test_performance_100k_pixels():
    rnd = _rng()
    lut = _make_lut(33, rnd)
    pixels = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(100_000)]
    start = time.perf_counter()
    out = lut.apply(pixels)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert len(out) == 100_000
    assert elapsed_ms <= 400.0, f"耗时 {elapsed_ms:.1f} ms 超过 400 ms 预算"


# ---------- 并发安全 ----------

def test_concurrent_apply_is_safe_and_consistent():
    rnd = _rng()
    lut = _make_lut(17, rnd)
    pixels = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(20_000)]
    expected = lut.apply(pixels)

    n_threads = 8
    chunk = (len(pixels) + n_threads - 1) // n_threads
    results = [None] * n_threads
    errors = []

    def worker(tid):
        try:
            part = pixels[tid * chunk:(tid + 1) * chunk]
            results[tid] = lut.apply(part)
        except Exception as err:  # pragma: no cover - 仅用于并发失败上报
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    merged = [px for part in results if part for px in part]
    assert merged == expected


def test_concurrent_build_and_apply():
    rnd = _rng()
    pixels = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(5_000)]
    errors = []

    def build(tid):
        try:
            lut = LUT3D.from_transform(
                12, lambda p: Grade().grade_pixel(p, (1.2, 0.9, 1.1), 1.8)
            )
            lut.apply(pixels)
        except Exception as err:  # pragma: no cover
            errors.append(err)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(build, range(8)))
    assert not errors


# ---------- 不可变性与既有 API 兼容 ----------

def test_table_is_immutable():
    lut = _make_lut(3, _rng())
    assert isinstance(lut.table, tuple)
    with pytest.raises(TypeError):
        lut.table[0] = 1.0  # type: ignore[misc]
    # 采样不改变内部状态
    before = lut.table
    lut.apply([(0.3, 0.6, 0.9)] * 100)
    assert lut.table is before


def test_grade_api_surface_unchanged():
    grade = Grade()
    assert hasattr(grade, "white_balance")
    assert hasattr(grade, "curve")
    # 既有 smoke 行为：返回三元组 / 标量 float
    assert len(grade.white_balance((0.1, 0.2, 0.3), (1.0, 2.0, 0.5))) == 3
    assert isinstance(grade.curve(0.7, 2.2), float)
