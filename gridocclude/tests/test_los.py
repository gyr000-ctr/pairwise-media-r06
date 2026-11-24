"""gridocclude 题面约束的验收用例（``-k los`` 选中本文件）。

覆盖：可见集合与「逐格 Bresenham」独立参照逐格一致、对称性（A 见 B 则
B 见 A）、遮挡体不透明、512×512 性能上界、非法网格/观察者分类报错、非有限
输入单独分类、并发安全、结果确定且为独立新列表。

参照片段刻意不复用被测实现：线段用独立的经典整数误差式 Bresenham，并按
字典序把端点规范成起点，使内部格集合与遍历方向无关（对称性）。
"""

import math
import random
import threading
import time

import pytest

import gridocclude
from gridocclude import (
    InvalidGridError,
    InvalidOriginError,
    Los,
    NonFiniteError,
)

G = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


# ---------------------------------------------------------------------------
# 独立参照实现：经典整数误差式 Bresenham（端点按字典序规范）
# ---------------------------------------------------------------------------
def reference_bresenham(x0, y0, x1, y1):
    """经典整数误差式 Bresenham，返回含两端的完整格序列。"""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def reference_line(origin, target):
    """端点按字典序排序后的规范直线；对称性由构造保证。"""
    if origin[0] < target[0] or (origin[0] == target[0] and origin[1] <= target[1]):
        return reference_bresenham(origin[0], origin[1], target[0], target[1])
    line = reference_bresenham(target[0], target[1], origin[0], origin[1])
    line.reverse()
    return line


def reference_visible(grid, origin):
    """逐格 Bresenham 暴力参照，返回行优先列表。"""
    height = len(grid)
    width = len(grid[0])
    ox, oy = origin
    out = []
    for y in range(height):
        for x in range(width):
            if (x, y) == (ox, oy):
                out.append((x, y))
                continue
            if grid[y][x] != 0:
                continue
            path = reference_line(origin, (x, y))
            if all(grid[gy][gx] == 0 for gx, gy in path[1:-1]):
                out.append((x, y))
    return out


def _random_case(rng, max_size=16, density_choices=(0.0, 0.05, 0.15, 0.3, 0.45)):
    width = rng.randint(1, max_size)
    height = rng.randint(1, max_size)
    density = rng.choice(density_choices)
    grid = [
        [1 if rng.random() < density else 0 for _ in range(width)]
        for _ in range(height)
    ]
    ox = rng.randrange(width)
    oy = rng.randrange(height)
    grid[oy][ox] = 0
    return grid, (ox, oy)


# ---------------------------------------------------------------------------
# 可见集合与参照一致
# ---------------------------------------------------------------------------
def test_los_nonempty():
    assert len(Los().visible(G, (0, 0))) > 0


def test_los_wall_hidden():
    assert (1, 1) not in Los().visible(G, (0, 0))


def test_los_origin_visible():
    assert (0, 0) in Los().visible(G, (0, 0))


def test_los_bounds():
    got = Los().visible(G, (0, 0))
    assert all(0 <= x < 3 and 0 <= y < 3 for x, y in got)


@pytest.mark.parametrize("seed", range(40))
def test_los_matches_bresenham_reference(seed):
    rng = random.Random(seed)
    grid, origin = _random_case(rng)
    assert Los().visible(grid, origin) == reference_visible(grid, origin)


def test_los_matches_reference_with_walls():
    grid = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    for origin in [(0, 0), (3, 3), (0, 3), (3, 0), (1, 0), (0, 1)]:
        assert Los().visible(grid, origin) == reference_visible(grid, origin)


def test_los_open_grid_sees_everything():
    size = 9
    grid = [[0] * size for _ in range(size)]
    for origin in [(0, 0), (4, 4), (8, 8), (0, 8)]:
        assert len(Los().visible(grid, origin)) == size * size


# ---------------------------------------------------------------------------
# 对称性：A 见 B 则 B 见 A
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_los_symmetry(seed):
    rng = random.Random(1000 + seed)
    grid, _ = _random_case(rng, density_choices=(0.0, 0.1, 0.25))
    los = Los()
    cells = [(x, y) for y in range(len(grid)) for x in range(len(grid[0]))]
    for origin in cells:
        if grid[origin[1]][origin[0]] != 0:
            continue
        visible = set(los.visible(grid, origin))
        for target in visible:
            if target == origin:
                continue
            assert origin in set(los.visible(grid, target)), (
                origin, target, grid
            )


# ---------------------------------------------------------------------------
# 遮挡体不透明
# ---------------------------------------------------------------------------
def test_los_occluders_are_opaque():
    got = Los().visible(G, (0, 0))
    assert (1, 1) not in got
    assert all(G[y][x] == 0 for x, y in got)


def test_los_wall_shadows_cells_behind_it():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    got = Los().visible(grid, (2, 0))
    assert (2, 1) not in got  # 墙本身
    assert (2, 2) not in got  # 墙正后方
    assert got == reference_visible(grid, (2, 0))


def test_los_fully_walled_room_only_own_cell():
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    assert Los().visible(grid, (1, 1)) == [(1, 1)]


def test_los_origin_inside_wall_sees_outward():
    # 观察者落在遮挡格上：该格恒可见，且照常向外发射视线；遮挡格只在其
    # 作为中间格时阻挡（与同族 gridfov 的相机语义一致）。
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    got = Los().visible(grid, (1, 1))
    assert (1, 1) in got
    assert got == reference_visible(grid, (1, 1))


# ---------------------------------------------------------------------------
# 性能上界
# ---------------------------------------------------------------------------
def _perf_grid(size, density, seed):
    rng = random.Random(seed)
    return [
        [1 if rng.random() < density else 0 for _ in range(size)]
        for _ in range(size)
    ]


@pytest.mark.parametrize(
    "density,origin",
    [
        (0.0, (256, 256)),
        (0.002, (256, 256)),
        (0.02, (256, 256)),
        (0.1, (0, 0)),
        (0.3, (511, 511)),
    ],
)
def test_los_performance_512_under_500ms(density, origin):
    size = 512
    grid = _perf_grid(size, density, seed=2024)
    ox, oy = origin
    grid[oy][ox] = 0
    culler = Los()
    start = time.perf_counter()
    got = culler.visible(grid, origin)
    elapsed = time.perf_counter() - start
    assert len(got) >= 1
    assert elapsed < 0.5, "512x512 density=%s took %.0f ms" % (density, elapsed * 1000)


def test_los_performance_512_open_grid():
    size = 512
    grid = [[0] * size for _ in range(size)]
    start = time.perf_counter()
    got = Los().visible(grid, (0, 0))
    elapsed = time.perf_counter() - start
    assert len(got) == size * size
    assert elapsed < 0.5, "512x512 open grid took %.0f ms" % (elapsed * 1000)


# ---------------------------------------------------------------------------
# 非法参数：分类报错
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "origin",
    [
        (9, 9),  # 落在网格外
        (-1, 0),
        (0, 2),
        (0, 0, 0),  # 长度过多
        (0,),  # 长度不足
        (0.5, 0.0),  # 非整数
    ],
)
def test_los_bad_origin_raises_value_error(origin):
    with pytest.raises(ValueError):
        Los().visible([[0, 0], [0, 0]], origin)


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [[]],
        [[0, 0], [0]],  # 非矩形
        [[0, 0], [0, 0], [0]],  # 非矩形（尾行较短）
    ],
)
def test_los_bad_grid_raises_value_error(grid):
    with pytest.raises(ValueError):
        Los().visible(grid, (0, 0))


@pytest.mark.parametrize(
    "grid",
    [
        [[0, 0], [0, float("nan")]],
        [[0, float("inf")], [0, 0]],
        [[float("-inf"), 0], [0, 0]],
    ],
)
def test_los_nonfinite_grid_raises(grid):
    with pytest.raises(ValueError):
        Los().visible(grid, (0, 0))


@pytest.mark.parametrize(
    "origin",
    [
        (float("nan"), 0),
        (0, float("inf")),
        (0, float("-inf")),
    ],
)
def test_los_nonfinite_origin_raises(origin):
    with pytest.raises(ValueError):
        Los().visible([[0, 0], [0, 0]], origin)


def test_los_error_types_are_distinguishable():
    grid = [[0, 0], [0, 0]]
    with pytest.raises(InvalidGridError):
        Los().visible([], (0, 0))
    with pytest.raises(InvalidOriginError):
        Los().visible(grid, (9, 9))
    with pytest.raises(NonFiniteError):
        Los().visible([[0, float("nan")]], (0, 0))
    for exc in (InvalidGridError, InvalidOriginError, NonFiniteError):
        assert issubclass(exc, ValueError)


@pytest.mark.parametrize("origin", [None, 3.0, "ab", (0.0, "x")])
def test_los_bad_origin_type_raises_type_error(origin):
    with pytest.raises(TypeError):
        Los().visible([[0, 0], [0, 0]], origin)


@pytest.mark.parametrize("grid", [None, 5, [[0, "x"], [0, 0]], "abcd"])
def test_los_bad_grid_type_raises_type_error(grid):
    with pytest.raises(TypeError):
        Los().visible(grid, (0, 0))


# ---------------------------------------------------------------------------
# 结果形状：类型、确定性、独立性
# ---------------------------------------------------------------------------
def test_los_result_is_list_of_int_tuples():
    got = Los().visible([[0, 0], [0, 0]], (0, 0))
    assert isinstance(got, list)
    assert all(isinstance(cell, tuple) and len(cell) == 2 for cell in got)
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in got)


def test_los_result_row_major_order():
    got = Los().visible([[0] * 5 for _ in range(5)], (2, 2))
    assert got == sorted(got, key=lambda p: (p[1], p[0]))


def test_los_deterministic():
    grid = [[0, 1, 0], [0, 0, 0], [1, 0, 0]]
    los = Los()
    assert los.visible(grid, (0, 0)) == los.visible(grid, (0, 0))


def test_los_returned_list_is_independent():
    los = Los()
    grid = [[0, 0], [0, 0]]
    first = los.visible(grid, (0, 0))
    first.clear()
    assert len(los.visible(grid, (0, 0))) == 4


def test_los_accepts_numpy_grid():
    np = pytest.importorskip("numpy")
    assert Los().visible(np.asarray(G), (0, 0)) == Los().visible(G, (0, 0))


# ---------------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------------
def test_los_concurrent_calls_consistent():
    los = Los()
    cases = [
        (G, (0, 0)),
        ([[0, 0], [0, 0]], (1, 1)),
        ([[0] * 8 for _ in range(8)], (3, 3)),
        ([[0, 1, 0], [0, 0, 0], [1, 0, 0]], (0, 1)),
    ]
    expected = [los.visible(grid, origin) for grid, origin in cases]
    errors = []

    def worker():
        try:
            for _ in range(200):
                for (grid, origin), want in zip(cases, expected):
                    got = los.visible(grid, origin)
                    if got != want:
                        errors.append((grid, origin, got, want))
                        return
                    got.append((0, 0))  # 修改返回值不得影响后续调用
        except Exception as exc:  # pragma: no cover - 失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_los_concurrent_matches_serial():
    size = 128
    grid = [[0] * size for _ in range(size)]
    origin = (1, 1)
    los = Los()
    serial = los.visible(grid, origin)
    results = []
    lock = threading.Lock()

    def worker():
        got = los.visible(grid, origin)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 8
    assert all(result == serial for result in results)


# ---------------------------------------------------------------------------
# 无 numpy 的退化路径：结果必须与参照逐格一致
# ---------------------------------------------------------------------------
def test_los_without_numpy_matches_reference(monkeypatch):
    import gridocclude.core as core

    monkeypatch.setattr(core, "_np", None)
    rng = random.Random(4242)
    los = Los()
    for _ in range(30):
        grid, origin = _random_case(rng)
        assert los.visible(grid, origin) == reference_visible(grid, origin)


def test_los_without_numpy_matches_numpy_path(monkeypatch):
    import gridocclude.core as core

    assert core._np is not None, "本用例假设环境装有 numpy"
    rng = random.Random(909)
    cases = [_random_case(rng, max_size=24) for _ in range(15)]
    los = Los()
    with_numpy = [los.visible(grid, origin) for grid, origin in cases]
    monkeypatch.setattr(core, "_np", None)
    without_numpy = [los.visible(grid, origin) for grid, origin in cases]
    assert without_numpy == with_numpy
