"""针对 meshsmooth 题面与评审侧边界补充的回归用例。

只新增文件，不改动既有的 tests/test_core.py。
"""

import math
import threading
import time

import pytest

import meshsmooth


def _unit(v):
    n = math.hypot(v[0], v[1], v[2])
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _face_normal(verts, face):
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    return _cross(u, v)


# 两个三角形共享边 (0,1)，二面角 90°。
V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
F = [(0, 1, 2), (0, 1, 3)]


def _s():
    return meshsmooth.Smooth()


def test_normals_unit_and_count():
    ns = _s().normals(V, F, 60.0)
    assert len(ns) == len(V)
    for n in ns:
        assert abs(math.hypot(*n) - 1.0) <= 1e-9


def test_hard_soft_split_by_angle():
    # 90° 二面角：阈值 89° 分两组，阈值 95° 合为一组。
    assert len(_s().smoothing_groups(V, F, 89.0)) == 2
    assert len(_s().smoothing_groups(V, F, 95.0)) == 1
    # 硬边两侧同一顶点的角法线不再相同。
    hard = _s().corner_normals(V, F, 89.0)
    v0 = {hard[fi][k] for fi in range(2) for k in range(3) if F[fi][k] == 0}
    assert len(v0) == 2
    # 软边组内同一顶点的角法线完全一致。
    soft = _s().corner_normals(V, F, 95.0)
    v0 = {soft[fi][k] for fi in range(2) for k in range(3) if F[fi][k] == 0}
    assert len(v0) == 1


def test_grouping_matches_half_edge_traversal():
    # 独立实现：共享边邻接 + 二面角阈值，BFS 求连通分量后与 API 对照。
    V5 = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    F5 = [(0, 1, 2), (1, 3, 2), (0, 1, 4)]
    for angle in (1.0, 45.0, 90.0, 180.0):
        units = [_unit(_face_normal(V5, f)) for f in F5]
        cos_t = math.cos(math.radians(angle))
        edges = {}
        for fi, f in enumerate(F5):
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                edges.setdefault((min(a, b), max(a, b)), []).append(fi)
        adj = {i: set() for i in range(len(F5))}
        for shared in edges.values():
            for i in range(len(shared)):
                for j in range(i + 1, len(shared)):
                    a, b = shared[i], shared[j]
                    dot = sum(units[a][k] * units[b][k] for k in range(3))
                    if dot >= cos_t - 1e-12:
                        adj[a].add(b)
                        adj[b].add(a)
        seen, expect = set(), []
        for start in range(len(F5)):
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                comp.append(node)
                for other in adj[node]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            expect.append(sorted(comp))
        expect = sorted(expect)
        got = sorted(sorted(g) for g in _s().smoothing_groups(V5, F5, angle))
        assert got == expect


def test_area_weighted_matches_brute_force():
    # 单组平滑曲面：逐顶点按面积加权平均面法线，对照误差 ≤1e-6。
    Vq = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 1.0),
    ]
    Fq = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    angle = 180.0
    assert len(_s().smoothing_groups(Vq, Fq, angle)) == 1
    ns = _s().normals(Vq, Fq, angle)
    accum = {i: [0.0, 0.0, 0.0] for i in range(len(Vq))}
    for f in Fq:
        n = _face_normal(Vq, f)
        area = 0.5 * math.hypot(*n)
        u = _unit(n)
        for v in f:
            for k in range(3):
                accum[v][k] += u[k] * area
    for v, total in accum.items():
        brute = _unit(total)
        assert max(abs(brute[k] - ns[v][k]) for k in range(3)) <= 1e-6


def test_degenerate_index_and_nonfinite_are_distinct():
    Vdeg = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]  # 共线
    with pytest.raises(meshsmooth.MeshDegenerateError):
        _s().normals(Vdeg, [(0, 1, 2)], 60.0)
    with pytest.raises(meshsmooth.MeshIndexError):
        _s().normals(V, [(0, 1, 99)], 60.0)
    with pytest.raises(meshsmooth.MeshValueError):
        _s().normals([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, float("nan"), 0.0)],
                     [(0, 1, 2)], 60.0)
    # 三者都是 ValueError 子类但互不为子类，可区分捕获。
    kinds = (meshsmooth.MeshValueError, meshsmooth.MeshIndexError,
             meshsmooth.MeshDegenerateError)
    for a in kinds:
        assert issubclass(a, ValueError)
        for b in kinds:
            if a is not b:
                assert not issubclass(a, b)


def test_bad_angle_and_structure():
    for angle in (0.0, -1.0, 181.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            _s().normals(V, F, angle)
    with pytest.raises(ValueError):
        _s().normals([], F, 60.0)
    with pytest.raises(ValueError):
        _s().normals(V, [], 60.0)
    with pytest.raises(ValueError):
        _s().normals(V, [(0, 1)], 60.0)


def test_deterministic():
    assert _s().normals(V, F, 60.0) == _s().normals(V, F, 60.0)
    assert _s().corner_normals(V, F, 60.0) == _s().corner_normals(V, F, 60.0)


def test_thread_safety():
    baseline = _s().normals(V, F, 60.0)
    errors = []

    def worker():
        try:
            for _ in range(50):
                assert _s().normals(V, F, 60.0) == baseline
        except Exception as exc:  # noqa: BLE001 - 收集后统一断言
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_performance_10k_faces():
    side = 101  # (101-1)^2 * 2 = 20000 个三角形；取一半约 1 万面
    verts = [
        (float(i % side), float(i // side), math.sin(i * 0.37))
        for i in range(side * side)
    ]
    faces = []
    for y in range(side - 1):
        for x in range(side - 1):
            i = y * side + x
            faces.append((i, i + 1, i + side))
            if len(faces) >= 10000:
                break
        if len(faces) >= 10000:
            break
    start = time.perf_counter()
    ns = _s().normals(verts, faces, 60.0)
    elapsed = time.perf_counter() - start
    assert len(ns) == len(verts)
    assert elapsed <= 0.5
