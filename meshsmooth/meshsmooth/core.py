"""meshsmooth 核心 API：按平滑角分组的顶点法线。

设计要点：

- 三角形面法线用 ``(b - a) × (c - a)`` 表示，其模长恰为两倍面积，因此直接
  累加该向量即得面积加权平均，无需再乘一次面积系数；归一化后即为单位法线。
- 平滑组以**共享边**为邻接（半边结构里对偶边连通的两条半边），当相邻两面
  单位法线夹角小于等于平滑角阈值时并入同一组，用并查集求连通分量。这与
  沿半边结构做面环遍历得到的连通结果一致：只在顶点相接、不共享边的面不会
  被并到一起。
- :meth:`Smooth.normals` 返回每个顶点一个单位法线（长度等于顶点数）。同一
  顶点若分属多个平滑组，取面积总和最大的组（面积相同取组号较小者），保证
  结果确定且可复现。
- :meth:`Smooth.corner_normals` 返回与每个三角形三个角一一对应的单位法线，
  可完整表达硬边两侧不同的法线。
- :meth:`Smooth.smoothing_groups` 暴露分组结果，便于与半边遍历对照。
- 所有计算均为纯函数，不读写实例可变状态，天然并发安全。

错误分类（都可作为 ``ValueError`` 捕获，但互不为子类，可区分）：

- :class:`MeshValueError`：非有限坐标、角度越界、点/面结构非法；
- :class:`MeshIndexError`：面索引非整数、为负或越界；
- :class:`MeshDegenerateError`：退化面（三点共线或索引重复，面积为零）。
"""

from __future__ import annotations

import math
import operator

__all__ = [
    "Smooth",
    "MeshError",
    "MeshValueError",
    "MeshIndexError",
    "MeshDegenerateError",
]

_COS_TOL = 1e-12
# 大桶（非流形网格）走空间哈希的阈值；不超过此值仍用精确逐对比较。
_BUCKET_LINEAR = 64


class MeshError(ValueError):
    """网格相关错误基类（同时是 ValueError，兼容通用异常处理）。"""


class MeshValueError(MeshError):
    """结构或数值非法：非有限坐标、角度越界、点/面格式错误。"""


class MeshIndexError(MeshError, IndexError):
    """面索引非法：非整数、负数或越界。

    同时继承 ``IndexError``，便于按索引语义单独捕获，又与退化面错误互不为
    子类，可区分。
    """


class MeshDegenerateError(MeshError):
    """退化面：三点共线或索引重复，导致面积为零。"""


def _as_float(value, name):
    """转为有限 float，非数值或 NaN/Inf 抛 MeshValueError。"""
    if isinstance(value, bool):
        raise MeshValueError(f"{name} 必须为数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as err:
        raise MeshValueError(f"{name} 必须为数值") from err
    if not math.isfinite(number):
        raise MeshValueError(f"{name} 必须为有限值")
    return number


def _smoothing_angle(angle):
    """校验平滑角：单位为度，范围 (0, 180]。"""
    value = _as_float(angle, "angle")
    if not 0.0 < value <= 180.0:
        raise MeshValueError("angle 必须位于 (0, 180] 度区间")
    return value


def _read_verts(verts):
    """校验并转为 (x, y, z) 浮点元组列表。"""
    try:
        items = list(verts)
    except TypeError as err:
        raise MeshValueError("verts 必须是可迭代的三维点序列") from err
    if not items:
        raise MeshValueError("verts 不能为空")
    points = []
    for index, point in enumerate(items):
        try:
            coords = list(point)
        except TypeError as err:
            raise MeshValueError(f"verts[{index}] 必须是三维点") from err
        if len(coords) != 3:
            raise MeshValueError(f"verts[{index}] 必须是三维点")
        points.append(
            (
                _as_float(coords[0], f"verts[{index}].x"),
                _as_float(coords[1], f"verts[{index}].y"),
                _as_float(coords[2], f"verts[{index}].z"),
            )
        )
    return points


def _read_faces(faces, vertex_count):
    """校验并转为三角形索引元组列表。"""
    try:
        items = list(faces)
    except TypeError as err:
        raise MeshValueError("faces 必须是可迭代的三角形索引序列") from err
    if not items:
        raise MeshValueError("faces 不能为空")
    triangles = []
    for face_index, face in enumerate(items):
        try:
            indices = list(face)
        except TypeError as err:
            raise MeshValueError(f"faces[{face_index}] 必须是三个索引") from err
        if len(indices) != 3:
            raise MeshValueError(f"faces[{face_index}] 必须是三角形（3 个索引）")
        triangle = []
        for corner, raw in enumerate(indices):
            if isinstance(raw, bool):
                raise MeshIndexError(f"faces[{face_index}][{corner}] 必须为整数索引")
            try:
                value = operator.index(raw)
            except TypeError as err:
                raise MeshIndexError(
                    f"faces[{face_index}][{corner}] 必须为整数索引"
                ) from err
            if value < 0 or value >= vertex_count:
                raise MeshIndexError(
                    f"faces[{face_index}][{corner}]={value} 越界"
                    f"（顶点数 {vertex_count}）"
                )
            triangle.append(value)
        triangles.append((triangle[0], triangle[1], triangle[2]))
    return triangles


def _face_geometry(points, triangles):
    """计算每个面的面积加权法线、面积与单位法线。

    面积加权法线为未归一化的 ``(b-a)×(c-a)``，模长为两倍面积。
    """
    weighted = []
    areas = []
    units = []
    for face_index, (ia, ib, ic) in enumerate(triangles):
        ax, ay, az = points[ia]
        bx, by, bz = points[ib]
        cx, cy, cz = points[ic]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        if nx == 0.0 and ny == 0.0 and nz == 0.0:
            raise MeshDegenerateError(f"faces[{face_index}] 为退化面（面积为零）")
        # math.hypot 逐项缩放，避免各分量平方后溢出
        length = math.hypot(nx, ny, nz)
        if not math.isfinite(length) or length == 0.0:
            raise MeshValueError(f"faces[{face_index}] 法线溢出，坐标量级过大")
        weighted.append((nx, ny, nz))
        areas.append(0.5 * length)
        units.append((nx / length, ny / length, nz / length))
    return weighted, areas, units


def _merge_pairwise(sharing, units, tol_cos, union, find):
    """小桶：逐对比较两条相邻面的法线，夹角合格则合并。"""
    count = len(sharing)
    for i in range(count):
        left = sharing[i]
        ln = units[left]
        for j in range(i + 1, count):
            right = sharing[j]
            if find(left) == find(right):
                continue
            rn = units[right]
            dot = ln[0] * rn[0] + ln[1] * rn[1] + ln[2] * rn[2]
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            if dot >= tol_cos:
                union(left, right)


def _merge_bucket(sharing, units, cos_threshold, union, find):
    """把一个共享边上的所有面按二面角阈值合并成若干连通块。

    小桶走逐对比较（O(k^2) 但 k 很小）；大桶（非流形网格里一条边被大量面
    共用）改在对偶方向做空间哈希 + 泛洪，避免退化到 O(k^2)。
    """
    count = len(sharing)
    tol_cos = cos_threshold - _COS_TOL
    if tol_cos <= -1.0:
        # 阈值 180°：任意两法线夹角都不超过 π，整桶同组。
        first = sharing[0]
        for index in range(1, count):
            union(first, sharing[index])
        return
    if count <= _BUCKET_LINEAR:
        _merge_pairwise(sharing, units, tol_cos, union, find)
        return

    # 单元边长取阈值角对应弦长：夹角合格的两条法线坐标差 < 弦长，
    # 因而一定落在相邻单元，扫描 27 邻域即可覆盖所有合格对。
    theta = math.acos(max(-1.0, min(1.0, cos_threshold)))
    cell = 2.0 * math.sin(theta * 0.5) * (1.0 + 1e-9)
    if cell < 1e-9:
        cell = 1e-9
    inv = 1.0 / cell
    cells = {}
    cell_of = [None] * len(units)
    for face_index in sharing:
        nx, ny, nz = units[face_index]
        key = (
            math.floor(nx * inv),
            math.floor(ny * inv),
            math.floor(nz * inv),
        )
        cell_of[face_index] = key
        cells.setdefault(key, {})[face_index] = None

    remaining = set(sharing)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    while remaining:
        seed = remaining.pop()
        cells[cell_of[seed]].pop(seed, None)
        stack = [seed]
        while stack:
            current = stack.pop()
            cn = units[current]
            cx, cy, cz = cell_of[current]
            for dx, dy, dz in offsets:
                bucket = cells.get((cx + dx, cy + dy, cz + dz))
                if not bucket:
                    continue
                matched = []
                for other in bucket:
                    on = units[other]
                    dot = cn[0] * on[0] + cn[1] * on[1] + cn[2] * on[2]
                    if dot > 1.0:
                        dot = 1.0
                    elif dot < -1.0:
                        dot = -1.0
                    if dot >= tol_cos:
                        matched.append(other)
                for other in matched:
                    del bucket[other]
                    remaining.discard(other)
                    union(current, other)
                    stack.append(other)


def _group_faces(triangles, units, angle_deg):
    """按共享边 + 二面角阈值求平滑组连通分量。

    返回 ``(groups, group_of)``：``groups`` 为按最小面号排序的面号列表，
    ``group_of[fi]`` 为第 fi 个面所属组号。
    """
    face_count = len(triangles)
    parent = list(range(face_count))

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left, right):
        rl, rr = find(left), find(right)
        if rl == rr:
            return
        # 以较小面号作根，保证结果确定
        if rl < rr:
            parent[rr] = rl
        else:
            parent[rl] = rr

    edge_faces = {}
    for face_index, (ia, ib, ic) in enumerate(triangles):
        for v0, v1 in ((ia, ib), (ib, ic), (ic, ia)):
            key = (v0, v1) if v0 < v1 else (v1, v0)
            bucket = edge_faces.get(key)
            if bucket is None:
                edge_faces[key] = [face_index]
            else:
                bucket.append(face_index)

    # 夹角 <= 阈值  <=>  cos(夹角) >= cos(阈值)
    cos_threshold = math.cos(math.radians(angle_deg))
    for sharing in edge_faces.values():
        if len(sharing) < 2:
            continue
        _merge_bucket(sharing, units, cos_threshold, union, find)

    components = {}
    for face_index in range(face_count):
        components.setdefault(find(face_index), []).append(face_index)
    groups = sorted(components.values(), key=lambda group: group[0])
    group_of = [0] * face_count
    for group_id, members in enumerate(groups):
        for face_index in members:
            group_of[face_index] = group_id
    return groups, group_of


def _accumulate(triangles, weighted, areas, group_of, vertex_count):
    """逐顶点、逐平滑组累加面积加权面法线与总面积。"""
    accumulators = [None] * vertex_count
    for face_index, (ia, ib, ic) in enumerate(triangles):
        group_id = group_of[face_index]
        nx, ny, nz = weighted[face_index]
        area = areas[face_index]
        for vertex in (ia, ib, ic):
            per_vertex = accumulators[vertex]
            if per_vertex is None:
                per_vertex = accumulators[vertex] = {}
            bucket = per_vertex.get(group_id)
            if bucket is None:
                per_vertex[group_id] = [nx, ny, nz, area]
            else:
                bucket[0] += nx
                bucket[1] += ny
                bucket[2] += nz
                bucket[3] += area
    return accumulators


def _normalize(vector):
    """归一化，零向量或非有限返回 None。"""
    x, y, z = vector
    length = math.hypot(x, y, z)
    if length <= 0.0 or not math.isfinite(length):
        return None
    return (x / length, y / length, z / length)


def _prepare(verts, faces, angle):
    """统一校验并计算全部中间量。"""
    points = _read_verts(verts)
    triangles = _read_faces(faces, len(points))
    angle_deg = _smoothing_angle(angle)
    weighted, areas, units = _face_geometry(points, triangles)
    groups, group_of = _group_faces(triangles, units, angle_deg)
    accumulators = _accumulate(
        triangles, weighted, areas, group_of, len(points)
    )
    return triangles, groups, group_of, accumulators


class Smooth:
    """顶点法线平滑组计算器（无状态、可并发调用）。"""

    def normals(self, verts, faces, angle):
        """返回每个顶点一个单位法线（列表长度等于 ``len(verts)``）。

        平滑组内按面积加权平均；顶点若分属多个平滑组，取面积总和最大的组。
        未被任何面引用的顶点返回默认单位法线 ``(0, 0, 1)``。
        """
        _triangles, _groups, _group_of, accumulators = _prepare(
            verts, faces, angle
        )
        result = []
        for per_vertex in accumulators:
            if not per_vertex:
                result.append((0.0, 0.0, 1.0))
                continue
            best_id = -1
            best_area = -1.0
            for group_id, bucket in per_vertex.items():
                area = bucket[3]
                if area > best_area or (
                    area == best_area and group_id < best_id
                ):
                    best_area = area
                    best_id = group_id
            normal = _normalize(per_vertex[best_id][:3])
            result.append(normal if normal is not None else (0.0, 0.0, 1.0))
        return result

    def corner_normals(self, verts, faces, angle):
        """返回与 ``faces`` 对齐的逐角单位法线。

        ``result[fi][k]`` 是第 fi 个面第 k 个角的法线；同一顶点若分属多个
        平滑组，不同组的面角会得到各自组的法线，从而正确表达硬边。
        """
        triangles, _groups, group_of, accumulators = _prepare(
            verts, faces, angle
        )
        result = []
        for face_index, (ia, ib, ic) in enumerate(triangles):
            group_id = group_of[face_index]
            corners = []
            for vertex in (ia, ib, ic):
                bucket = accumulators[vertex].get(group_id)
                normal = _normalize(bucket[:3]) if bucket is not None else None
                corners.append(normal if normal is not None else (0.0, 0.0, 1.0))
            result.append((corners[0], corners[1], corners[2]))
        return result

    def smoothing_groups(self, verts, faces, angle):
        """返回平滑组，每组是按面号升序排列的面号列表。"""
        _triangles, groups, _group_of, _accumulators = _prepare(
            verts, faces, angle
        )
        return [list(group) for group in groups]
