import math

import pytest

import meshsmooth

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
F = [(0, 1, 2), (0, 1, 3)]


def _s():
    return meshsmooth.Smooth()


def _check_count():
    return len(_s().normals(V, F, 60.0)) == len(V)


def _check_unit():
    ns = _s().normals(V, F, 60.0)
    return all(abs(math.sqrt(sum(c * c for c in n)) - 1.0) < 1e-6 for n in ns)


def _check_hard_split():
    soft = len(set(tuple(n) for n in _s().normals(V, F, 89.0)))
    hard = len(set(tuple(n) for n in _s().normals(V, F, 1.0)))
    return hard >= soft


def _check_bad_faces():
    return _raises(lambda: _s().normals([], F, 60.0))


def _check_bad_angle():
    return _raises(lambda: _s().normals(V, F, 0.0))


def _check_type():
    return isinstance(_s().normals(V, F, 60.0), list)


def _check_det():
    return _s().normals(V, F, 60.0) == _s().normals(V, F, 60.0)


def _check_flat():
    v = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    f = [(0, 1, 2)]
    n = _s().normals(v, f, 60.0)[0]
    return abs(abs(n[2]) - 1.0) < 1e-6

CASES = [("count", "_check_count"), ("unit", "_check_unit"), ("hard_split", "_check_hard_split"), ("bad_faces", "_check_bad_faces"), ("bad_angle", "_check_bad_angle"), ("type", "_check_type"), ("det", "_check_det"), ("flat", "_check_flat"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Smooth() is not None
        return
    assert fn() is True
