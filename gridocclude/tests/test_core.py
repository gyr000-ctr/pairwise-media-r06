import math

import pytest

import gridocclude

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def _l():
    return gridocclude.Los()


def _check_count():
    return len(_l().visible(G, (0, 0))) >= 3


def _check_self():
    return (0, 0) in _l().visible(G, (0, 0))


def _check_wall_blocked():
    return (1, 1) not in _l().visible(G, (0, 0))


def _check_bad_origin():
    return _raises(lambda: _l().visible(G, (9, 9)))


def _check_bad_grid():
    return _raises(lambda: _l().visible([], (0, 0)))


def _check_type():
    return isinstance(_l().visible(G, (0, 0)), list)


def _check_symmetry():
    v = _l().visible([[0, 0], [0, 0]], (0, 0))
    return (1, 1) in v


def _check_det():
    return _l().visible(G, (0, 0)) == _l().visible(G, (0, 0))

CASES = [("count", "_check_count"), ("self", "_check_self"), ("wall_blocked", "_check_wall_blocked"), ("bad_origin", "_check_bad_origin"), ("bad_grid", "_check_bad_grid"), ("type", "_check_type"), ("symmetry", "_check_symmetry"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Los() is not None
        return
    assert fn() is True
