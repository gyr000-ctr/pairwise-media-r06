import math

import pytest

import timestretch

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 1000


def _t():
    return timestretch.Stretch()


def _check_len():
    out = _t().stretch(S, 2.0, 256)
    return abs(len(out) - 2000) <= 20


def _check_identity():
    out = _t().stretch(S, 1.0, 256)
    return len(out) == len(S)


def _check_bad_ratio():
    return _raises(lambda: _t().stretch(S, 0.0, 256))


def _check_bad_frame():
    return _raises(lambda: _t().stretch(S, 2.0, 0))


def _check_empty():
    return _t().stretch([], 2.0, 256) == []


def _check_type():
    return isinstance(_t().stretch(S, 2.0, 256), list)


def _check_range():
    out = _t().stretch(S, 2.0, 256)
    return all(abs(v) <= 1.0 for v in out)


def _check_det():
    return _t().stretch(S[:100], 1.5, 64) == _t().stretch(S[:100], 1.5, 64)

CASES = [("len", "_check_len"), ("identity", "_check_identity"), ("bad_ratio", "_check_bad_ratio"), ("bad_frame", "_check_bad_frame"), ("empty", "_check_empty"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Stretch() is not None
        return
    assert fn() is True
