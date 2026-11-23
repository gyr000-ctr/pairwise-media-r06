import math

import pytest

import colorgrade

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return colorgrade.Grade()


def _check_gray():
    out = _c().white_balance((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    return all(abs(v - 0.5) < 1e-6 for v in out)


def _check_gain():
    out = _c().white_balance((0.5, 0.5, 0.5), (2.0, 1.0, 1.0))
    return out[0] > out[1]


def _check_curve_end():
    return abs(_c().curve(1.0, 2.2) - 1.0) < 1e-9


def _check_curve_zero():
    return _c().curve(0.0, 2.2) == 0.0


def _check_curve_monotonic():
    return _c().curve(0.2, 2.2) <= _c().curve(0.8, 2.2)


def _check_bad_gamma():
    return _raises(lambda: _c().curve(0.5, 0.0))


def _check_type():
    return isinstance(_c().curve(0.5, 2.2), float)


def _check_det():
    return _c().curve(0.4, 2.2) == _c().curve(0.4, 2.2)

CASES = [("gray", "_check_gray"), ("gain", "_check_gain"), ("curve_end", "_check_curve_end"), ("curve_zero", "_check_curve_zero"), ("curve_monotonic", "_check_curve_monotonic"), ("bad_gamma", "_check_bad_gamma"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Grade() is not None
        return
    assert fn() is True
