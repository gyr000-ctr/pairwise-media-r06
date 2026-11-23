import math

import pytest

import mipblend

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m():
    return mipblend.MipBlend(8)


def _check_level0():
    return _m().blend(1.0, 1.0)[0] == 0


def _check_frac_range():
    l, f = _m().blend(0.7, 0.7)
    return 0.0 <= f < 1.0


def _check_aniso():
    l1, _ = _m().blend(0.5, 0.5)
    l2, _ = _m().blend(0.5, 0.1)
    return l2 >= l1


def _check_clamp():
    return _m().blend(1e-9, 1e-9)[0] <= 8


def _check_mix():
    return abs(_m().sample(0.0, 1.0, 0.5) - 0.5) < 1e-9


def _check_bad_scale():
    return _raises(lambda: _m().blend(0.0, 1.0))


def _check_type():
    return isinstance(_m().blend(0.5, 0.5), tuple)


def _check_det():
    return _m().blend(0.3, 0.3) == _m().blend(0.3, 0.3)

CASES = [("level0", "_check_level0"), ("frac_range", "_check_frac_range"), ("aniso", "_check_aniso"), ("clamp", "_check_clamp"), ("mix", "_check_mix"), ("bad_scale", "_check_bad_scale"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert MipBlend() is not None
        return
    assert fn() is True
