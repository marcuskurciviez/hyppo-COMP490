import numpy as np
from hyppo.conditional import KCI
from hyppo.tools import perm_test
from hyppo.conditional import kcipt


def test_PKCIT():
    # Test with two independent variables
    x = np.random.rand(100, 3)
    y = np.random.rand(100, 1)
    stat, pvalue = kcipt.PKCIT(x, y)
    assert np.isfinite(stat)
    assert np.isfinite(pvalue)
    assert 0 <= pvalue <= 1

    # Test with two dependent variables
    x = np.random.rand(100, 3)
    y = x[:, [0]]
    stat, pvalue = kcipt.PKCIT(x, y)
    assert np.isfinite(stat)
    assert np.isfinite(pvalue)
    assert pvalue > 0.05

    # Test with different number of dimensions
    x = np.random.rand(100, 2)
    y = np.random.rand(100, 1)
    try:
        kcipt.PKCIT(x, y)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
