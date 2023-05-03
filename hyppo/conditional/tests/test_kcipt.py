import numpy as np
from hyppo.conditional.kcipt import PKCIT

def test_PKCIT():
    np.random.seed(42)
    n = 100
    p = 3
    X = np.random.normal(size=(n, p))
    Y = np.random.normal(size=(n, p))
    Z = np.random.normal(size=(n, p))
    obs_stat, pvalue = PKCIT(X, Y, Z, n_perm=100, alpha=0.05, sigma=1.0)
    assert isinstance(obs_stat, float)
    assert isinstance(pvalue, float)
    assert obs_stat >= 0.0
    assert pvalue >= 0.0 and pvalue <= 1.0
