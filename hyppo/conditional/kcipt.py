import numpy as np
from hyppo.conditional import KCI
from hyppo.tools import perm_test


def PKCIT(x, y, n_perm=1000, alpha=0.05):
    """
    Permutation-Based Kernel Conditional Independence Test.
    Parameters
    ----------
    x, y : ndarray
        Input data matrices. ``x`` and ``y`` must have the same number of
        columns. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
        `n` is the dimension of samples and `p` is the number of
        dimensions.
    n_perm : int, optional
        The number of permutations used to estimate the null distribution of the
        test statistic. Default is 1000.
    alpha : float, optional
        The level of the test. Default is 0.05.

    Returns
    -------
    stat : float
        The computed PKCIT statistic.
    pvalue : float
        The computed PKCIT p-value.
    """
    kcit = KCI()
    obs_stat = kcit.statistic(x, y)
    null_dist = perm_test(kcit, x, y, n_perm=n_perm)
    pvalue = (null_dist >= obs_stat).sum() / n_perm
    return obs_stat, pvalue
