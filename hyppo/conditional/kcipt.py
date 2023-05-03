import numpy as np


def gaussian_kernel(X, Y, sigma=1.0):
    """
    Computes the Gaussian kernel between two matrices X and Y.
    """
    pairwise_dists = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
    K = np.exp(-pairwise_dists / (2 * sigma ** 2))
    return K


def PKCIT(X, Y, Z, n_perm=1000, alpha=0.05, sigma=1.0):
    """
    Permutation-Based Kernel Conditional Independence Test.

    Parameters
    ----------
    X, Y, Z : ndarray
        Input data matrices. ``X``, ``Y``, and ``Z`` must have the same number of
        rows. That is, the shapes must be ``(n, p)`` where
        `n` is the dimension of samples and `p` is the number of
        dimensions.
    n_perm : int, optional
        The number of permutations used to estimate the null distribution of the
        test statistic. Default is 1000.
    alpha : float, optional
        The level of the test. Default is 0.05.
    sigma : float, optional
        The bandwidth parameter of the Gaussian kernel. Default is 1.0.

    Returns
    -------
    stat : float
        The computed PKCIT statistic.
    pvalue : float
        The computed PKCIT p-value.
    """
    n = X.shape[0]
    perm_stats = np.zeros(n_perm)
    for i in range(n_perm):
        idx = np.random.permutation(n)
        X_perm = X[idx, :]
        Y_perm = Y[idx, :]
        Z_perm = Z[idx, :]
        K_XZ = gaussian_kernel(X_perm, Z_perm, sigma=sigma)
        K_YZ = gaussian_kernel(Y_perm, Z_perm, sigma=sigma)
        K_XYZ = K_XZ * K_YZ
        stat = np.sum(K_XYZ) / n
        perm_stats[i] = stat
    obs_K_XZ = gaussian_kernel(X, Z, sigma=sigma)
    obs_K_YZ = gaussian_kernel(Y, Z, sigma=sigma)
    obs_K_XYZ = obs_K_XZ * obs_K_YZ
    obs_stat = np.sum(obs_K_XYZ) / n
    pvalue = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)
    return obs_stat, pvalue
