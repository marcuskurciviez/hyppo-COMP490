import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def kcip_test(x, y, z, kernel='rbf', alpha=0.05, n_permutations=1000):
    """
    Permutation-Based Kernel Conditional Independence (KCIP) Test
    x: array-like, shape (n_samples, n_features)
    y: array-like, shape (n_samples, n_features)
    z: array-like, shape (n_samples, n_features)
    kernel: string or callable, kernel function
    alpha: float, significance level
    n_permutations: int, number of permutations
    Returns:
    p_value: float, p-value of the test
    """

    n = x.shape[0]
    indices = np.arange(n)

    # Compute the KCIP test statistic
    k_xz = pairwise_kernels(x, z, metric=kernel)
    k_yz = pairwise_kernels(y, z, metric=kernel)
    k_xy = pairwise_kernels(x, y, metric=kernel)
    k_xz_y = np.dot(k_xz.T, k_yz) / n
    k_xz_x = np.dot(k_xz.T, k_xz) / n
    k_yz_y = np.dot(k_yz.T, k_yz) / n
    k_xy_xz_y = np.dot(k_xy.T, k_xz_y) / n
    k_xx_xz = np.dot(k_xy.T, k_xz_x) / n
    k_yy_yz = np.dot(k_xy, k_yz_y) / n
    k_xx_xz_yy_yz = np.dot(k_xx_xz, k_yy_yz)
    k_xx_xz_yy_yz -= np.diag(np.diag(k_xx_xz_yy_yz))
    c_statistic = np.sum(k_xx_xz_yy_yz) / (n * (n - 1))

    # Permutation-based test
    c_null = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(indices)
        k_xz_perm = pairwise_kernels(x[indices], z, metric=kernel)
        k_yz_perm = pairwise_kernels(y[indices], z, metric=kernel)
        k_xy_perm = pairwise_kernels(x[indices], y, metric=kernel)
        k_xz_y_perm = np.dot(k_xz_perm.T, k_yz_perm) / n
        k_xz_x_perm = np.dot(k_xz_perm.T, k_xz_perm) / n
        k_yz_y_perm = np.dot(k_yz_perm.T, k_yz_perm) / n
        k_xy_xz_y_perm = np.dot(k_xy_perm.T, k_xz_y_perm) / n
        k_xx_xz_perm = np.dot(k_xy_perm.T, k_xz_x_perm) / n
        k_yy_yz_perm = np.dot(k_xy_perm, k_yz_y_perm) / n
        k_xx_xz_yy_yz_perm = np.dot(k_xx_xz_perm, k_yy_yz_perm)
        k_xx_xz_yy_yz_perm -= np.diag(np.diag(k_xx_xz_yy_yz_perm))
        c_null[i] = np.sum(k_xx_xz_yy_yz_perm) / (n * (n - 1))

    p_value = np.mean(c_null >= c_statistic)
    return p_value
