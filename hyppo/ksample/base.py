from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import percentileofscore
from sklearn.metrics import pairwise_distances, pairwise_kernels

from examples.permutation_tree import X, Y

from tutorials.ksample import y, x


class KSampleTestOutput(NamedTuple):
    stat: float
    pvalue: float


class KSampleTest(ABC):
    """
    A base class for a *k*-sample test.

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean" or "gaussian"
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``compute_distance`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``] See the documentation for
              :mod:`scipy.spatial.distance` for details
              on these metrics.
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Alternatively, this function computes the kernel similarity among the
        samples within each data matrix.
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
    bias : bool (default: False)
        Whether or not to use the biased or unbiased test statistics. Only
        applies to ``Dcorr`` and ``Hsic``.
    **kwargs
        Arbitrary keyword arguments for ``compute_distkern``.
    """
    def _permute(self, X, Y, n_perm, block_size, compute_distance=None, **kwargs):
        n = X.shape[0]
        m = Y.shape[0]
        xy = np.vstack([X, Y])
        indices = np.arange(n + m)
        shuffle_indices = np.random.permutation(indices)
        shuffle_indices_X = shuffle_indices[:n]
        shuffle_indices_Y = shuffle_indices[n:]

        stat_perm = []
        for i in range(n_perm):
            block_indices_X = np.split(shuffle_indices_X, range(block_size, n, block_size))
            block_indices_Y = np.split(shuffle_indices_Y, range(block_size, m, block_size))

            # within-block permutation of X and Y
            for j in range(len(block_indices_X)):
                block_indices_X[j] = np.random.permutation(block_indices_X[j])
            for j in range(len(block_indices_Y)):
                block_indices_Y[j] = np.random.permutation(block_indices_Y[j])

            # reconstruct the permuted matrix
            permuted_indices = np.hstack(block_indices_X + block_indices_Y)
            permuted_xy = xy[permuted_indices]

            # compute test statistic of the permuted matrix
            stat_perm.append(self.statistic(*np.split(permuted_xy, [n])))

        # calculate p-value
        self.pvalue = (percentileofscore(stat_perm, self.stat) / 100.0)
    def __init__(self, compute_distance=None, bias=False, **kwargs):
        self.compute_distance = compute_distance
        self.bias = bias
        self.kwargs = kwargs

    def _block_permutation(self, X, block_size):
        n = X.shape[0]
        n_blocks = n // block_size
        permuted_X = np.zeros_like(X)

        for i in range(n_blocks):
            block_start = i * block_size
            block_end = (i + 1) * block_size
            block = X[block_start:block_end]
            permuted_X[block_start:block_end] = np.random.permutation(block)

        return permuted_X

    def _compute_distance(self, X, Y=None):
        if callable(self.compute_distance):
            return self.compute_distance(X, Y, **self.kwargs)
        elif self.compute_distance in pairwise_distances.VALID_METRICS:
            return pairwise_distances(X, Y, metric=self.compute_distance, **self.kwargs)
        elif self.compute_distance in pairwise_kernels.VALID_KERNELS:
            return pairwise_kernels(X, Y, metric=self.compute_distance, **self.kwargs)
        else:
            raise ValueError("Invalid distance metric specified.")
    @abstractmethod
    def statistic(self, *args):
        r"""
        Calulates the *k*-sample test statistic.

        Parameters
        ----------
        *args : ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.

        Returns
        -------
        stat : float
            The computed *k*-Sample statistic.
        """

    @abstractmethod
    def test(self, *args, reps=1000, workers=1, random_state=None, block_size=None):
        r"""
        Calculates the *k*-sample test statistic and p-value.

        Parameters
        ----------
        *args : ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed *k*-sample statistic.
        pvalue : float
            The computed *k*-sample p-value.
        """
        if block_size is not None:
            # perform block permutation
            permute_args = (X, Y, reps, block_size, self.compute_distance) + self.kwargs.items()
            stat_perm = Parallel(n_jobs=workers, verbose=0)(
                delayed(self._permute)(*permute_args, random_state=random_state + i)
                for i in range(reps))
            self.pvalue = np.mean(stat_perm >= self.stat)
        else:
            # perform regular permutation
            perms = np.array([
                np.random.permutation(np.vstack([x, y]))
                for _ in range(reps)])
            pstat = np.array([
                self.statistic(*np.split(perm, [x.shape[0]]))
                for perm in perms])
            self.stat = self.statistic(*args)
            self.pvalue = np.mean(pstat >= self.stat)

        return KSampleTestOutput(stat=self.stat, pvalue=self.pvalue)

