from functools import reduce
from operator import concat

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ..base import KSampleTest
from hyppo.tools import power, rot_ksamp
from .. import KSample
from kfda import Kfda
import numpy as np
from hyppo.independence import Dcorr
from hyppo.ksample.base import KSampleTest


class TestKSample:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue, indep_test",
        [(1000, 4.28e-7, 1.0, "CCA"), (100, 8.24e-5, 0.001, "Dcorr")],
    )
    def test_twosamp_linear_oned(self, n, obs_stat, obs_pvalue, indep_test):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = KSample(indep_test).test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)

    def test_twosamp_kdf(self):
        np.random.seed(123456789)
        NewArray = [5],[5]

        x, y = rot_ksamp("linear", 10, 5, k=2)
        stat, pvalue = KSample("CCA").test(x, y)
        for z in range(5,5):
         for a in range(5,5):

            NewArray[z]=stat
            NewArray[a]=pvalue
        cls = Kfda(n_components=2, kernel='linear')
        cls.fit(NewArray,[2,1])
        cls.predict(NewArray)
        assert cls.predict(NewArray).any()

    def test_rf(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        stat, _, _ = KSample("KMERF").test(x, y, reps=0)

        assert_almost_equal(stat, 0.2714, decimal=1)

    def test_maxmargin(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        stat, _ = KSample(["MaxMargin", "Dcorr"]).test(x, y, reps=0)

        assert_almost_equal(stat, 0.0317, decimal=1)

    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue, indep_test",
        [(100, 8.24e-5, 0.001, "Dcorr")],
    )
    def test_rep(self, n, obs_stat, obs_pvalue, indep_test):
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = KSample(indep_test).test(x, y)
        stat2, pvalue2 = KSample(indep_test).test(x, y)

        assert stat == stat2
        assert pvalue == pvalue2


class TestKSampleErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_no_indeptest(self):
        # raises error if not indep test
        assert_raises(ValueError, KSample, "abcd")

    def test_no_second_maxmargin(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        assert_raises(ValueError, KSample, ["MaxMargin", "abcd"])


class TestKSampleTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "CCA",
            sim_type="ksamp",
            sim="multimodal_independence",
            k=2,
            n=100,
            p=1,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)


class DcorrKSampleTest(KSampleTest):
    def __init__(self, compute_distance='euclidean', **kwargs):
        super().__init__(compute_distance=compute_distance, **kwargs)

    def test(self, *args, reps=1000, workers=1, random_state=None, block_size=None):
        if block_size:
            args = self._block_permutation(args, block_size)
        return super().test(*args, reps=reps, workers=workers, random_state=random_state)

    def _block_permutation(self, arrays, block_size):
        arrays = list(arrays)
        for i, arr in enumerate(arrays):
            num_blocks = len(arr) // block_size
            permuted_blocks = np.random.permutation(np.split(arr, num_blocks))
            arrays[i] = np.concatenate(permuted_blocks)
        return tuple(arrays)

    def statistic(self, *args):
        dcorr = Dcorr(compute_distance=self.compute_distance)
        return dcorr.statistic(*args)

class TestKSampleBlockPerm:
    def test_block_permutation_stat(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        kstest = DcorrKSampleTest(compute_distance="euclidean")
        stat, pvalue = kstest.test(x, y, reps=100, workers=1, random_state=None, block_size=10)


    def test_block_permutation_pvalue(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        kstest = DcorrKSampleTest(compute_distance="euclidean")
        stat, pvalue = kstest.test(x, y, reps=100, workers=1, random_state=None, block_size=10)
        # test the statistic value after block permutation
        assert_almost_equal(stat, 0.0317, decimal=1)
