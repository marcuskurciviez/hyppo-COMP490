from functools import reduce
from operator import concat

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ..base import KSampleTest
from hyppo.tools import power, rot_ksamp
from .. import KSample
from kfda import Kfda
import unittest
from ..base import KSampleTest


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
        NewArray = [5], [5]

        x, y = rot_ksamp("linear", 10, 5, k=2)
        stat, pvalue = KSample("CCA").test(x, y)
        for z in range(5, 5):
            for a in range(5, 5):
                NewArray[z] = stat
                NewArray[a] = pvalue
        cls = Kfda(n_components=2, kernel='linear')
        cls.fit(NewArray, [2, 1])
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

@pytest.fixture
def ksample_test():
    return KSampleTest()

def test_block_permutation(ksample_test):
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
    block_size = 2
    permuted_X = ksample_test._block_permutation(X, block_size)

    # Test that each block of the permuted_X is a permutation of the corresponding block in X
    for i in range(0, X.shape[0], block_size):
        assert sorted(X[i:i+block_size].flatten()) == sorted(permuted_X[i:i+block_size].flatten())

    # Test that the total number of elements is the same
    assert X.size == permuted_X.size

