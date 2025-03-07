import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ...tools import power, rot_ksamp, nonlinear
from .. import MANOVA


class TestManova:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(1000, 0.005062841807278008, 1.0), (100, 8.24e-5, 0.9762956529114515)],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        """Checks performance of the MANOVA test on linear data in 1D"""
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        stat, pvalue = MANOVA().test(x, y)
        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)

    def test_mismatched_dimensions(self):
        """Checks that an error is raised if the dimensions of the input data are mismatched"""
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 100, 1, k=2, noise=False)
        z = np.random.randn(50, 1)
        assert_raises(ValueError, MANOVA().test, x, y, z)

    def test_nonlinear_oned(self):
        """Checks performance of the MANOVA test on nonlinear data in 1D"""
        np.random.seed(123456789)
        x, y = nonlinear(100, 1)
        stat, pvalue = MANOVA().test(x, y)
        assert pvalue > 0.05

    def test_zero_variance(self):
        """Checks that an error is raised if the input data has zero variance"""
        np.random.seed(123456789)
        x, y = np.zeros((100, 2)), np.zeros((100, 1))
        assert_raises(ValueError, MANOVA().test, x, y)


class TestManovaErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_no_indeptest(self):
        # raises error if not indep test
        x = np.arange(100).reshape(5, 20)
        y = np.arange(50, 150).reshape(5, 20)
        assert_raises(ValueError, MANOVA().test, x, y)



class TestManovaTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "MANOVA",
            sim_type="gauss",
            sim="multimodal_independence",
            case=1,
            n=100,
            alpha=0.05,
        )

        assert est_power <= 0.05
