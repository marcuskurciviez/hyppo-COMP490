import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ...tools import power, rot_ksamp
from .. import MANOVA


class TestManova:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(1000, 0.005062841807278008, 1.0), (100, 8.24e-5, 0.9762956529114515)],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        stat, pvalue = MANOVA().test(x, y)
        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)

    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(1000, 0.005062841807278008, 1.0), (100, 8.24e-5, 0.9762956529114515)],
    )
    def test_nonlinear_oned(self,n,obs_stat,obs_pvalue):
       np.random.seed(987654321)
       x,y=rot_ksamp("nonlinear",n,0,k=5, noise=False)
       stat, pvalue=MANOVA().test(x,y)
       assert assert_almost_equal(stat,obs_stat, decimal=0)
       assert assert_almost_equal(pvalue,obs_pvalue, decimal=0)

    def test_mismatched_dimensions(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 100, 1, k=2, noise=False)
        z = np.random.randn(50, 1)
        assert_raises(ValueError, MANOVA().test, x, y, z)


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
