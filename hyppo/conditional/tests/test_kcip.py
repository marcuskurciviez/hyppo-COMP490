# import numpy as np
# from numpy.testing import assert_almost_equal
# from scipy.stats import norm
# from hyppo.conditional.kcip import kcip_test
#
# def test_kcip_test():
#     # Generate random data
#     n = 100
#     d = 2
#     rng = np.random.RandomState(0)
#     x = rng.randn(n, d)
#     y = rng.randn(n, d)
#     z = rng.randn(n, d)
#
#     # Test with null hypothesis (independent variables)
#     p_value = kcip_test(x, y, z)
#     assert p_value > 0.05
#
#     # Test with alternative hypothesis (dependent variables)
#     y = x + z
#     p_value = kcip_test(x, y, z)
#     assert p_value < 0.05
#
#     # Test with known p-value
#     y = x + z
#     p_value = kcip_test(x, y, z, n_permutations=10000)
#     p_value_true = norm.sf(2.58)
#     assert_almost_equal(p_value, p_value_true, decimal=2)
