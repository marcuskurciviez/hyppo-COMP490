from operator import concat
from random import random
from hyppo.conditional.FIT import cross_validate
from numpy import floor
from numpy import np
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pytest
import numpy as np
from hyppo.conditional.FIT import interleave, cv_besttree

# Test the interleave function
def test_interleave():
    x = np.array([[1, 2], [3, 4]])
    z = np.array([[5, 6], [7, 8]])
    result = interleave(x, z)
    assert result.shape == (2, 4), "Shape of interleaved array is incorrect."
    assert np.all(np.sort(result, axis=1) == np.array([[1, 2, 5, 6], [3, 4, 7, 8]])), "Interleaving is incorrect."

# Test the cv_besttree function
def test_cv_besttree():
    x = np.random.rand(50, 2)
    y = np.random.rand(50, 1)
    z = np.random.rand(50, 2)
    cv_grid = [2, 8]
    logdim = False
    verbose = False
    prop_test = 0.1
    clf = cv_besttree(x, y, z, cv_grid, logdim, verbose, prop_test)
    assert clf, "cv_besttree should return a DecisionTreeRegressor."

def test_FIT_TEST(x , y , z , n_perm =8 , frac_test=.1):
    x = np.random.rand(50, 2)
    y = np.random.rand(50, 1)
    z = np.random.rand(50, 2)
    best_tree_x=cross_validate(x,y,z)
    best_tree_x.fit(np.concatenate((x,z), axis=1), y)
    mses_x = list()
    mses_x.append(
        mses_x(best_tree_x.predict(np.concatenate((x, z), axis=1)), y))
    mses_x.append(mses_x(best_tree_x.predict(z), y))
    assert best_tree_x
