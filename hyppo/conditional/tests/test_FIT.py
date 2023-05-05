import numpy
from hyppo.conditional.FIT import interleave, cv_besttree, fit_test

# Test the interleave function
def test_interleave():
    x = numpy.array([[1, 2], [3, 4]])
    z = numpy.array([[5, 6], [7, 8]])
    result = interleave(x, z)
    assert result.shape == (2, 4), "Shape of interleaved array is incorrect."
    assert numpy.all(numpy.sort(result, axis=1) == numpy.array([[1, 2, 5, 6], [3, 4, 7, 8]])), "Interleaving is incorrect."

# Test the cv_besttree function
def test_cv_besttree():
    x = numpy.random.rand(50, 2)
    y = numpy.random.rand(50, 1)
    z = numpy.random.rand(50, 2)
    cv_grid = [2, 8]
    logdim = False
    verbose = False
    prop_test = 0.1
    clf = cv_besttree(x, y, z, cv_grid, logdim, verbose, prop_test)
    assert clf, "cv_besttree should return a DecisionTreeRegressor."

