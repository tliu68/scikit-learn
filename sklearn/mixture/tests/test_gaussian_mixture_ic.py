"""Testing for GaussianMixtureIC"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_rand_score

from sklearn.mixture import GaussianMixtureIC


def _test_inputs(X, error_type, **kws):
    with pytest.raises(error_type):
        gmIC = GaussianMixtureIC(**kws)
        gmIC.fit(X)


def test_n_components():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    # min_components < 1
    _test_inputs(X, ValueError, min_components=0)

    # min_components integer
    _test_inputs(X, TypeError, min_components="1")

    # max_components < min_components
    _test_inputs(X, ValueError, max_components=0)

    # # max_components integer
    _test_inputs(X, TypeError, max_components="1")

    # min_cluster > n_samples when max_cluster is None
    with pytest.raises(ValueError):
        gmIC = GaussianMixtureIC(1000)
        gmIC.fit(X)

    # max_cluster > n_samples when max_cluster is not None
    _test_inputs(
        X, ValueError, **{"min_components": 10, "max_components": 101}
    )

    # max_cluster > n_samples when max_cluster is None
    with pytest.raises(ValueError):
        gmIC = GaussianMixtureIC(1000)
        gmIC.fit(X)

    with pytest.raises(ValueError):
        gmIC = GaussianMixtureIC(10, 1001)
        gmIC.fit_predict(X)

    # min_cluster > n_samples when max_cluster is not None
    with pytest.raises(ValueError):
        gmIC = GaussianMixtureIC(1000, 1001)
        gmIC.fit(X)

    with pytest.raises(ValueError):
        gmIC = GaussianMixtureIC(1000, 1001)
        gmIC.fit_predict(X)


def test_input_param():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    # affinity is not an array, string or list
    _test_inputs(X, TypeError, affinity=1)

    # affinity is not in ['euclidean', 'manhattan', 'cosine', 'none']
    _test_inputs(X, ValueError, affinity="1")

    # linkage is not an array, string or list
    _test_inputs(X, TypeError, linkage=1)

    # linkage is not in ['single', 'average', 'complete', 'ward']
    _test_inputs(X, ValueError, linkage="1")

    # covariance type is not an array, string or list
    _test_inputs(X, TypeError, covariance_type=1)

    # covariance type is not in ['spherical', 'diag', 'tied', 'full']
    _test_inputs(X, ValueError, covariance_type="1")

    # euclidean is not an affinity option when ward is a linkage option
    _test_inputs(X, ValueError, **{"affinity": "manhattan", "linkage": "ward"})

    # criter = cic
    _test_inputs(X, ValueError, selection_criteria="cic")


def test_labels_init():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    # label_init is not a 1-D array
    _test_inputs(X, TypeError, label_init=np.zeros([100, 2]))

    # label_init is not 1-D array, a list or None.
    _test_inputs(X, TypeError, label_init="label")

    # label_init length is not equal to n_samples
    _test_inputs(X, ValueError, label_init=np.zeros([50, 1]))

    # label_init length does not match min_components and max_components
    _test_inputs(
        X,
        ValueError,
        **{
            "label_init": np.zeros([100, 1]),
            "min_components": 2,
            "max_components": 3,
        },
    )


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(NotFittedError):
        gmIC = GaussianMixtureIC(min_components=2)
        gmIC.predict(X)


def test_cosine_with_0():
    X = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
        ]
    )

    with pytest.warns(UserWarning):
        gmIC = GaussianMixtureIC(min_components=2, affinity="all")
        gmIC.fit(X)


def _test_two_class(AIC=False, **kws):
    """
    Easily separable two gaussian problem.
    """
    np.random.seed(1)

    n = 100
    d = 3

    X1 = np.random.normal(2, 0.5, size=(n, d))
    X2 = np.random.normal(-2, 0.5, size=(n, d))
    X = np.vstack((X1, X2))
    y = np.repeat([0, 1], n)

    gmIC = GaussianMixtureIC(max_components=5, **kws)
    gmIC.fit(X, y)

    n_components = gmIC.n_components_

    if AIC is False:
        # Assert that the two cluster model is the best
        assert_equal(n_components, 2)

        # Asser that we get perfect clustering
        ari = adjusted_rand_score(y, gmIC.fit_predict(X))
        assert_allclose(ari, 1)
    else:
        # AIC gets the number of components wrong
        assert_equal(n_components >= 1, True)
        assert_equal(n_components <= 5, True)


def test_two_class():
    _test_two_class(AIC=False)


def test_two_class_parallel():
    _test_two_class(n_jobs=2)


def test_two_class_aic():
    _test_two_class(AIC=True)


def _test_five_class(AIC=False, **kws):
    """
    Easily separable five gaussian problem.
    """
    np.random.seed(1)

    n = 100
    mus = [[i * 5, 0] for i in range(5)]
    cov = np.eye(2)  # balls

    X = np.vstack([np.random.multivariate_normal(mu, cov, n) for mu in mus])

    gmIC = GaussianMixtureIC(
        min_components=3, max_components=10, covariance_type="all", **kws
    )
    gmIC.fit(X)

    if AIC is False:
        assert_equal(gmIC.n_components_, 5)
    else:
        # AIC fails often so there is no assertion here
        assert_equal(gmIC.n_components_ >= 3, True)
        assert_equal(gmIC.n_components_ <= 10, True)


def test_five_class():
    _test_five_class(AIC=False)


def test_five_class_aic():
    _test_five_class(AIC=True)


def test_covariances():
    """
    Easily separable two gaussian problem.
    """
    np.random.seed(1)

    n = 100
    mu1 = [-10, 0]
    mu2 = [10, 0]

    # Spherical
    cov1 = 2 * np.eye(2)
    cov2 = 2 * np.eye(2)

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gmIC = GaussianMixtureIC(min_components=2, covariance_type="all")
    gmIC.fit(X)
    assert_equal(gmIC.covariance_type_, "spherical")

    # Diagonal
    np.random.seed(10)
    cov1 = np.diag([1, 1])
    cov2 = np.diag([2, 1])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gmIC = GaussianMixtureIC(max_components=2, covariance_type="all")
    gmIC.fit(X)
    assert_equal(gmIC.covariance_type_, "diag")

    # Tied
    cov1 = np.array([[2, 1], [1, 2]])
    cov2 = np.array([[2, 1], [1, 2]])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gmIC = GaussianMixtureIC(max_components=2, covariance_type="all")
    gmIC.fit(X)
    assert_equal(gmIC.covariance_type_, "tied")

    # Full
    cov1 = np.array([[2, -1], [-1, 2]])
    cov2 = np.array([[2, 1], [1, 2]])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gmIC = GaussianMixtureIC(max_components=2, covariance_type="all")
    gmIC.fit(X)
    assert_equal(gmIC.covariance_type_, "full")
