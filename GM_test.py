import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.utils._testing import create_memmap_backed_data
from sklearn.utils._testing import set_random_state, assert_array_equal
from sklearn.mixture import GaussianMixtureIC

clusterer = GaussianMixtureIC(max_iter=5, affinity="euclidean",
    covariance_type="tied", min_components=3, max_components=3, linkage=['ward', 'complete'])
X, y = make_blobs(n_samples=50, random_state=1)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)
rng = np.random.RandomState(7)
X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])
X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

set_random_state(clusterer)
# fit
clusterer.fit(X)
# with lists
clusterer.fit(X.tolist())

pred = clusterer.labels_
# check all BIC values
print([result.criterion for result in clusterer.results_])
# check the linkage parameter of the AgglomerativeClustering initialization
# of the best model
print(clusterer.linkage_)
print(pred)

set_random_state(clusterer)
pred2 = clusterer.fit_predict(X)
print([result.criterion for result in clusterer.results_])
print(clusterer.linkage_)
print(pred2)
assert_array_equal(pred, pred2)