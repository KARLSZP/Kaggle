from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import pickle as pk
import pandas as pd
import numpy as np
import random as rd
import time
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier

N_JOBS = 4


class RandomForest:
    """A RandomForest classifier.

    Parameters
    ----------
    n_estimators : int, optional (default=10)
        The number of estimators in the forest.

    max_depth : int or None, optional (default=None)
        The maximum depth of each tree. 
        If None, then nodes are expanded until all leaves are pure
        or until all leaves contain less than `min_leaf_size` samples.

    verbose : int, optional (default=0)
        The level of debugging message in parallel jobs.
        The higher the number is, more detailed messages are printed.

    min_leaf_size : int, optional (default=2)
        The minimum number of samples required to split an internal(non-leaf) node.

    n_features : int, string or None, optional (default=sqrt)
        The number of features to consider when looking for the best split.

            - If int, then consider `n_features` features at each split, randomly.
            - If "auto", then `n_features=sqrt(n_features)`.
            - If "sqrt", then `n_features=sqrt(n_features)`.
            - If "log2", then `n_features=log2(n_features)`.
            - If None, then `n_features=n_features`.

    n_samples : float, optional (default=0.67)
        The proportion of the number of the rows bootstraped during sampling.

    Attributes
    ----------
    n_estimators : int
        The number of estimators in the forest.

    max_depth : int or None
        The maximum depth of the tree.

    verbose : int
        The level of debugging message in parallel jobs.
        The higher the number is, more detailed messages are printed.

    min_leaf_size : int
        The minimum number of samples required to split an internal(non-leaf) node.

    n_features : 
        The number of features to consider when looking for the best split.

    estimators : list of `DecisionTreeClassifier`
        The list of trees.

    Examples
    --------
    >>> from RF import RandomForest

    >>> train_data = ...

    >>> clf = RandomForest()

    >>> clf.fit(train_data)

    >>> test_sample = ...

    >>> print(clf.predict(test_sample))
    [1]

    References
    ----------

    .. [1] sklearn.ensemble forest.py
    .. [2] sklearn.tree tree.py

    Copyright
    ---------
    KarlSzp
    """

    def __init__(self, n_estimators=10, verbose=0,
                 max_depth=None, min_leaf_size=2,
                 n_samples=0.67, n_features="sqrt"):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_features = n_features
        self.n_samples = n_samples
        self.verbose = verbose
        self.estimators = []

    def __bootstrap(self, dataset):
        sampled, unsampled = train_test_split(
            dataset, train_size=self.n_samples, shuffle=True, stratify=dataset[:, -1])
        return sampled, unsampled

    def __buildEstimator(self, sampled):
        clf = DecisionTreeClassifier(
            max_depth=self.max_depth, min_samples_leaf=self.min_leaf_size, max_features=self.n_features, random_state=None)
        clf.fit(sampled[:, :-1], sampled[:, -1])
        return clf

    def fit(self, dataset):
        bootstraped_datasets = [self.__bootstrap(
            dataset) for i in range(self.n_estimators)]
        self.estimators = Parallel(n_jobs=N_JOBS, verbose=self.verbose, prefer="threads")(
            delayed(self.__buildEstimator)(i[0]) for i in bootstraped_datasets)
        return

    def __predict(self, est, case):
        return est.predict(case)[0]

    def predict(self, case):
        predictions = [self.__predict(est, case) for est in self.estimators]
        # predictions = Parallel(n_jobs=N_JOBS, batch_size=25, backend="threading")(
        #     delayed(self.__predict)(est, case) for est in self.estimators)
        return max(set(predictions), key=predictions.count)


def _predictWithComb(dataset, combination):
    n_estimators, max_depth, min_leaf_size = combination

    clf = RandomForest(n_estimators=n_estimators,
                       max_depth=max_depth, min_leaf_size=min_leaf_size)
    clf.fit(dataset)

    acr = 0.0
    for i in dataset:
        if i[-1] == clf.predict(i[:-1].reshape(1, -1)):
            acr += 1

    # print("ACC:", acr / dataset.shape[0], " in ", et-st, "s.")
    return acr / dataset.shape[0], combination


def cross_validation(dataset, para_dict):
    _n_estimators = []
    _max_depth = []
    _min_leaf_size = []
    if "n_estimators" in para_dict.keys():
        _n_estimators = para_dict["n_estimators"]
    if "max_depth" in para_dict.keys():
        _max_depth = para_dict["max_depth"]
    if "min_leaf_size" in para_dict.keys():
        _min_leaf_size = para_dict["min_leaf_size"]

    comb = []
    for i in _n_estimators if len(_n_estimators) else [None]:
        for j in _max_depth if len(_max_depth) else [None]:
            for k in _min_leaf_size if len(_min_leaf_size) else [2]:
                comb.append((i, j, k))

    res = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(_predictWithComb)(dataset, i) for i in comb)

    return res[np.argmax([r[0] for r in res])]
