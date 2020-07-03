import pickle as pk
import pandas as pd
import numpy as np
import threading
from joblib import Parallel, delayed
import time
import random as rd
from tqdm import tqdm


class DecisionTree:
    """A decision tree(CART) classifier.

    Parameters
    ----------
    dataset : pandas.DataFrame or list
        The training dataset used to generate the tree.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. 
        If None, then nodes are expanded until all leaves are pure
        or until all leaves contain less than `min_leaf_size` samples.

    min_leaf_size : int, optional (default=2)
        The minimum number of samples required to split an internal(non-leaf) node.

    n_features : int, string or None, optional (default=sqrt)
        The number of features to consider when looking for the best split.

            - If int, then consider `n_features` features at each split, randomly.
            - If "auto", then `n_features=sqrt(n_features)`.
            - If "sqrt", then `n_features=sqrt(n_features)`.
            - If "log2", then `n_features=log2(n_features)`.
            - If None, then `n_features=n_features`.

    Attributes
    ----------
    max_depth : int or None
        The maximum depth of the tree.

    min_leaf_size : int
        The minimum number of samples required to split an internal(non-leaf) node.

    dataset ：numpy.ndarray
        The training material.

    features : list of string
        Features retrieved from dataset.

    labels : numpy.ndarray
        Labels retrieved from dataset.

    n_features : 
        The number of features to consider when looking for the best split.

    root : dict
        The tree root built with __generateDecisionTree().

    Examples
    --------
    >>> from DT import DecisionTree

    >>> train_data = ...

    >>> clf = DecisionTree(dataset=train_data, max_depth=10,
    ...                    min_leaf_size=5, n_features="auto")

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

    def __init__(self, dataset, max_depth=None, min_leaf_size=2, n_features="sqrt"):

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

        if isinstance(dataset, pd.DataFrame):
            self.features = dataset.columns[:-1].to_list()
            self.dataset = dataset.values
        elif isinstance(dataset, np.ndarray):
            self.features = list(range(dataset.shape[1]-1))
            self.dataset = dataset
        elif isinstance(dataset, list):
            self.features = dataset[0][:-1]
            self.dataset = np.array([x[:-1] for x in dataset[1:]])
        else:
            raise ValueError("dataset should be a DataFrame or a 2-d list.")

        if isinstance(n_features, int):
            self.n_features = n_features
        elif isinstance(n_features, str):
            if n_features == "auto" or n_features == "sqrt":
                self.n_features = np.int(np.sqrt(len(self.features)))
            elif n_features == "log2":
                self.n_features = np.int(np.log2((len(self.features))))
            else:
                raise ValueError(
                    "n_features only support methods 'auto', 'sqrt' and 'log2'.")
        elif n_features is None:
            self.n_features = len(self.features)
        else:
            raise ValueError(
                "n_features should be of type int, string or None")

        self.root = self.__generateDecisionTree()

    def __str__(self):
        return "hello"

    def __dataSplit(self, index, value, splitted_dataset):
        left = splitted_dataset[splitted_dataset[:, index] < value]
        right = splitted_dataset[splitted_dataset[:, index] >= value]
        return left, right

    def __gini(self, splitted_dataset):
        labels = [sample[-1] for sample in splitted_dataset]
        # To calculate Pr by
        # Pr = sample_num / total_num
        labels_counts = [labels.count(label) for label in set(labels)]
        probs = [prob/len(splitted_dataset) for prob in labels_counts]
        # Calculate final result
        return 1 - np.sum(np.power(probs, 2))

    def __giniIndex(self, splitted_datasets):
        gini_index = 0.0
        total_size = np.sum([len(x) for x in splitted_datasets])
        for splitted_dataset in splitted_datasets:
            ratio = len(splitted_dataset) / total_size
            gini_index += ratio * self.__gini(splitted_dataset)
        return gini_index

    def __getSplitPoint(self, splitted_dataset):
        features = rd.sample(
            range(0, len(splitted_dataset[0])-1), self.n_features)

        b_score, b_index, b_value, b_splits = 1, 0, 0, None
        for index in tqdm(features):
            ginis = [(index, self.__giniIndex(self.__dataSplit(index, row[index], splitted_dataset)), row[index])
                     for row in splitted_dataset]
            min_gini = np.argmin(ginis, axis=0)[1]
            if b_score > ginis[min_gini][1]:
                b_index = ginis[min_gini][0]
                b_score = ginis[min_gini][1]
                b_value = ginis[min_gini][2]
                b_splits = self.__dataSplit(b_index, b_value, splitted_dataset)

        # Return
        return {'index': b_index,
                'value': b_value,
                'score': b_score,
                'splits': b_splits}

    def __vote(self, splitted_dataset):
        # print("Voting:", splitted_dataset.shape)
        labels = [sample[-1] for sample in splitted_dataset]
        res = max(set(labels), key=labels.count)
        # print("Voting:", res)
        return res
        # return max(set(labels), key=labels.count)

    def __split(self, node, depth):
        left, right = node['splits']
        del node['splits']
        if not len(left) or not len(right):
            node['left'] = node['right'] = self.__vote(
                np.append(left, right, axis=0))
            return

        if self.max_depth is not None and depth >= self.max_depth:
            node['left'], node['right'] = self.__vote(left), self.__vote(right)
            return

        if len(left) <= self.min_leaf_size:
            node['left'] = self.__vote(left)
        else:
            node['left'] = self.__getSplitPoint(left)
            self.__split(node['left'], depth + 1)

        if len(right) <= self.min_leaf_size:
            node['right'] = self.__vote(right)
        else:
            node['right'] = self.__getSplitPoint(right)
            self.__split(node['right'], depth + 1)

    def __generateDecisionTree(self):
        print(">>> generating...")
        root = self.__getSplitPoint(self.dataset)
        self.__split(root, 1)
        return root

    def __predict(self, node, case):
        if case[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.__predict(node['left'], case)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.__predict(node['right'], case)
            else:
                return node['right']

    def predict(self, case):
        # print(case.shape, case)
        return self.__predict(self.root, case)
