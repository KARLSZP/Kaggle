# Report - Subtask1

>  Research on (MBTI) Myers-Briggs Personality Type Dataset - Ensemble Technique(Random Forest)

[toc]

## 1 Abstract

This report is mainly about the following 5 sections:

1. An implementation of `Random Forest` in Python.

   Although I've implemented a Decision Tree class as the base estimator, unfortunately, without the cython technique, it perform really slow on dataset that large. So the base estimator used in Random Forest is DecisionTreeClassifier imported from sklearn.

2. Preprocessing of the MBTI dataset.

3. Training 4 classifier on those 4 axes, respectively.

   - **clf_ie**: Introversion (I) – Extroversion (E)
   - **clf_ns**: Intuition (N) – Sensing (S)
   - **clf_tf**: Thinking (T) – Feeling (F)
   - **clf_jp**: Judging (J) – Perceiving (P)

4. Separate and overall benchmarks.

5. Interesting research on Kaggle ForumMessages.

---

<div style="page-break-after: always;"></div>

## 2 Random Forest

Code in this section is stored in `RF.py` and `DT.py`.

### 2.1 Logic

The logic of Random Forest is simple and clear:

1. APIs are defined similarly as sklearn does.
2. Use joblib.Parallel technique to accelerate the fitting process among n_estimators.
3. Prediction is determined by all estimators together, that is, each tree vote for its prediction, and majority wins.

```python
from joblib import Parallel, delayed
import pickle as pk
import pandas as pd
import numpy as np
import random as rd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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
```

---

### 2.2 A simple example

<img src="/home/karl/.config/Typora/typora-user-images/image-20200603140908258.png" alt="image-20200603140908258" style="zoom:80%;" />

---

### 2.3 Appendix - Decision Tree

Although I use Decision Tree from sklearn in order to accelerate through Cython technique, I also implement a decision tree, as follow:

```python
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
        labels_counts = [labels.count(label) for label in set(labels)]
        probs = [prob/len(splitted_dataset) for prob in labels_counts]
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

        return {'index': b_index,
                'value': b_value,
                'score': b_score,
                'splits': b_splits}

    def __vote(self, splitted_dataset):
        labels = [sample[-1] for sample in splitted_dataset]
        res = max(set(labels), key=labels.count)
        return res

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
        return self.__predict(self.root, case)
```

---

<div style="page-break-after: always;"></div>

## 3 Preprocessing

Preprocessing is done in `Preprocessor.ipynb`.

As text material, data is preprocessed and store as `tf-idf`.

![image-20200603142142978](/home/karl/.config/Typora/typora-user-images/image-20200603142142978.png)

![image-20200603142207054](/home/karl/.config/Typora/typora-user-images/image-20200603142207054.png)

![image-20200603142237644](/home/karl/.config/Typora/typora-user-images/image-20200603142237644.png)

![image-20200603142252176](/home/karl/.config/Typora/typora-user-images/image-20200603142252176.png)

---

<div style="page-break-after: always;"></div>

## 4 Training

Training and bench marking is done in `mbti-random-forest.ipynb`.

![image-20200603142648305](/home/karl/.config/Typora/typora-user-images/image-20200603142648305.png)

![image-20200603142704244](/home/karl/.config/Typora/typora-user-images/image-20200603142704244.png)

![image-20200603142733759](/home/karl/.config/Typora/typora-user-images/image-20200603142733759.png)

---

<div style="page-break-after: always;"></div>

## 5 Separate and overall benchmarks

For each type, train a type-specified classifier and benchmark individually.

![image-20200603142809456](/home/karl/.config/Typora/typora-user-images/image-20200603142809456.png)

For the whole dataset, combine classifiers above and benchmark.

![image-20200603142854245](/home/karl/.config/Typora/typora-user-images/image-20200603142854245.png)

![image-20200603142919357](/home/karl/.config/Typora/typora-user-images/image-20200603142919357.png)

**NOTE**

1. Those case that f1-score is 0.0 means the prediction incorrectly missed some existed label:

   e.g.: true = [1, 0, 1], prediction = [0, 0, 0] --> f1-score = 0.0

2. In my opinion, the non-strict benchmarker seems to be more dependable.

---

<div style="page-break-after: always;"></div>

## 6 Further study

For further study, I download Kaggle ForumMessages and do an interesting research on it.

![image-20200603151830650](/home/karl/.config/Typora/typora-user-images/image-20200603151830650.png)

---

<div style="page-break-after: always;"></div>

## 7 Conclusion

Thanks for your reading and please refer to (Jupyter notebook necessary):

1. [src/mbti-random-forest.ipynb](src/mbti-random-forest.ipynb)
2. [src/RF.py](src/RF.py)
3. [src/DT.py](src/DT.py)
4. [src/utilities.py](src/utilities.py)

for a better experience!



In this experiment, it's clear that:

The model performs bad when making classification on type "I-E" and "N-S". 

<img src="/home/karl/.config/Typora/typora-user-images/image-20200603155144513.png" alt="image-20200603155144513" style="zoom: 50%;" /> <img src="/home/karl/.config/Typora/typora-user-images/image-20200603155209452.png" alt="image-20200603155209452" style="zoom: 50%;" />

While, it performs well on type "T-F" and "J-P". 

<img src="/home/karl/.config/Typora/typora-user-images/image-20200603155330762.png" alt="image-20200603155330762" style="zoom:50%;" /> <img src="/home/karl/.config/Typora/typora-user-images/image-20200603155342746.png" alt="image-20200603155342746" style="zoom:50%;" />

To tell why, I observe the log many times and finally found a possible reason:

The inbalance in training data cause that.

---



## 8 References

1. [Kaggle - MBTI dataset](https://www.kaggle.com/datasnaek/mbti-type)
2. [Myersbriggs - mbti-basics](https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/home.htm)
3. [Devdocs - scikit-learn documentation](https://devdocs.io/scikit_learn/)
4. [Joblib.Parallel](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)

---

2020/6

Karl

[BACK TO TOP](# Report - Subtask1)

