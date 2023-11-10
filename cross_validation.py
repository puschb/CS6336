import numpy as np
import math


def k_partition(X, y, k=10, i=0):
    """
    takes a dataset X and a label set y,
    returns a separated version of these datasets
    such that the ith fold of the dataset has been
    extracted from the dataset
    """
    if k > len(X):
        raise Exception("k is larger than the size of the dataset")
    if i >= k:
        raise Exception("i is larger than k")
    partition_size = int(round(len(X) / k))
    left_side = i * partition_size
    right_side = (i + 1) * partition_size if i < k - 1 else len(X)
    val_X = X[left_side:right_side]
    val_y = y[left_side:right_side]
    test_X = np.asarray(
        [x for (ind, x) in enumerate(X) if ind not in range(left_side, right_side)]
    )
    test_y = np.asarray(
        [
            item
            for (ind, item) in enumerate(y)
            if ind not in range(left_side, right_side)
        ]
    )
    return test_X, test_y, val_X, val_y


def k_cross_val(func, k, X, y):
    """
    takes a callable (must return a classifier), k, X, and y
    runs k-fold cross validation on the classifier
    returns the accuracy of the model with the i'th fold used for the validation set
    """
    acc = []

    # for every fold
    for i in range(k):
        tx, ty, vx, vy = k_partition(X, y, k=k, i=i)
        clf = func(tx, ty)
        a = clf.score(vx, vy)
        acc.append(a)

    return acc
