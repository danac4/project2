import sys
import numpy as np
import pandas as pd
np.random.seed(0)
MAX_ITER = 300


def validity_check_k():
    assert int(sys.argv[1]), 'Invalid Input!'
    k_res = int(sys.argv[1])
    assert k_res > 1, 'Invalid Input!'
    return k_res


def validity_check_max_iter():
    assert int(sys.argv[2]), 'Invalid Input!'
    res = int(sys.argv[2])
    assert res > 0, 'Invalid Input!'
    return res


def validity_check_eps(num):
    assert float(num), 'Invalid Input!'
    res = float(num)
    assert res >= float(0), 'Invalid Input!'
    return res


def main_validity_check():
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Invalid Input!")
        exit(1)
    k = validity_check_k()
    if len(sys.argv) == 6:
        max_iter = validity_check_max_iter()
        eps = validity_check_eps(sys.argv[3])
        index_1 = 4
    else:
        max_iter = MAX_ITER
        eps = validity_check_eps(sys.argv[2])
        index_1 = 3
    return k, max_iter, eps, sys.argv[index_1], sys.argv[index_1 + 1]


def read_files(path1, path2):
    try:
        file1 = pd.read_csv(path1, header=None)
        try:
            file2 = pd.read_csv(path2, header=None)
            mergedData = pd.merge(file1, file2, on=0)
            mergedData = mergedData.sort_values(0)
            indices = mergedData[0].to_numpy(np.int32)
            mergedData = (mergedData.drop(0, axis=1)).to_numpy()
            return indices, mergedData
        except OSError: #FileNotFound is subclass of OSError
            print("Invalid Input!")
            exit(1)
    except OSError:
        print("Invalid Input!")
        exit(1)


def k_means_pp(pIndicies, points, k): #implementation of k-means++
        N, dim = points.shape
        centroids = []


k, max_iter, eps, path1, path2 = main_validity_check()
indices, points = read_files(path1, path2)
if k > points.shape[0]: #k is larger then the number of points
    print("Invalid Input!")
    exit(1)

