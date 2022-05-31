import sys
import numpy as np
import pandas as pd
import mykmeanssp
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
    try:
        res = float(num)
        if res < 0:
            print("Invalid Input!")
            exit(1)
        else:
            return res
    except ValueError:
        print("Invalid Input!")
        exit(1)


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
        n, dim = points.shape
        init_centroids = []
        min_dis = np.inf
        p = None
        for j in range(k):
            curr = np.random.choice(n, p=p)
            init_centroids.append(curr)
            distances = np.power((points-points[curr]), 2).sum(axis=1)
            min_dis = np.minimum(distances, min_dis)
            p = np.divide(min_dis, min_dis.sum())
        res = mykmeanssp.fit((k, n, dim, eps, max_iter, points[init_centroids].tolist(), points.tolist()))
        res_centroids = np.round(np.array(res), 4)
        print(','.join([str(i) for i in pIndicies[init_centroids]]))
        for row in range(k):
            for col in range(dim):
                s = str(res_centroids[row][col])
                if col < dim-1:
                    print(s + ",")
                else:
                    print(s + "\n")



k, max_iter, eps, path1, path2 = main_validity_check()
indices, points = read_files(path1, path2)
if k > points.shape[0]: #k is larger then the number of points
    print("Invalid Input!")
    exit(1)
k_means_pp(indices, points, k)
