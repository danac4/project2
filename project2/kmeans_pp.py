import sys
import numpy as np
import pandas as pd
import mykmeanssp

np.random.seed(0)
MAX_ITER = 300


# Checking that k is an integer and is larger then 1
def validity_check_k():
    assert int(sys.argv[1]), 'Invalid Input!'
    k_res = int(sys.argv[1])
    assert k_res > 1, 'Invalid Input!'
    return k_res


# Checking that max_iter is an integer and is larger the 0
def validity_check_max_iter():
    assert int(sys.argv[2]), 'Invalid Input!'
    res = int(sys.argv[2])
    assert res > 0, 'Invalid Input!'
    return res


# Checking that epsilon is a non negative float
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


# main input validity check, checks that given input is in correct format and has the correct arguments
def main_validity_check():
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Invalid Input!")
        exit(1)
    k = validity_check_k()
    if len(sys.argv) == 6:  # max_iter is provided
        max_iter = validity_check_max_iter()
        eps = validity_check_eps(sys.argv[3])
        index_1 = 4
    else:  # max_iter = 300
        max_iter = MAX_ITER
        eps = validity_check_eps(sys.argv[2])
        index_1 = 3
    return k, max_iter, eps, sys.argv[index_1], sys.argv[index_1 + 1]


# Collecting data from input files,
# if one of them is incorrect an Invalid Input! message is displayed and the program is terminated.
def read_files(path1, path2):
    try:
        file1 = pd.read_csv(path1, header=None)
        try:
            file2 = pd.read_csv(path2, header=None)
            mergedData = pd.merge(file1, file2, on=0)  # innerjoin of two dataFrames using the first column as key
            mergedData = mergedData.sort_values(0)
            indices = mergedData[0].to_numpy(np.int32)
            mergedData = (mergedData.drop(0, axis=1)).to_numpy()
            return indices, mergedData
        except OSError:  # file2 is invalid
            print("Invalid Input!")
            exit(1)
    except OSError:  # file1 is invalid
        print("Invalid Input!")
        exit(1)


# implementation of k-means++, n = number of points, dim = dimension of each point,
# init_centroids = array with the index of k points selected to be the initial centroids
# min_dis = at first is a constant of infinity, after first iteration will be an np array of size n containing
# the min distances of the points given from all of the centroids that have been chosen.
# p = the probability to be selected of each point in points, at start the probability of each point is equal.
def k_means_pp(pIndicies, points, k):
        n, dim = points.shape
        init_centroids = []
        min_dis = np.inf
        p = None
        for j in range(k):  # initializing k centroids as in k-means++ initialization
            curr = np.random.choice(n, p=p)  # picking a random index of points provided
            init_centroids.append(curr)
            distances = np.power((points-points[curr]), 2).sum(axis=1)
            min_dis = np.minimum(distances, min_dis)
            p = np.divide(min_dis, min_dis.sum())
        res = mykmeanssp.fit(k, n, dim, max_iter, eps,  points[init_centroids].tolist(), points.tolist())
        print(','.join([str(i) for i in pIndicies[init_centroids]]))  # prints the indices of observations chosen by
        # the K-means++ algorithm as the initial centroids.
        for centroid in res:  # prints the final centroids from the K-means algorithm executed in c
            print(",".join(str(format(np.round(coord, 4))) for coord in centroid))


if __name__ == "__main__":
    k, max_iter, eps, path1, path2 = main_validity_check()  # checks input is correct
    indices, points = read_files(path1, path2)  # load data from input files
    if k > points.shape[0]:  # k is larger then the number of points
        print("Invalid Input!")
        exit(1)
    k_means_pp(indices, points, k)  # implement K-means++ on processed data
