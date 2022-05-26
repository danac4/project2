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


def valid_int(num):
    assert int(num), 'Invalid Input!'
    res = int(num)
    assert res > 0, 'Invalid Input!'
    return res


def main_validity_check():
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Invalid Input!")
        exit(1)
    k = validity_check_k()
    if len(sys.argv) == 6:
        max_iter = valid_int(sys.argv[2])
        eps = valid_int(sys.argv[3])
        index_1 = 4
    else:
        max_iter = MAX_ITER
        eps = valid_int(sys.argv[2])
        index_1 = 3
    return k, max_iter, eps, sys.argv[index_1], sys.argv[index_1+1]



k, max_iter, eps, path1, path2 = main_validity_check()
