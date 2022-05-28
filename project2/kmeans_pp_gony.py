import sys
import numpy as np
import pandas as pd


def prepare():
    assert 4 < len(sys.argv) < 7, 'Invalid Input!'
    k = validity_check_k()
    max_iter = validity_check_iter()
    if len(sys.argv) > 5:
        file_1 = sys.argv[4]
        file_2 = sys.argv[5]
        epsilon = sys.argv[3]
    else:
        file_1 = sys.argv[3]
        file_2 = sys.argv[4]
        epsilon = sys.argv[2]
    epsilon = epsilon_check(epsilon)
    points_matrix = process_file(file_1, file_2)
    assert k <= points_matrix.shape[0], 'Invalid Input!'
    return k, max_iter, epsilon, points_matrix


def validity_check_iter():
    max_iter = 300
    if len(sys.argv) > 5:
        assert sys.argv[2].isnumeric(), 'Invalid Input!'
        max_iter = int(sys.argv[2])
        assert max_iter > 0, 'Invalid Input!'
    return max_iter


def epsilon_check(epsilon):
    try:
        result = float(epsilon)
    except ValueError:
        print('Invalid Input!')
    return result


def process_file(file_1, file_2):
    frame_1 = pd.read_csv(file_1, header=None)
    frame_1 = frame_1.sort_values(by=0)
    frame_2 = pd.read_csv(file_2, header=None)
    frame_2 = frame_2.sort_values(by=0)
    frame = pd.merge(frame_1, frame_2, on=0)
    frame = frame.drop(columns=[0])
    frame.columns = [i for i in range(frame.shape[1])]
    return frame


def validity_check_k():
    assert sys.argv[1].isnumeric(), 'Invalid Input!'
    k = int(sys.argv[1])
    assert k > 1, 'Invalid Input!'
    return k


def table_construction(n):
    auxiliary = pd.DataFrame({"probability": np.zeros(n), "min_D": np.full(n, np.NaN), "curr_D": np.full(n, np.NaN)})
    return auxiliary


def distance_calc(column, index, auxiliary, points):
    x = points - points.loc[index]
    auxiliary[column] = np.sum(x * x, axis=1)


def probability_calc(auxiliary):
    total = auxiliary["min_D"].sum()
    auxiliary["probability"] = auxiliary["min_D"]/total


def kmeans_pp():
    k, max_iter, epsilon, points_matrix = prepare()
    points_list = points_matrix.values.tolist()
    n = points_matrix.shape[0]
    auxiliary = table_construction(n)
    centroids_by_index = []
    centroids = []

    np.random.seed(0)
    c = np.random.choice(n - 1, 1)[0]
    distance_calc("min_D", c, auxiliary, points_matrix)
    centroids_by_index.append(c)
    centroids.append(points_list[c])

    z = 1
    while z < k:
        curr_miu = centroids_by_index[z-1]
        distance_calc("curr_D", curr_miu, auxiliary, points_matrix)
        auxiliary["min_D"] = np.minimum(auxiliary["min_D"], auxiliary["curr_D"])
        probability_calc(auxiliary)
        tmp = np.random.choice(n, 1, p=auxiliary["probability"])[0]
        centroids_by_index.append(tmp)
        centroids.append(points_list[tmp])
        z += 1


if __name__ == '__main__':
    kmeans_pp()
