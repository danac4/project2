import sys
import pandas as pd
import numpy as np
import mykmeanssp


def isNaturalNumber(string):
    """validating numeric arguments"""
    return string.isnumeric and int(float(string)) == float(string) and float(string) > 0


def validation_and_initialization(arr_of_arg):
    """receiving and processing given arguments"""
    res_tuple = [0, 300, "", ""]
    len_arg_arr = len(arr_of_arg)
    assert isNaturalNumber(arr_of_arg[1]), "Invalid k value"
    res_tuple[0] = int(arr_of_arg[1])
    if len_arg_arr == 5:  # max iteration value is given
        assert isNaturalNumber(sys.argv[2]), "Invalid max iteration value"
        res_tuple[1] = int(arr_of_arg[2])
        res_tuple[2] = arr_of_arg[3]
        res_tuple[3] = arr_of_arg[4]
    else:
        res_tuple[2] = arr_of_arg[2]
        res_tuple[3] = arr_of_arg[3]
    return res_tuple


def processed_data(path_file1, path_file2):
    file1 = open(path_file1, "r")
    d = len(file1.readline().split(","))
    df1 = pd.read_csv(path_file1, delimiter=",", names=["id"] + [i for i in range(1, d)])
    df2 = pd.read_csv(path_file2, delimiter=",", names=["id"] + [i for i in range(d, 2 * d - 1)])
    merged = pd.merge(df1, df2, on="id")  # inner join by id column
    merged = merged.sort_values(by="id")  # sorting the points by index
    return merged


def norma(df_vector1, df_vector2):
    """calculating the distance between two points"""
    res = 0
    d = len(df_vector1)
    for i in range(0, d):
        x = df_vector1[i]
        y = df_vector2[i]
        res += float(float(x) - float(y)) ** 2
    return res


def k_mean_pp(k, arr_points):
    """a function that receives all of the points and returns chosen k centroids"""
    np.random.seed(0)
    n = len(arr_points)
    assert(k<n), "Invalid k value"
    centorids_list = []
    random_row = np.random.choice(n, 1)  # choosing a random index
    centorids_list.append(arr_points[random_row[0]])
    for z in range(1, k):
        d_arr = np.zeros(n)
        for i in range(n):
            x = arr_points[i]
            d_arr[i] = min([norma(x[1:], centorids_list[j][1:]) for j in range(0, z)])  # calculating D value of points
        p_arr = np.zeros(n)
        for i in range(n):
            p_arr[i] = float(float(d_arr[i]) / float(sum(d_arr)))  # calculating P value for each point
        random_row = np.random.choice(n, 1, p=p_arr)   # choosing a new centroid with the P probabilty array
        centorids_list.append(arr_points[random_row[0]])
    return centorids_list


if __name__ == "__main__":
    arg_tuple = validation_and_initialization(sys.argv)
    k = arg_tuple[0]
    max_iter = arg_tuple[1]
    path_file1 = arg_tuple[2]
    path_file2 = arg_tuple[3]
    df_points = processed_data(path_file1, path_file2)
    arr_points = df_points.to_numpy()   # numpy conversion
    arr_points_py_list = []
    for lst in arr_points:  # list conversion of each point
        lst = lst.tolist()
        arr_points_py_list.append(lst)

    centorids_list = k_mean_pp(k, arr_points)
    centorids_list_as_py = []
    for lst in centorids_list:   # list conversion of each centroid
        lst = lst.tolist()
        centorids_list_as_py.append(lst)

    centroid_indices = []
    for centroid in centorids_list:
        centroid_indices.append(int(centroid[0]))  # extracting each index of chosen centroid
    print(",".join([str(i) for i in centroid_indices]))
    n = len(arr_points)
    d = len(arr_points[0]) - 1
    # calling the k_means module in C and returning the final centroids
    res = mykmeanssp.fit(k, n, d, max_iter, arr_points_py_list, centorids_list_as_py)
    for centroid in res:
        print(",".join(str(format(np.round(coord, 4))) for coord in centroid))
