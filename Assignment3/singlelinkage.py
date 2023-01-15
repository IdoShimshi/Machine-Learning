import numpy as np
import scipy as sio
from scipy.spatial import distance_matrix


def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m = X.shape[0]
    cluster_distances = distance_matrix(X,X,2)
    np.fill_diagonal(cluster_distances, np.inf)
    clusters = np.arange(m)
    for j in range(m-k):
        argmin = np.unravel_index(cluster_distances.argmin(), cluster_distances.shape)
        for i in range(len(clusters)):
            if clusters[i] == argmin[0]:
                clusters[i] = argmin[1]
        new_distances = [min(cluster_distances[i,argmin[0]],cluster_distances[i,argmin[1]]) if cluster_distances[i,argmin[1]] != np.inf else np.inf for i in range(m)]
        cluster_distances[:,argmin[1]] = new_distances
        cluster_distances[argmin[1]] = new_distances
        cluster_distances[:,argmin[0]] = np.full((m),np.inf)
        cluster_distances[argmin[0]] = np.full((m),np.inf)


    replace_large_numbers(clusters)
    return np.array(clusters).reshape(-1,1)

def replace_large_numbers(arr):
    # Create a dictionary to store the mapping of large numbers to small numbers
    mapping = {}
    # Iterate through the array
    for i in range(len(arr)):
        # If the current number is not already in the mapping
        if arr[i] not in mapping:
            # Assign it the next available number
            mapping[arr[i]] = len(mapping)
    # Iterate through the array
    for i in range(len(arr)):
        # Replace each element with its corresponding small number
        arr[i] = mapping[arr[i]]
    return arr


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    # X = np.concatenate((data['train0'], data['train1']))
    data_i = data[f'train0']
    np.random.shuffle(data_i)
    X = data_i[:30]
    for i in range(1,10):
        data_i = data[f'train{i}']
        np.random.shuffle(data_i)
        X = np.concatenate((X,data_i[:30]))
    m, d = X.shape

    # run single-linkage
    c = singlelinkage(X, k=10)
    print(c)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
