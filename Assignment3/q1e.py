from singlelinkage import singlelinkage
from kmeans import kmeans
import numpy as np
import matplotlib.pyplot as plt
from q1c import get_table

def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]

def get_graph_data(sample_size, k):
    data = np.load('mnist_all.npz')
    data_x, data_y = [],[]
    for i in range(10):
        data_x.append(data[f'train{i}'])
        data_y.append(i)
    X,Y = gensmallm(data_x, data_y, sample_size)

    kmeans_table, kmeans_error = get_table(X,Y,k,kmeans)
    linkage_table, linkage_error = get_table(X,Y,k,singlelinkage)
    print(f"The classification error for kmeans with k={k} and a random sample of size {sample_size} was:{kmeans_error/Y.shape[0]}")
    print(f"The classification error for singlelinkage with k={k} and a random sample of size {sample_size} was:{linkage_error/Y.shape[0]}")
    return kmeans_table, linkage_table


def main():
    k= 10
    sample_size = 1000
    kmeans_table, linkage_table = get_graph_data(sample_size, 6)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5))
    ax1.axis('tight')
    ax1.axis('off')
    ax1.table(cellText=kmeans_table,colLabels=None,loc='center')
    ax2.axis('tight')
    ax2.axis('off')
    ax2.table(cellText=linkage_table,colLabels=None,loc='center')
    fig.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()