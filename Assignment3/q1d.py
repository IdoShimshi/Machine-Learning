from singlelinkage import singlelinkage
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

    table, error = get_table(X,Y,k,singlelinkage)
    print(f"The classification error for singlelinkage with k={k} and a random sample of size {sample_size} was:{error/Y.shape[0]}")
    return table


def main():
    k= 10
    sample_size = 1000
    table = get_graph_data(sample_size, k)
    print(table)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(table, loc='center')

    fig.tight_layout()

    plt.show()
    

if __name__ == '__main__':
    main()