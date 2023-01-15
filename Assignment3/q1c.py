from kmeans import kmeans
import numpy as np
import matplotlib.pyplot as plt


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

def get_table(X,Y,k,alg):
    clustering = alg(X,k)
    table = [["Cluster Size", "common label", "% of common label"]]
    cluster_labels = {}
    for i in range(k):
        indexes = np.where(clustering == i)[0]
        label_count = {}
        max_label = -1
        max_label_count = 0
        for j in indexes:
            label = Y[j]
            if  label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1
            if max_label_count < label_count[label]:
                max_label_count = label_count[label]
                max_label = label
        cluster_labels[i] = max_label
        
        table.append([len(indexes), int(max_label), f"{round((max_label_count/len(indexes))*100,2)}%"])

    classification = np.copy(clustering).reshape(1,-1)[0]
    for i in range(len(classification)):
        classification[i] = cluster_labels[classification[i]]
    
    wrong_label_count = 0
    for i in range(Y.shape[0]):
        if Y[i] != classification[i]:
            wrong_label_count += 1

    return table, wrong_label_count

def get_graph_data(sample_size, k):
    data = np.load('mnist_all.npz')
    data_x, data_y = [],[]
    for i in range(10):
        data_x.append(data[f'train{i}'])
        data_y.append(i)
    X,Y = gensmallm(data_x, data_y, sample_size)

    table, error = get_table(X,Y,k,kmeans)
    print(f"The classification error for kmeans with k={k} and a random sample of size {sample_size} was:{error/Y.shape[0]}")
    return table


def main():
    k= 10
    sample_size = 1000
    table = get_graph_data(sample_size, k)
    print(table)

    # plt.style.use('seaborn-whitegrid')
    # plt.xlabel("lambda")
    # plt.ylabel("mean error")
    # plt.title("mean error versus lambda with 100 sample size")
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(table, loc='center')

    fig.tight_layout()

    plt.show()
    

if __name__ == '__main__':
    main()