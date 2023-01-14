import numpy as np


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


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    
    return [x_train, y_train, k]

def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    ret_labels = []
    x_train = classifier[0]
    y_train = classifier[1]
    k = classifier[2]
    for x in x_test:
        dist = np.apply_along_axis(np.linalg.norm, 1, x_train-x)
        index_k_smallest = np.argpartition(dist, k)[:k]
        labels = [y_train[i] for i in index_k_smallest]
        ret_labels.append(max(set(labels), key = labels.count))
    
    ret_labels = np.array(ret_labels)
    ret_labels.shape = [ret_labels.shape[0],1]

    return ret_labels

def simple_test():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    x_train, y_train = gensmallm([train2, train3, train5, train6], [2,3,5,6], 100)

    x_test, y_test = gensmallm([test2, test3, test5, test6], [2,3,5,6], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    
    # print(preds.shape)
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")
    print(f"The {i}'th test sample is {y_test[i]}")


if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    simple_test()


