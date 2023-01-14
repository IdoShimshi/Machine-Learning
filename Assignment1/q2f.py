from nearest_neighbour import learnknn, predictknn, gensmallm
import numpy as np
import matplotlib.pyplot as plt

def get_diff_label(curr_label, num_list):
    other_labels = np.setdiff1d(num_list,[curr_label])
    return np.random.choice(other_labels,1)[0]


def corrupt_set(y, num_list):
    ret = np.copy(y)
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    indices = indices[:int(0.15*y.shape[0])]
    
    for i in indices:
        ret[i] = get_diff_label(y[i],num_list)
    return ret

def get_graph_data(k_values):
    data = np.load('mnist_all.npz')
    num_list = [2,3,5,6]
    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    x_test, y_test = gensmallm([test2, test3, test5, test6], num_list, 100)
    y_test = corrupt_set(y_test, num_list)

    error_data = []

    for k in k_values:
        errs = []
        for i in range(10):
            x_train, y_train = gensmallm([train2, train3, train5, train6], [2,3,5,6], 200)
            y_train = corrupt_set(y_train,num_list)
            classifer = learnknn(k, x_train, y_train)
            preds = predictknn(classifer, x_test)

            count = 0
            for i in range(y_test.shape[0]):
                if y_test[i] != preds[i]:
                    count += 1
            errs.append(count/y_test.shape[0])

        error_data.append(np.mean(np.array(errs)))

    
    return error_data

def main():
    k_values = [i for i in range(1,12)]
    errors = get_graph_data(k_values)
    plt.style.use('seaborn-whitegrid')
    plt.scatter(k_values, errors)
    plt.xlabel("k")
    plt.ylabel("mean error")

    plt.show()
    

if __name__ == '__main__':
    main()