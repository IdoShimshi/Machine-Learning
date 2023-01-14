from nearest_neighbour import learnknn, predictknn, gensmallm
import numpy as np
import matplotlib.pyplot as plt


def get_graph_data(train_sizes):
    data = np.load('mnist_all.npz')
    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    x_test, y_test = gensmallm([test2, test3, test5, test6], [2,3,5,6], 100)

    error_data = []
    err_mins =[]
    err_maxs = []

    for tsize in train_sizes:
        errs = []
        for i in range(10):
            x_train, y_train = gensmallm([train2, train3, train5, train6], [2,3,5,6], tsize)
            classifer = learnknn(1, x_train, y_train)
            preds = predictknn(classifer, x_test)

            count = 0
            for i in range(y_test.shape[0]):
                if y_test[i] != preds[i]:
                    count += 1
            errs.append(count/y_test.shape[0])

        error_data.append(np.mean(np.array(errs)))
        err_maxs.append(max(errs) - error_data[-1])
        err_mins.append(error_data[-1] - min(errs))

    
    return error_data, err_mins, err_maxs

def main():
    sample_sizes = [i for i in range(10,101)[::10]]
    errors, err_mins, err_maxs = get_graph_data(sample_sizes)

    plt.style.use('seaborn-whitegrid')
    plt.xlabel("training sample size")
    plt.ylabel("mean error")
    plt.errorbar(sample_sizes, errors, yerr=[err_mins,err_maxs], fmt='o')

    plt.show()
    

if __name__ == '__main__':
    main()