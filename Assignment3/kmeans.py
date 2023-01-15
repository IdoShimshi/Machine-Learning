import numpy as np


def kmeans(X, k, t=100):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    centers = np.random.rand(k,X.shape[1])
    clusters_old = split_to_cluster(X,centers)
    clusters_new = []
    for i in range(t):
        if np.array_equal(clusters_old,clusters_new):
            print("converged")
            return np.array(clusters_old).reshape(-1,1) 
        clusters_old = clusters_new
        clusters_new = split_to_cluster(X,centers)
        centers = calc_centroids(clusters_new,X,k, centers)  
    return np.array(clusters_new).reshape(-1,1) 


def split_to_cluster(X,centers):
    clusters = []
    for i in range(X.shape[0]):
        distances = []
        for j in range(centers.shape[0]):
            distances.append(np.linalg.norm(X[i]-centers[j]))
        clusters.append(np.argmin(distances))
    return np.array(clusters)

def calc_centroids(clusters,X,k, old_centers):
    centroids = []
    for i in range(k):
        cluster = X[clusters == i]
        if len(cluster) == 0:
            centroids.append(old_centers[i])
        else:
            centroids.append(np.mean(cluster,axis=0))
    return np.array(centroids)

def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape
    # run K-means
    c = kmeans(X, k=10, t=10)

    print(c)
    print(c.shape)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
