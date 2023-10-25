import argparse
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets
from sklearn.cluster import KMeans


def sampling_pp2(data, k):
    num_data_points = data.shape[0]
    random_index = np.random.choice(num_data_points)
    current_centroid = data[random_index]
    centroid_pp = np.ndarray(shape=(k, 2))
    centroid_pp[0] = current_centroid
    for j in range(k - 1):
        total_distance = 0
        distances = np.zeros(num_data_points)
        for i_i in range(num_data_points):
            distances[i_i] = np.linalg.norm(data - current_centroid) ** 2
            total_distance += distances[i_i]
        distances = distances / total_distance
        current_centroid = data[np.random.choice(num_data_points, p=distances)]
        centroid_pp[j + 1] = current_centroid
    return centroid_pp


def sampling_pp(data, k):
    num_data_points = data.shape[0]
    random_index = np.random.choice(num_data_points)
    centroid_pp = np.ndarray(shape=(k, 2))
    centroid_pp[0] = data[random_index]

    distances = np.linalg.norm(data - [centroid_pp[0]], axis=1)

    for i in range(1, k):
        # choose the next centroid, the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid
        prob = distances ** 2
        rand_index = np.random.choice(data.shape[0], size=1, p=prob / np.sum(prob))
        centroid_pp[i] = data[rand_index]

        if i == k - 1:
            break
        distances_new = np.linalg.norm(data - [centroid_pp[i]], axis=1)
        distances = np.min(np.vstack((distances, distances_new)), axis=0)
    return centroid_pp

def min_max_scale(data):
    data = np.array(data)
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data


def standardize(data):
    data = np.array(data)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data


def elbow_method(data, max_k=12):
    wcss_values = []
    for k in range(1, max_k):
        y, elbow_centroid = k_means(data, k)
        wcss = calc_wcss(data, k, elbow_centroid, y)
        wcss_values.append(wcss)
    plt.clf()
    plt.plot(range(1, max_k), wcss_values, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within Sum of Squares')
    # plt.savefig('nonelbow method.png', dpi=300)
    plt.show()
    return wcss_values


def calc_wcss(data, k, wcc_centroids, y):
    wcss = 0
    for cluster in range(k):
        cluster_points = data[y == cluster]
        for point in cluster_points:
            wcss += np.linalg.norm(point - wcc_centroids[cluster])
    return wcss


def k_means(data, k, max_iter=100, plot_tag=False):
    data = standardize(data)
    y = np.zeros(data.shape[0])
    cost_function = np.zeros(max_iter)
    np.random.seed(42)

    k_means_centroids = sampling_pp(data, k)  # Initialize centroids with k-means++
    #k_means_centroids = data[np.random.choice(data.shape[0], k, replace=False)]  # Initialize centroids randomly
    if plot_tag:
        plot_data(data, y, title='Initial Data Points')
        plot_data(data, y, k_means_centroids, 'Initial Centroids with k-means++ centroids')
    k_iter = 0
    prev_cost = 0
    while k_iter < max_iter:
        cost_iter = 0
        for index, data_point in enumerate(data):
            distances = np.linalg.norm(data_point - k_means_centroids, axis=1)
            y[index] = np.argmin(distances)
            cost_iter += np.min(distances)
        if abs(cost_iter - prev_cost) > 0.0001:
            prev_cost = cost_iter
            if plot_tag and k_iter < 3:
                plot_data(data, y, k_means_centroids, 'Iteration ' + str(k_iter + 1))
        else:
            if plot_tag:
                plot_data(data, y, k_means_centroids, 'Last Iteration (' + str(k_iter + 1) + ')')
            break
        k_means_centroids = np.array([np.mean(data[y == i_i], axis=0) for i_i in range(k)])
        cost_function[k_iter] = cost_iter
        k_iter += 1

    return_y = y
    return_centroids = k_means_centroids

    while k_iter < max_iter:
        cost_iter = 0
        for index, data_point in enumerate(data):
            distances = np.linalg.norm(data_point - k_means_centroids, axis=1)
            y[index] = np.argmin(distances)
            cost_iter += np.min(distances)
        k_means_centroids = np.array([np.mean(data[y == i_i], axis=0) for i_i in range(k)])
        cost_function[k_iter] = cost_iter
        k_iter += 1

    if plot_tag:
        plt.clf()
        plt.plot(range(1, 16), cost_function[:15], marker='o')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost function')
        plt.ylim(0, cost_function[0]+ 0.1 * cost_function[0])
        # plt.savefig('cost_function'+ str(k) +'.png', dpi=300)
        plt.show()
    return return_y, return_centroids


def plot_data(data, y, plt_centroids=None, title=None):
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], c=y, s=2)
    plt.title(title)
    if plt_centroids is not None:
        plt.scatter(plt_centroids[:, 0], plt_centroids[:, 1], c='r', marker='x')

    # plt.savefig(title + '.png', dpi=300)
    plt.show()



def parse_cluster_std(value):
    l = [float(x) for x in value.split(',')]
    return l[0] if len(l) == 1 else l

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000, help='Specify the number of samples for the dataset. Int value greater than 0.')
    parser.add_argument('--dataset', type=str, default='blobs', help = 'Specify the dataset. \'blobs\' or \'moon\'')
    parser.add_argument('--centers', type=int, default=3, help='Specify the number of centers for blobs dataset. Int value greater than 0.')
    parser.add_argument('--noise', type=float, default=0.1, help='Specify the noise for the dataset if you are going to use moon datset. Float value between 0 and 1')
    parser.add_argument('--shuffle', type=bool, default=True, help='Specify whether to shuffle the dataset. True or False, Boolean.')
    parser.add_argument('--cluster_std', type=float, default = 0.3,
                        help='Specify the cluster_std for blobs dataset. Float value between 0 and 1')
    parser.add_argument('--k', type=int, default= 0, help='Specify the number of clusters if you do not want to use the automatic elbow method. Int value greater than 0.')
    args = parser.parse_args()
    data_set = None
    if args.dataset == 'blobs':
        data_set = datasets.make_blobs(n_samples=args.n_samples, centers=args.centers,shuffle= args.shuffle,cluster_std= args.cluster_std, n_features= 2)

    elif args.dataset == 'moon':
        data_set = datasets.make_moons(n_samples=args.n_samples, shuffle=args.shuffle, noise=args.noise)

    if args.k > 0:
        k = args.k
    else:
        wcss_values = elbow_method(data_set[0])
        diff = np.diff(wcss_values)
        ratios = []
        for i in range(len(diff) - 1):
            ratios.append(diff[i] / diff[i + 1])
        max_index = ratios.index(max(ratios))
        k = max_index + 2

    labels, centroids = k_means(data_set[0], k, plot_tag=True)

    kmeans_sckit = KMeans(random_state=0, n_init="auto",init='k-means++', n_clusters=k).fit(data_set[0])
    plt.clf()
    plt.scatter(data_set[0][:, 0], data_set[0][:, 1], c=kmeans_sckit.labels_, s=2)
    plt.title('K-means with sklearn')
    # plt.savefig('nonK-means with sklearn6.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()