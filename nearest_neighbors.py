import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import distances


class KNNClassifier:

    def __init__(self, k, strategy='my_own', metric='euclidean', weights=False, test_block_size=None):
        if not isinstance(k, int) or k < 1:
            raise AttributeError('Incorrect "k" parameter')
        if strategy != 'my_own' and strategy != 'brute' and strategy != 'kd_tree' and strategy != 'ball_tree':
            raise AttributeError('Incorrect "strategy" parameter')
        if metric != 'euclidean' and metric != 'cosine':
            raise AttributeError('Incorrect "metric" parameter')
        if not isinstance(weights, bool):
            raise AttributeError('Incorrect "weights" parameter')
        if (not isinstance(test_block_size, int) and test_block_size is not None) \
                or (isinstance(test_block_size, int) and test_block_size < 1):
            raise AttributeError('Incorrect "test_block_size" parameter')
        if (strategy == 'kd_tree' or strategy == 'ball_tree') and metric == 'cosine':
            raise AttributeError('Cannot use cosine metric with kd_tree or ball_tree')

        self.strategy = strategy
        self.weights = weights
        self.k = k
        self.training_labels = None
        if strategy == 'my_own':
            self.metric = metric
            self.test_block_size = test_block_size
            self.training_data = None
        else:
            self.data = NearestNeighbors(n_neighbors=k, algorithm=strategy, leaf_size=30, metric=metric)

    def fit(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[0] < self.k:
            raise AttributeError('Incorrect training set')
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise AttributeError('Incorrect labels on training set')
        if x.shape[0] != y.shape[0]:
            raise AttributeError('Mismatch between training set and its labels')

        if self.strategy == 'my_own':
            self.training_data = x
            self.training_labels = y
        else:
            self.data.fit(x)
            self.training_labels = y

    def find_kneighbors(self, x, return_distance=True):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise AttributeError('Incorrect array')
        if not isinstance(return_distance, bool):
            raise AttributeError('Incorrect "return_distance" parameter')

        if self.strategy == 'my_own':
            if self.test_block_size is None or self.test_block_size >= x.shape[0]:
                if self.metric == 'euclidean':
                    dist_matrix = distances.euclidean_distance(x, self.training_data)
                else:
                    dist_matrix = distances.cosine_distance(x, self.training_data)
                if not return_distance:
                    res_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    np.argmin(dist_matrix, axis=1, out=res_index)
                    dist_matrix[np.arange(dist_matrix.shape[0]), res_index] = np.inf
                    res_index = res_index.reshape((-1, 1))
                    for i in range(self.k - 1):
                        np.argmin(dist_matrix, axis=1, out=tmp_index)
                        dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                        res_index = np.hstack((res_index, tmp_index[:, np.newaxis]))
                    return res_index
                else:
                    res_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    np.argmin(dist_matrix, axis=1, out=res_index)
                    res_dist = dist_matrix[np.arange(dist_matrix.shape[0]), res_index]
                    dist_matrix[np.arange(dist_matrix.shape[0]), res_index] = np.inf
                    res_index = res_index.reshape((-1, 1))
                    res_dist = res_dist.reshape((-1, 1))
                    for i in range(self.k - 1):
                        np.argmin(dist_matrix, axis=1, out=tmp_index)
                        res_dist = np.hstack((res_dist,
                                              dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index][:, np.newaxis]))
                        dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                        res_index = np.hstack((res_index, tmp_index[:, np.newaxis]))
                    return res_dist, res_index
            else:
                count, mod = divmod(x.shape[0], self.test_block_size)
                if mod > 0:
                    count += 1
                if not return_distance:
                    x_block = x[0:self.test_block_size, :]
                    if self.metric == 'euclidean':
                        dist_matrix = distances.euclidean_distance(x_block, self.training_data)
                    else:
                        dist_matrix = distances.cosine_distance(x_block, self.training_data)
                    res_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    np.argmin(dist_matrix, axis=1, out=res_index)
                    dist_matrix[np.arange(dist_matrix.shape[0]), res_index] = np.inf
                    res_index = res_index.reshape((-1, 1))
                    for j in range(self.k - 1):
                        np.argmin(dist_matrix, axis=1, out=tmp_index)
                        dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                        res_index = np.hstack((res_index, tmp_index[:, np.newaxis]))
                    for i in range(1, count):
                        x_block = x[i*self.test_block_size:(i+1)*self.test_block_size, :]
                        if self.metric == 'euclidean':
                            dist_matrix = distances.euclidean_distance(x_block, self.training_data)
                        else:
                            dist_matrix = distances.cosine_distance(x_block, self.training_data)
                        res_index_tmp = np.empty(dist_matrix.shape[0], dtype=np.int64)
                        tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                        np.argmin(dist_matrix, axis=1, out=res_index_tmp)
                        dist_matrix[np.arange(dist_matrix.shape[0]), res_index_tmp] = np.inf
                        res_index_tmp = res_index_tmp.reshape((-1, 1))
                        for j in range(self.k - 1):
                            np.argmin(dist_matrix, axis=1, out=tmp_index)
                            dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                            res_index_tmp = np.hstack((res_index_tmp, tmp_index[:, np.newaxis]))
                        res_index = np.vstack((res_index, res_index_tmp))
                    return res_index
                else:
                    x_block = x[0:self.test_block_size, :]
                    if self.metric == 'euclidean':
                        dist_matrix = distances.euclidean_distance(x_block, self.training_data)
                    else:
                        dist_matrix = distances.cosine_distance(x_block, self.training_data)
                    res_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                    np.argmin(dist_matrix, axis=1, out=res_index)
                    res_dist = dist_matrix[np.arange(dist_matrix.shape[0]), res_index]
                    dist_matrix[np.arange(dist_matrix.shape[0]), res_index] = np.inf
                    res_dist = res_dist.reshape((-1, 1))
                    res_index = res_index.reshape((-1, 1))
                    for j in range(self.k - 1):
                        np.argmin(dist_matrix, axis=1, out=tmp_index)
                        res_dist = np.hstack((res_dist,
                                              dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index][:, np.newaxis]))
                        dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                        res_index = np.hstack((res_index, tmp_index[:, np.newaxis]))
                    for i in range(1, count):
                        x_block = x[i * self.test_block_size:(i + 1) * self.test_block_size, :]
                        if self.metric == 'euclidean':
                            dist_matrix = distances.euclidean_distance(x_block, self.training_data)
                        else:
                            dist_matrix = distances.cosine_distance(x_block, self.training_data)
                        res_index_tmp = np.empty(dist_matrix.shape[0], dtype=np.int64)
                        tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                        np.argmin(dist_matrix, axis=1, out=res_index_tmp)
                        res_dist_tmp = dist_matrix[np.arange(dist_matrix.shape[0]), res_index_tmp]
                        dist_matrix[np.arange(dist_matrix.shape[0]), res_index_tmp] = np.inf
                        res_dist_tmp = res_dist_tmp.reshape((-1, 1))
                        res_index_tmp = res_index_tmp.reshape((-1, 1))
                        for j in range(self.k - 1):
                            np.argmin(dist_matrix, axis=1, out=tmp_index)
                            res_dist_tmp = np.hstack((res_dist_tmp, dist_matrix[np.arange(dist_matrix.shape[0]),
                                                                                tmp_index][:, np.newaxis]))
                            dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                            res_index_tmp = np.hstack((res_index_tmp, tmp_index[:, np.newaxis]))
                        res_index = np.vstack((res_index, res_index_tmp))
                        res_dist = np.vstack((res_dist, res_dist_tmp))
                    return res_dist, res_index
        else:
            return self.data.kneighbors(x, return_distance=return_distance)

    def predict(self, x):
        vec_fun = np.vectorize(lambda z: self.training_labels[z])
        if self.weights:
            vec_weight = np.vectorize(lambda z: 1 / (z + 0.00001) if z > 0.0 else 0.0)
            nn_dist, nn_index = self.find_kneighbors(x)
            nn_index = vec_fun(nn_index)
            y_train_unique = np.unique(self.training_labels)
            res_labels = np.sum(vec_weight((nn_index == y_train_unique[0]) * nn_dist), axis=1).reshape((-1, 1))
            for i in y_train_unique[1:]:
                res_labels = np.hstack((res_labels,
                                        np.sum(vec_weight((nn_index == i) * nn_dist), axis=1)[:, np.newaxis]))
            return y_train_unique[np.argmax(res_labels, axis=1)]
        else:
            nn_index = vec_fun(self.find_kneighbors(x, return_distance=False))
            return np.apply_along_axis(lambda z: np.unique(z)[np.argmax(np.unique(z, return_counts=True)[1])],
                                       axis=1, arr=nn_index)
