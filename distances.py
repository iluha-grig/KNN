import numpy as np


def euclidean_distance(x, y):
    sqr_x = np.sum(x ** 2, axis=1)
    sqr_y = np.sum(y ** 2, axis=1)
    prod = np.matmul(x, y.T)
    return np.sqrt(sqr_x[:, np.newaxis] - 2 * prod + sqr_y)


def cosine_distance(x, y):
    prod = np.matmul(x, y.T)
    norm_x = np.sqrt(np.sum(x ** 2, axis=1))
    norm_y = np.sqrt(np.sum(y ** 2, axis=1))
    prod_norm = np.matmul(norm_x[:, np.newaxis], norm_y[np.newaxis, :])
    return np.ones((x.shape[0], y.shape[0])) - prod / prod_norm
