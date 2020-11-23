import numpy as np
from . import nearest_neighbors as nn
from skimage import transform, filters
from . image_shift import image_shift


def kfold(n, n_folds):
    if not isinstance(n, int) or not isinstance(n_folds, int) or n < 1 or n_folds < 2:
        raise AttributeError('Incorrect parameters')

    index_list = []
    d = n // n_folds
    for i in range(n_folds - 1):
        arr_val = np.arange(i * d, (i + 1) * d)
        arr_train = np.hstack((np.arange(0, i * d), np.arange((i + 1) * d, n)))
        index_list.append((arr_train, arr_val))
    index_list.append((np.arange((n_folds - 1) * d), np.arange((n_folds - 1) * d, n)))
    return index_list


def knn_cross_val_score_aug(x, y, k_list, score='accuracy', cv=None, **kwargs):

    def fun_count(z):
        d = {}
        max_count = 0
        counter = 0
        res = []
        for num, item in enumerate(z):
            if item in d:
                d[item] += 1
            else:
                d[item] = 1
            if d[item] > max_count:
                max_count = d[item]
                max_class = item
            if num == k_list[counter] - 1:
                counter += 1
                res.append(max_class)
        return np.array(res)

    def fun_count_weights(z):
        nonlocal k1
        k1 += 1
        line_l = np.array([])
        for cls in y_train_unique:
            line_cls = np.array([])
            counter = 0
            summer = 0
            for num2, row in enumerate(z):
                if row == cls:
                    summer += 1 / (nn_dist[k1, num2] + 0.00001)
                if num2 == k_list[counter] - 1:
                    counter += 1
                    line_cls = np.append(line_cls, summer)
            if line_l.shape[0] == 0:
                line_l = line_cls
            else:
                line_l = np.vstack((line_l, line_cls[np.newaxis, :]))
        return y_train_unique[np.argmax(line_l, axis=0)]

    x = np.hstack((x, np.arange(x.shape[0])[:, np.newaxis]))
    np.random.shuffle(x)
    y = y[x[:, -1].astype(int)]
    x = x[:, 0:-1]

    knn = nn.KNNClassifier(k_list[-1], **kwargs)
    if cv is None:
        cv = kfold(x.shape[0], 3)
    res_dict = dict.fromkeys(k_list)
    for i in res_dict:
        res_dict[i] = np.empty(len(cv))
    for fold_num, fold in enumerate(cv):

        # x_train_sh_left_x = np.apply_along_axis(lambda z: image_shift(z.reshape((28, 28)), x=-1).ravel(), axis=1,
        #                                         arr=x[fold[0]])
        # x_train_sh_right_x = np.apply_along_axis(lambda z: image_shift(z.reshape((28, 28)), x=1).ravel(), axis=1,
        #                                          arr=x[fold[0]])
        # x_train_sh_down_y = np.apply_along_axis(lambda z: image_shift(z.reshape((28, 28)), y=-1).ravel(), axis=1,
        #                                         arr=x[fold[0]])
        # x_train_sh_up_y = np.apply_along_axis(lambda z: image_shift(z.reshape((28, 28)), y=1).ravel(), axis=1,
        #                                       arr=x[fold[0]])
        x_train_rot_left = np.apply_along_axis(lambda z: transform.rotate(z.reshape((28, 28)), 5).ravel(), axis=1,
                                               arr=x[fold[0]])
        x_train_rot_right = np.apply_along_axis(lambda z: transform.rotate(z.reshape((28, 28)), -5).ravel(), axis=1,
                                                arr=x[fold[0]])
        x_train_bl_tmp = np.apply_along_axis(lambda z: filters.gaussian(z.reshape((28, 28)), sigma=1.0).ravel(),
                                             axis=1, arr=x[fold[0]])
        x_train_mix = np.vstack((x[fold[0]], x_train_bl_tmp, x_train_rot_left, x_train_rot_right))
        y_train_mix = np.tile(y[fold[0]], 4)

        knn.fit(x_train_mix, y_train_mix)
        vec_fun = np.vectorize(lambda z: y_train_mix[z])
        if kwargs.get('weights', False):
            nn_dist, nn_index = knn.find_kneighbors(x[fold[1]], return_distance=True)
            nn_index = vec_fun(nn_index)
            y_train_unique = np.unique(y_train_mix)

            k1 = -1
            res = np.apply_along_axis(fun_count_weights, axis=1, arr=nn_index)

            if score == 'accuracy':
                for num, line in enumerate(res.T):
                    res_dict[k_list[num]][fold_num] = np.sum(line == y[fold[1]]) / line.shape[0]
        else:
            nn_index = vec_fun(knn.find_kneighbors(x[fold[1]], return_distance=False))
            res = np.apply_along_axis(fun_count, axis=1, arr=nn_index)
            if score == 'accuracy':
                for num, line in enumerate(res.T):
                    res_dict[k_list[num]][fold_num] = np.sum(line == y[fold[1]]) / line.shape[0]
    return res_dict
