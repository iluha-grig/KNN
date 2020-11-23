import numpy as np


def image_shift(image, x=None, y=None):
    if x is not None and y is None:
        trans = np.zeros((28, abs(x)))
        if x > 0:
            return np.hstack((trans, image[:, 0:-x]))
        elif x < 0:
            return np.hstack((image[:, -x:], trans))
        else:
            return image
    elif x is None and y is not None:
        trans = np.zeros((abs(y), 28))
        if y > 0:
            return np.vstack((image[y:], trans))
        elif y < 0:
            return np.vstack((trans, image[0:y]))
        else:
            return image
    else:
        raise AttributeError('x is None and y is None')
