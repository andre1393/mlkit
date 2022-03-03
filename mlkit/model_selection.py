import numpy as np


def train_test_split(x, y, test_size=0.3):
    size = len(x)
    test_index = x.index.isin(np.sort(np.random.choice(range(len(x)), int(np.round(size*test_size)), replace=False)))
    x_train = x[~test_index]
    x_test = x[test_index]
    y_train = y[~test_index]
    y_test = y[test_index]
    return x_train, x_test, y_train, y_test
