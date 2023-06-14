import numpy as np
import pandas as pd
from mlkit.clustering.kmeans import KMeans
import matplotlib.pyplot as plt
import time


def kmeans_example(df, k):
    model = KMeans(k, max_iter=1000, step_by_step=False)
    print(model.fit_transform(df))
    model.plot()


def random_dataset():
    _df = pd.DataFrame()
    n_cols = 3
    n_sample = 99
    k = 3
    for axis, i in zip(['x', 'y', 'z', 'w'][:n_cols], range(n_cols)):
        _df[str(axis)] = np.random.normal(i + 1, 0.5, n_sample) * np.repeat([5, 10, 15], int(n_sample)/k)
    return _df


def elbow_method(df, k_values=None, plot=True):
    k_values = k_values or range(1, 10)
    process_init = time.time()
    inertia_values = []
    for k in k_values:
        model = KMeans(k)
        model.fit(df)
        inertia_values.append(model.inertia_)

    print(f'total elapsed time: {time.time() - process_init}')

    if plot:
        plt.plot(k_values, inertia_values)
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Elbow graph')
        plt.grid()
        plt.show()

    return k_values, inertia_values


if __name__ == '__main__':
    data = random_dataset()
    #elbow_method(data)
    kmeans_example(data, 3)


