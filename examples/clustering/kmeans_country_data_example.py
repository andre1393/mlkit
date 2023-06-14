import pandas as pd
from mlkit.clustering.kmeans import KMeans
import matplotlib.pyplot as plt
import time


def kmeans_example(df, k):
    model = KMeans(k, max_iter=1000, step_by_step=False)
    print(model.fit_transform(df))
    model.plot(columns=['income', 'inflation', 'life_expec'])


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
    data = pd.read_csv('../datasets/Country-data.csv')
    elbow_method(data.drop('country', axis=1))
    kmeans_example(data.drop('country', axis=1), 4)


