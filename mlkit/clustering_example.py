import numpy as np
import pandas as pd
from clustering.kmeans.KMeans import KMeans, elbow_method
import matplotlib.pyplot as plt

def elbow_method_example(df):
    elbow_method(df)

def kmeans_example(df, k):
    model = KMeans(k)
    model.fit(df)
    model.plot()

def random_dataset():
    df = pd.DataFrame()
    ncols = 2
    n_sample = 99
    k = 3
    for i in range(ncols):
        df[str(i)] = np.random.normal(i + 1, 0.2, n_sample) * np.repeat([5, 10, 15], int(n_sample)/k)
    return df

df = random_dataset()
kmeans_example(df, 3)
elbow_method(df)