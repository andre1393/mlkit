import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def kmeans(df, k = 2, max_iter = 5):
    ridx = np.random.choice(range(1,len(df)), k)
    cluster = df.loc[ridx,].reset_index()
    for i in range(0, max_iter):
        for i_row, row in df.iterrows():
            min_dist = math.inf
            nearst = -1
            for i_cluster, c in cluster.iterrows():
                c_dist = calc_dist(row, c)
                if(c_dist < min_dist):
                    min_dist = c_dist
                    df.loc[i_row, 'cluster'] = i_cluster
        for i_cluster, c in cluster.iterrows():
            cluster.loc[i_cluster,'x'] = df[df['cluster'] == i_cluster]['x'].mean()
            cluster.loc[i_cluster,'y'] = df[df['cluster'] == i_cluster]['y'].mean()

    return df, cluster

def calc_dist(row, cluster):
    r = row.drop('cluster').tolist()
    c = cluster.drop('cluster').drop('index').tolist()
    return np.sqrt(sum(map(lambda x, y: (x - y)**2, r, c)))

def plot_kmeans(df, cluster, color = None):
    k = len(cluster)
    
    if color == None:
        colors = ['blue', 'black', 'red', 'green', 'yellow', 'purple', 'gray']
        color = np.random.choice(colors, k, replace = False)
        
    for i in range(k):
        plt.scatter(df[df['cluster'] == i]['x'], df[df['cluster'] == i]['y'], color = color[i], s = 10)
        plt.scatter(cluster.loc[i, 'x'], cluster.loc[i, 'y'], color = color[i], marker='+', s = 1000)

def main():
    x = np.random.normal(1, 0.1, 99) * np.repeat([4, 7, 10], 33)
    y = np.random.normal(1, 0.1, 99) * np.repeat([2, 3, 4], 33)
    z = np.random.normal(1, 0.1, 99) * np.repeat([2, 3, 4], 33)
    
    df = pd.DataFrame({'x':x, 'y':y, 'z': z})
    df['cluster'] = 1
    df, cluster = kmeans(df, 2)
    plot_kmeans(df, cluster)
