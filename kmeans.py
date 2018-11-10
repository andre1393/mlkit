import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def kmeans(df, k = 2, max_iter = 5):
    df['cluster'] = 1
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
            for var in list(filter(lambda x: x != 'index', list(cluster))):
                cluster.loc[i_cluster, var] = df[df['cluster'] == i_cluster][var].mean()

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
    
    cols = list(filter(lambda x: x != 'index' and x != 'cluster', list(cluster)))
    
    for i in range(len(cols) - 1):
        for j in range(i + 1,len(cols)):
            fig, ax = plt.subplots()
            for p in range(k):                
                #plt.scatter(df[df['cluster'] == p]['x'], df[df['cluster'] == p]['y'], color = color[p], s = 10)
                #plt.scatter(cluster.loc[p, 'x'], cluster.loc[p, 'y'], color = color[p], marker='+', s = 1000)
                ax.scatter(df[df['cluster'] == p].iloc[:,i], df[df['cluster'] == p].iloc[:,j], color = color[p], s = 10)
                ax.scatter(cluster.loc[p, cols[i]], cluster.loc[p, cols[j]], color = color[p], marker='+', s = 1000)
                ax.set_title('Gráfico de {} versus {}'.format(cols[i], cols[j]))
                ax.set_xlabel('{}'.format(cols[i]))
                ax.set_ylabel('{}'.format(cols[j]))
                ax.grid(True)

def main():
    x = np.random.normal(1, 0.1, 99) * np.repeat([5, 10, 15], 33)
    y = np.random.normal(1, 0.1, 99) * np.repeat([5, 10, 15], 33)
    z = np.random.normal(1, 0.1, 99) * np.repeat([7, 5, 4], 33)
    
    df = pd.DataFrame({'x':x, 'y':y})
    df, cluster = kmeans(df, 3)
    plot_kmeans(df, cluster)
