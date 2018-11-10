import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class Kmeans:
    
    def __init__(self, k, max_iter = 50, delta_iter = 0):
        self.k = k
        self.max_iter = max_iter
        self.delta_iter = delta_iter
        
    def fit(self, df):
        df['cluster'] = 1
        ridx = np.random.choice(range(1,len(df)), self.k)
        cluster = df.loc[ridx,].reset_index()
        for i in range(0, self.max_iter):
            for i_row, row in df.iterrows():
                min_dist = math.inf
                nearst = -1
                for i_cluster, c in cluster.iterrows():
                    c_dist = calc_dist(row, c)
                    if(c_dist < min_dist):
                        min_dist = c_dist
                        df.loc[i_row, 'cluster'] = i_cluster

            delta_cluster = np.array([])
            for i_cluster, c in cluster.iterrows():
                delta_var = np.array([])
                for var in list(filter(lambda x: x != 'index', list(cluster))):
                    old = cluster.loc[i_cluster, var]
                    cluster.loc[i_cluster, var] = df[df['cluster'] == i_cluster][var].mean()
                    delta_var = np.append(delta_var, cluster.loc[i_cluster, var] - old)
                delta_cluster = np.append(delta_cluster, delta_var)

            if sum(abs(delta_cluster)) <= self.delta_iter:
                print('treinamento convergiu')
                break

        print('kmeans treinado em {} iteracoes'.format(i + 1))

        return df, cluster

    def calc_dist(self, row, cluster):
        r = row.drop('cluster').tolist()
        c = cluster.drop('cluster').drop('index').tolist()
        return np.sqrt(sum(map(lambda x, y: (x - y)**2, r, c)))

    def plot_kmeans(self, df, cluster, color = None):
        k = len(cluster)

        if color == None:
            colors = ['blue', 'black', 'red', 'green', 'yellow', 'purple', 'gray']
            color = np.random.choice(colors, k, replace = False)

        cols = list(filter(lambda x: x != 'index' and x != 'cluster', list(cluster)))

        for i in range(len(cols) - 1):
            for j in range(i + 1,len(cols)):
                fig, ax = plt.subplots()
                for p in range(k):                
                    ax.scatter(df[df['cluster'] == p].iloc[:,i], df[df['cluster'] == p].iloc[:,j], color = color[p], s = 10)
                    ax.scatter(cluster.loc[p, cols[i]], cluster.loc[p, cols[j]], color = color[p], marker='+', s = 1000)
                    ax.set_title('Gráfico de {} versus {}'.format(cols[i], cols[j]))
                    ax.set_xlabel('{}'.format(cols[i]))
                    ax.set_ylabel('{}'.format(cols[j]))
                    ax.grid(True)

    def fit_example(self, ncols = 2):
        df = pd.DataFrame()
        for i in range(ncols):
            df[str(i)] = np.random.normal(i, 0.2, 99) * np.repeat([5, 10, 15], 33)

        df, cluster = self.fit(df)
        plot_kmeans(df, cluster)
