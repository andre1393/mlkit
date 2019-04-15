import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

class KMeans:

    def __init__(self, k, max_iter = 50, delta_iter = 0, seed = None, color = None):
        self.k = k
        self.max_iter = max_iter
        self.delta_iter = delta_iter
        self.seed = seed
        
        if type(k) == list:
            self.initial_cluster = k
            self.cluster_centers
            self.k = len(k)
        else:
            self.k = k
            self.cluster_centers = None
            self.initial_cluster = None
            
        if seed != None:
            np.random.seed(seed)
        
        colors = ['blue', 'black', 'red', 'green', 'yellow', 'purple', 'gray', 'orange', 'brown', 'gold']
        while k > len(colors):
            colors += colors
        
        self.color = np.random.choice(colors, k, replace = False) if color == None else color
            
    def fit(self, df, step_by_step = False):
        # setando o data set usado no treinamento
        self.data = df
        
        # setando se quer executar o algoritmo passo a passo
        self.step_by_step = step_by_step
        
        process_init = time.time()
        df['cluster'] = 1
        
        if self.cluster_centers == None:
            ridx = np.random.choice(range(1,len(df)), self.k)
            self.cluster_centers = df.loc[ridx,].reset_index().iloc[:,1:]
        
            # setando o cluster inicial
            self.initial_cluster = self.cluster_centers.iloc[:,1:]
        
        for i in range(0, self.max_iter):
            if self.step_by_step:
                self.plot()
            for i_row, row in df.iterrows():
                min_dist = math.inf
                nearst = -1
                for i_cluster, c in self.cluster_centers.iterrows():
                    c_dist = self.calc_dist(row, c)
                    if(c_dist < min_dist):
                        min_dist = c_dist
                        df.loc[i_row, 'cluster'] = int(i_cluster)

            delta_cluster = np.array([])
            for i_cluster, c in self.cluster_centers.iterrows():
                delta_var = np.array([])
                for var in list(filter(lambda x: x != 'index', list(self.cluster_centers))):
                    old = self.cluster_centers.loc[i_cluster, var]
                    self.cluster_centers.loc[i_cluster, var] = df[df['cluster'] == i_cluster][var].mean()
                    delta_var = np.append(delta_var, self.cluster_centers.loc[i_cluster, var] - old)
                delta_cluster = np.append(delta_cluster, delta_var)
                
            if sum(abs(delta_cluster)) <= self.delta_iter:
                print('treinamento convergiu')
                break

        # setando o numero de iteracoes executadas
        self.n_iter = i
        
        # setando o tempo de processamento
        self.time_processing = time.time() - process_init
        
        # setando os labels de cada ponto 
        self.labels = df['cluster']
        
        print('kmeans treinado em {} iteracoes para k = {}'.format(i + 1, self.k))

        return df, self.cluster_centers

    def calc_dist(self, row, cluster):
        r = row.drop('cluster').tolist()
        c = cluster.drop('cluster').tolist()
        return np.sqrt(sum(map(lambda x, y: (x - y)**2, r, c)))

    def plot(self, axis = 'equal', grid = True, figsize = (13,9)):
        k = len(self.cluster_centers)
        cols = list(filter(lambda x: x != 'index' and x != 'cluster', list(self.cluster_centers)))

        for i in range(len(cols) - 1):
            for j in range(i + 1,len(cols)):
                fig, ax = plt.subplots(figsize = figsize)
                for p in range(k):
                    ax.scatter(self.data[self.data['cluster'] == p].iloc[:,i], self.data[self.data['cluster'] == p].iloc[:,j], color = self.color[p], s = 10)
                    ax.scatter(self.cluster_centers.loc[p, cols[i]], self.cluster_centers.loc[p, cols[j]], color = self.color[p], marker='+', s = 1000)
                    ax.set_title('Gráfico de {} versus {}'.format(cols[i], cols[j]))
                    ax.set_xlabel('{}'.format(cols[i]))
                    ax.set_ylabel('{}'.format(cols[j]))
                    ax.grid(grid)
                    ax.axis(axis)
        plt.show()
    
    @property
    def mean_dist(self):
        result = 0
        for i, item in self.data.iterrows():
            result += self.calc_dist(item, self.cluster_centers.iloc[self.labels[i],:])
        return result

def elbow_method(df, ks = range(1,10), plot = True):
    process_init = time.time()
    ev = []
    for k in ks:
        model = KMeans(k)
        model.fit(df)
        ev.append(model.mean_dist)
        
    print('tempo total de todos os treinamentos: {}s'.format(time.time() - process_init))
    
    if plot:
        plt.plot(ks, ev)
        plt.show()
        
    return ks, ev