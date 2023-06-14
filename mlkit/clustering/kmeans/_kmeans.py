"""
K-Means implementation
"""
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mlkit.clustering.base_clustering import BaseClustering
from mlkit.decorators import validate_model_fitted


class KMeans(BaseClustering):

    def __init__(
            self,
            k,
            max_iter=50,
            min_delta_iter=0.01,
            distance_calc_method=None,
            seed=None,
            colors=None,
            step_by_step=False
    ):
        """
        Class constructor

        :param k: number of clusters. Can be either an int (# of clusters) or a
         list (each item represent the center of each cluster)
        :param max_iter: max iterations
        :param min_delta_iter: min difference between iteration to consider the training converged
        :param seed: seed value. If not provided, seed won't be set
        :param colors: list of cluster colors, used for graph plot
        :param step_by_step: Rather the execution should run step by step, plotting the intermediate cluster assignment.
                             This is useful for debugging
        """
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.min_delta_iter = min_delta_iter
        self.seed = seed
        self.data = None
        self.labels_ = None
        self.n_iter_ = None
        self._calc_dist = distance_calc_method or self._euclidian_distance
        self.step_by_step = step_by_step

        if isinstance(k, list):
            self.initial_cluster = k
            self.cluster_centers_ = k
            self.k = len(k)
        else:
            self.k = k
            self.cluster_centers_ = None
            self.initial_cluster = None
            
        if seed is not None:
            np.random.seed(seed)
        
        default_colors = ['blue', 'black', 'red', 'green', 'purple', 'gray', 'orange', 'brown']
        while k > len(default_colors):
            default_colors += default_colors
        
        self.colors = np.random.choice(default_colors, k, replace=False) if colors is None else colors

    @validate_model_fitted
    def predict(self, features):
        """
        get the closest cluster for each row in `features` dataframe

        :param features: features dataframe to apply the prediction
        :return: pandas series with the respective cluster for each row
        """
        return features.apply(self._get_closest_cluster, axis=1)

    @validate_model_fitted
    def transform(self, features):
        """
        Transform the `features` dataframe into an n_samples by n_clusters, where each value represents the distance of
        the cluster distance from the sample

        :param features: features dataframe
        :return: pandas dataframe where rows=samples and columns=clusters
        """
        return features.apply(lambda row: pd.Series(self._get_clusters_distances(row)), axis=1)
    
    def _fit(self, df):
        """
        Method to fit the model

        :param df: Dataframe to train with
        :return: dataset labels
        """
        process_init = time.time()
        self.data = df.copy()

        self.data['cluster'] = np.random.choice(range(self.k), len(self.data))
        
        if self.cluster_centers_ is None:
            self.cluster_centers_ = self.data.sample(self.k).drop('cluster', axis=1).reset_index(drop=True)
            self.initial_cluster = self.cluster_centers_
        
        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            if self.step_by_step:
                self.plot(iteration=it)

            self.data['cluster'] = self.data.drop('cluster', axis=1).apply(self._get_closest_cluster, axis=1)

            delta_diff = self._calculate_delta_cluster_center(self.data)
            self.cluster_centers_ = self.data.groupby('cluster').mean()

            if delta_diff <= self.min_delta_iter:
                print(f'The clusters centers has converged. The total difference between two iterations: {delta_diff}')
                break
            if it == self.max_iter:
                print(f'max iterations reached. {self.max_iter}')

        print(f'time_spent: {time.time() - process_init}')

        self.labels_ = self.data['cluster']

        print(f'kmeans trained in {self.n_iter_} interactions for k = {self.k}')

        return self.labels_

    def _calculate_delta_cluster_center(self, df):
        """
        Calculates the total delta difference of the new cluster centers after iteration.
        Example:
        ::
            old cluster centers:  new cluster centers:
            a | b                       a | b\n
            1 | 2                       1 | 3\n
            3 | 4                       2 | 5\n

        In this case, the absolute difference would be
        ::
            a | b
            0 | 1
            1 | 1
        And after applying sum() twice, it will first sum the columns [1, 2] and then sum again resulting 3

        :param df: dataframe
        :return: the sum of all delta difference between the old and new cluster centers
        """
        new_cluster_centers = df.groupby('cluster').mean()
        return abs(self.cluster_centers_ - new_cluster_centers).sum().sum()

    def _get_clusters_distances(self, row):
        """
        Calculates the distance from each cluster center for the specific row

        :param row: pandas Series row from the features dataframe
        :return: list of distances from each cluster
        """
        return [self._calc_dist(row.values, cluster_center) for cluster_center in self.cluster_centers_.values]

    def _get_closest_cluster(self, row):
        """
        Get the closest cluster id from the row

        :param row: pandas series row from features dataframe
        :return: cluster id
        """
        return np.array(self._get_clusters_distances(row)).argmin()

    @staticmethod
    def _euclidian_distance(row, cluster):
        """
        Calculates the euclidian distance of the row from the cluster center

        :param row: pandas series row from the features dataframe
        :param cluster: cluster center represented by a list
        :return: the euclidian distance from the row to the cluster center
        """
        return np.sqrt(sum((row-cluster)**2))

    def plot(self, axis=None, grid=True, fig_size=None, iteration='', columns=None):
        """
        Plot the dataset along with the cluster centers

        :param axis: can be used to set some axis properties.
                     https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib-pyplot-axis
        :param grid: Rather the plot should have grids or not.
        :param fig_size: plot figure size
        :param iteration: this param might be useful to highlight the iteration if the plot method
                          is called before the end.
        :param columns: columns to plot
        :return:
        """
        cols = columns or list(filter(lambda x: x != 'index' and x != 'cluster', self.data.columns))

        if len(cols) not in [2, 3]:
            raise ValueError(
                f'It is not possible to plot a {len(cols)} dimensions data'
                '. Please, provide a subset o columns with size 2 or 3'
            )
        projection = None if len(cols) == 2 else '3d'

        fig_size = fig_size or (13, 9)
        k = len(self.cluster_centers_)

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(projection=projection)
        for p in range(k):
            ax.scatter(
                *[self.data[self.data['cluster'] == p][axis] for axis in cols],
                color=self.colors[p], s=10
            )
            ax.scatter(
                *[self.cluster_centers_.loc[p, :][c] for c in cols],
                color=self.colors[p],
                marker='+',
                s=1000
            )

            ax.set_title(f'{" vs ".join(cols)}')
            ax.set_xlabel(f'{cols[0]}')
            ax.set_ylabel(f'{cols[1]}')
            ax.grid(grid)
            ax.axis(axis)
        plt.show()

    @property
    def inertia_(self):
        return sum([
            sum((item - self.cluster_centers_.iloc[label, :].values)**2)
            for item, label in zip(self.data.drop('cluster', axis=1).values, self.labels_.values)
        ])
