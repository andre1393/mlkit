import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit(df = pd.DataFrame(np.transpose(np.array([np.random.normal(1, 0.1, 99) * np.repeat([5, 6, 7], 33), np.random.normal(1, 0.1, 99) * np.repeat([5, 6, 7], 33)])), columns = ['x', 'y']), k = 2, max_iter = 50):
	colors = ['blue', 'black', 'red', 'green', 'yellow', 'purple', 'gray']
	color = np.random.choice(colors, k, replace = False)


	ridx = np.random.choice(range(1,len(x)), k)

	cluster = df.loc[ridx,].reset_index()

	for i in range(0, max_iter):
		for i_row, row in df.iterrows():
			min_dist = math.inf
			nearst = -1
			for i_cluster, c in cluster.iterrows():
				c_dist = np.sqrt((row['x'] - c['x'])**2 + (row['y'] - c['y'])**2)
				if(c_dist < min_dist):
					min_dist = c_dist
					df.loc[i_row, 'cluster'] = i_cluster

		for i_cluster, c in cluster.iterrows():
			cluster.loc[i_cluster,'x'] = df[df['cluster'] == i_cluster]['x'].mean()
			cluster.loc[i_cluster,'y'] = df[df['cluster'] == i_cluster]['y'].mean()
			
	return df, cluster
		
def plot_kmeans(df, cluster, color):
	for i in range(k):
		print(color[i])
		plt.scatter(df[df['cluster'] == i]['x'], df[df['cluster'] == i]['y'], color = color[i], s = 10)
		plt.scatter(cluster.loc[i, 'x'], cluster.loc[i, 'y'], color = color[i], marker='+', s = 1000)
		
def fit():
	x = np.random.normal(1, 0.1, 99) * np.repeat([5, 6, 7], 33)
	y = np.random.normal(1, 0.1, 99) * np.repeat([5, 6, 7], 33)

	
	df['cluster'] = 1
	
	fit(df, 5)