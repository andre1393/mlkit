import numpy as np
import pandas as pd
from Node import Node

class DecisionTree:

	def fit(self, df, classe):
		self.tree = self.gain(df, classe, None)

	def gain(self, df, classe, link):
		arr = df[classe].value_counts().values
		p = arr/arr.sum()
		s = sum(-1*p*np.log2(p))
		
		entropy_s = self.entropyS(df[classe])
		sa = []
		max_gain = ('', -float("inf"))
		for column in df.columns:
			if column != classe:
				gain_value = entropy_s
				for key, value in df[column].value_counts().items():
					gain_value -= self.get_proportion(df[column])[key][0] * self.entropyS(df[df[column] == key][classe])
		
				if gain_value > max_gain[1]:
						max_gain = (column, gain_value)
		
		if df.columns.size <= 1:
			counts = df.loc[:,classe].value_counts()
			return Node('value', [counts.keys()[0]], df.columns[0])
		elif max_gain[1] <= 0:
			counts = df.loc[:,classe].value_counts()
			return Node('value', [counts.keys()[0]], df.columns[0])
		
		node = Node(link, None, max_gain[0])
		for key, value in df[max_gain[0]].value_counts().items():
			children = self.gain(df[df[max_gain[0]] == key].drop(max_gain[0], axis = 1), classe, key)
			node.add_node(key, children)
		
		return node

	def get_proportion(self, column):
		counts = column.value_counts()
		counts = counts/counts.sum()
		return pd.DataFrame(counts).pivot_table(columns = counts.keys()).reset_index(drop = True)
		
	def entropyS(self, column):
		arr = column.value_counts().values
		p = arr/arr.sum()
		s = sum(-1*p*np.log2(p))
		return s

	def predict(self, x):
		t = self.tree
		while len(t.children) > 1:
			t = t.children[x[t.attribute_name]]
		
		return t.children['value'][0]