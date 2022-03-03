"""
Decision Tree implementation using ID3 algorithm

References:
	https://towardsdatascience.com/decision-trees-for-classification-id3-algorithm-explained-89df76e72df1
	https://iq.opengenus.org/id3-algorithm/
	https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html

"""
from typing import Union, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from mlkit.classification.base_classifier import BaseClassifier
from mlkit.classification.decision_tree._node import Node
from mlkit.exceptions import NotFittedException


class DecisionTree(BaseClassifier):

	def __init__(self):
		super().__init__()
		self._tree = None

	def fit(self, x: DataFrame, y: Series) -> None:
		"""
		Uses id3 algorithm to fit a decision tree

		:param x: X dataset (features)
		:param y: y (target)
		:return None
		"""
		self._tree = self._id3(x, y)

	def _id3(self, x: DataFrame, y: Series, parent: Optional[Node] = None) -> Node:
		"""
		Iteratively computes max information for each feature and fit a decision tree

		:param x: X dataset (features)
		:param y: y (target)
		:param parent: parent Node
		:return: Node
		"""

		max_gain = max(
			[(column, self._information_gain(x[column], y)) for column in x.columns], key=lambda d: d[1]
		)

		if (x.columns.size <= 1) or (max_gain[1] <= 0):
			counts = y.value_counts()
			return Node(parent, None, counts.keys()[0])

		node = Node(parent, None, max_gain[0])
		for key, value in x[max_gain[0]].value_counts().items():
			idx = x[max_gain[0]] == key
			children = self._id3(x[idx].drop(max_gain[0], axis=1), y[idx], node)
			node.add_node(key, children)

		return node

	def _information_gain(self, column, y):
		"""
		Given an entropy and Computes information gain

		https://victorzhou.com/blog/information-gain/

		:param column: column to compute IG
		:param y: y (target)
		:return: information gain value
		"""
		return self._entropy(y) - sum([
			(self._get_proportion(column)[key][0] * self._entropy(y[column == key]))
			for key, value in column.value_counts().items()]
		)

	@staticmethod
	def _get_proportion(column: Series) -> DataFrame:
		"""
		Calculates the proportion of each value in a specific column

		Example:
			get_proportion(['a', 'b', 'b', 'a', 'a']) will return the following dataframe::

				a 	| 	b
				0.6	|  0.4


		:param column: column to calculate the proportion
		:return: DataFrame where column is one of values presents in the input, and their respective values are in row 0
		"""
		counts = column.value_counts()
		counts = counts/counts.sum()
		return pd.DataFrame(counts).pivot_table(columns=counts.keys()).reset_index(drop=True)

	@staticmethod
	def _entropy(column: Series) -> float:
		"""
		Computes a column entropy

		https://victorzhou.com/blog/information-gain/

		:param column: column to compute entropy
		:return: entropy value
		"""
		arr = column.value_counts().values
		p = arr/arr.sum()
		s = sum(-1*p*np.log2(p))
		return s

	def predict(self, x: Union[Series, DataFrame]) -> DataFrame:
		"""
		Given a set of features (DataFrame or Series) perform predictions
		:param x: Set of features. DataFrame (for multiple) or Series (for single)
		:return: predictions
		"""
		if not self._tree:
			raise NotFittedException()

		def _predict(_x: Series):
			t = self._tree
			while len(t.children) > 0:
				t = t.children[_x[t.attribute_name]]
			return t.attribute_name

		if isinstance(x, Series):
			return _predict(x)
		elif isinstance(x, DataFrame):
			return x.apply(lambda features: _predict(features), axis=1)

	def predict_proba(self, _):
		raise NotImplementedError()
