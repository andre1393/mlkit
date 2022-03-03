import pandas as pd
from mlkit.classification.decision_tree import DecisionTree


def example_tennis_categorical():
	target = 'target'
	df = pd.read_csv('examples/datasets/play_tennis.csv')
	model = DecisionTree()

	model.fit(df.drop(target, axis=1), df[target])
	x_test = pd.DataFrame({'Outlook': ['Sunny'], 'Temperature': ['Hot'], 'Humidity': ['Normal'], 'Wind': ['Weak']})
	print(model.predict(x_test))


if __name__ == '__main__':
	example_tennis_categorical()
