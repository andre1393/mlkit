import numpy as np
import pandas as pd
from nlp.bayesian_sets import BayesianSets
import sys
import re

def main(corpus_file, column):
	df = pd.read_csv(corpus_file)

	# remove release year from movie title
	df['title'] = df['title'].apply(lambda x: re.sub("\\([0-9]+\\)", "", x))
	bs = BayesianSets()

	user_input = ""
	while user_input != "fim!":
		user_input = input('query: ')

		s = bs.search_query([user_input] , df[column])

		data = {'title': df['title'], 'value': np.asarray(s).tolist()}
		df = pd.DataFrame(data)
		df['value'] = df['value'].apply(lambda x: x[0])
		df.sort_values('value', ascending = False).iloc[0:30, :]

if __name__ == "__main__":
	#~/github/data/datasets/movies.csv
    corpus_file = sys.argv[1]
    column = sys.argv[2]
    main(corpus_file, column)