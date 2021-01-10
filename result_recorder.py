import pandas as pd


class Recorder:
	def __init__(self, data):
		self.data = data
	def record(self, filename):
		movies = []

		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			properties = {}
			properties['id'] = movie['id']
			properties['label'] = movie['label']
			movies += [properties]

		movie_df = pd.DataFrame(movies)
		movie_df.to_csv(filename, sep='\t')		


