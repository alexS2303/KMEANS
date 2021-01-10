import pandas as pd
import json
from KMEANS import distance
import sys

class Data:
	def __init__(self, filepath, init):
		self.data = pd.read_csv(filepath)
		self.init = init
		self.clean_data()
		self.handle_PCA()
		self.normalize()
		self.data.insert(len(self.data.columns), 'label', 0)
		print('\n done preprocessing \n')

	def clean_data(self):
		# Cleans data of NaNs

		self.data.loc[self.data['original_title'] == 'To Be Frank, Sinatra at 100', 'runtime'] = 21
		self.data.loc[self.data['original_title'] == 'To Be Frank, Sinatra at 100', 'revenue'] = 926

		self.data.loc[self.data['original_title'] == 'Chiamatemi Francesco - Il Papa della gente', 'runtime'] = 98
		self.data.loc[self.data['original_title'] == 'Chiamatemi Francesco - Il Papa della gente', 'revenue'] = 3925769


	def handle_PCA(self):
		if self.init == 'PCA':
			#does PCA using scikit-learn module
			def get_total_votes(row):
				return row['vote_count']*row['vote_average']
			
			self.data['total_votes'] = self.data.apply(lambda row: get_total_votes(row), axis=1)
			self.data.sort_values(by=['total_votes'], inplace=True, ascending=False)
			self.data = self.data.head(250)
			self.data = self.data.reset_index(drop=True)


	def normalize(self):
		#Z-Score normalization for appropriate variables, mapped to (0, 1)
		to_normalize = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
		if self.init == 'PCA': to_normalize += ['total_votes']
		for attribute in to_normalize:
		    mu = self.data[attribute].mean()
		    sig = self.data[attribute].std()
		    self.data[attribute] = pd.Series([(x - mu) / (sig) for x in self.data[attribute]])
		    max_val = max(self.data[attribute])
		    min_val = min(self.data[attribute])
		    
		    # Map all data to [0, 1]
		    self.data[attribute] = pd.Series([(x - min_val) / (max_val - min_val) for x in self.data[attribute]])


	def get_data(self):
		return self.data
















"""

class Data:
	def __init__(self, filepath):
		# Read and normalize the data, prepare for return
		self.data = pd.read_csv(filepath) # Use pandas to read file, define as data frame
		self.parse_json()
		self.clean_data()
		self.normalize()
		self.data.insert(len(self.data.columns), 'label', 0)
		print('\n done preprocessing \n')

	def parse_json(self):
		# Parses json objects into an attribute array
		json_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
		for column in self.data:
			if column in json_columns:
				for movie_num in range(len(self.data[column])):
					category = json.loads(self.data[column][movie_num])
				
					to_set = ''
					if column == 'production_companies':
						to_set = 'id'
					elif column == 'production_countries':
						to_set = 'iso_3166_1'
					elif column == 'spoken_languages':
						to_set = 'iso_639_1'
					else:
						to_set = 'name'

					names_array = []
					for value in category:
						names_array += [str(value[to_set])]

					self.data.at[movie_num, column] = names_array

	def normalize(self):
		#Z-Score normalization for appropriate variables
		to_normalize = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
		for attribute in to_normalize:
		    mu = self.data[attribute].mean()
		    sig = self.data[attribute].std()
		    self.data[attribute] = pd.Series([(x - mu) / (sig) for x in self.data[attribute]])
		    max_val = max(self.data[attribute])
		    min_val = min(self.data[attribute])
		    # Map all data to [0, 1]
		    self.data[attribute] = pd.Series([(x - min_val) / (max_val - min_val) for x in self.data[attribute]])


	Unecessary  :/

	def set_distances(self):
		# Look up table for distances, makes execution quicker
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			darray = []
			for j in range(self.data.shape[0]):
				other_movie = self.data.iloc[j, :]
				darray += [distance(self.data.columns, movie, other_movie)] #gets distance
			
			print(darray)
			#self.data.at[i, 'distance'] = darray

	def clean_data(self):
		# Cleans data of NaNs
		# THERE ARE TWO F***ING ROWS THAT ARE EMPTY IN SOME PLACES
		# I SPENT 3 HOURS DEBUGGING THIS
		real_values = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			for attribute in real_values:
				if pd.isnull(movie[attribute]):
					if movie['original_title'] == 'Chiamatemi Francesco - Il Papa della gente':
						self.data.at[i, 'revenue'] = 3925769 # found online
						self.data.at[i, 'runtime'] = 98
					elif movie['original_title'] == 'To Be Frank, Sinatra at 100':
						self.data.at[i, 'revenue'] = 926
						self.data.at[i, 'runtime'] = 21
					else:
						print('fuck off mate')
						sys.exit()
					


	def get_data(self):
		return self.data





"""



