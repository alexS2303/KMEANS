import math
import pandas as pd
import random
import sys
from collections import Counter
from numpy.random import choice
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


class KMEANS:
	def __init__(self, k, init, data):
		self.k = k
		self.init = init
		self.data = data
		self.real_values = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
		self.centroids = pd.DataFrame(columns=(self.data.columns))
		if self.init == 'PCA': self.real_values += ['total_votes']

	def run(self):
		#runs k means

		#1D works in a different way than regular K means
		if self.init == '1D':
			self.D1_init()
			return

		#initialize appropriate centroids
		self.init_centroids()
		#assign initial labels to all data
		self.assign_labels()
		#get inital cost
		cost = self.get_cost()
		prev_cost = 100 + cost
		iteration_counter = 0
		convergence_condition = 1
		cost_array = []

		while(convergence_condition < prev_cost - cost):
			print('\n iteration number: ' + str(iteration_counter))
			prev_cost = cost
			self.average_centroids()
			cost = self.get_cost()
			if cost > prev_cost: 
				prev_cost = cost
				break
			self.assign_labels()
			print('\n cost for iteration ' + str(iteration_counter) + ' is ' + str(cost) + '\n')
			iteration_counter += 1

		
		print('\n final cost is: ' + str(cost))

		if self.init == 'PCA':
			self.do_PCA()


	def average_centroids(self):
		# average out each centroid
		self.data.sort_values(by=['label'], inplace=True)
		self.data = self.data.reset_index(drop=True)
		print('\n averaging centroids\n')
		movie_list = []

		for i in range(self.centroids.shape[0]):
			
			#thank god for pandas
			movie_list = self.data.loc[self.data['label'] == i]

			if movie_list.empty:
				continue

			self.centroids.at[i, 'num_assigned'] = movie_list.shape[0]


			for value in self.real_values:
				self.centroids.at[i, value] = movie_list[value].mean()

		for i in range(self.k):
			print('\n Cluster ' + str(i) + ' has ' + str(self.centroids.iloc[i]['num_assigned']) + ' movies assigned \n')


	def init_centroids(self):
		# initializes appropriate centroids
		if self.init == 'random':
			self.random_init()
		elif self.init == 'kmeans++':
			self.kplus_init()
		elif self.init == 'PCA':
			self.PCA_init()

	def random_init(self):
		# randomly picks centroids
		print('\n initializing random kmeans \n')

		for i in range(self.k):
			centroid = self.data.iloc[0, :].copy(deep=True)
			centroid['label'] = i
			for value in self.real_values:
				centroid[value] = random.uniform(0, 1)


			self.centroids = self.centroids.append(centroid, ignore_index=True)

	def kplus_init(self):
		# using kmeans++ procedure
		print('\n initializing kmeans++ \n')
		random_entry = random.randint(0, self.data.shape[0])
		already_selected = [random_entry] # keep list of ids of centroids that were already selected
		prob_dist = []
		
		#draw = choice(list_of_candidates, number_of_items_to_pick,p=probability_distribution)

		for i in range(self.k - 1):
			# keep going until there are k centroids
			distances = []
			for movie_num in range(self.data.shape[0]):
				# if already selected this movie, continue
				if movie_num in already_selected:
					distances += [0] #make sure this isn't selected
					continue
				else:
					distances += [self.find_nearest_centroid_distance(movie_num, already_selected)]

			
			probabilities = [x/sum(distances) for x in distances]

			selected = choice(list(range(0, self.data.shape[0])), 1, p=probabilities)
			already_selected += [selected[0]]


		for i in range(self.k):
			centroid = self.data.iloc[already_selected[i], :].copy(deep=True)
			centroid['label'] = i

			self.centroids = self.centroids.append(centroid, ignore_index=True)

	def PCA_init(self):
		# data handling is done in data_handle.py
		self.kplus_init()

	def do_PCA(self):
		# Breaks down data frame to two components
		pca = PCA(n_components=2)
		x = self.data.loc[:, self.real_values]
		#print x
		#x = sc.fit_transform(x.values)
		x = pca.fit_transform(x)

		PCA_df = pd.DataFrame(data = x, columns = ['PC1', 'PC2'])

		PCA_df = pd.concat([PCA_df, self.data['label']], axis = 1)

		print('CAUTION: PCA supports 9 or less clusters')

		colors = ['rs', 'gs', 'bs', 'ws', 'ks', 'ys', 'cs', 'ms']

		for i in range(self.k):
			plt.plot(PCA_df.loc[PCA_df['label'] == i]['PC1'], PCA_df.loc[PCA_df['label'] == i]['PC2'], colors[i]) 


		plt.show()
		
		"""
		fig = plt.figure(figsize = (20,20))
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_title('2 component PCA', fontsize = 20)

		colors = ['r', 'g', 'b']
		for color in zip(colors):
		    indicesToKeep = PCA_df['target'] == target
		    ax.scatter(PCA_df.loc[indicesToKeep, 'principal component 1']
		               , PCA_df.loc[indicesToKeep, 'principal component 2']
		               , c = color
		               , s = 50)
		ax.legend(targets)
		ax.grid()

		sys.exit()
		"""

	def D1_init(self):
		# Initializes and runs 1 dimensional k means
		pca = PCA(n_components=1)
		x = self.data.loc[:, self.real_values]
		#print x
		#x = sc.fit_transform(x.values)
		x = pca.fit_transform(x)
		PCA_df = pd.DataFrame(data = x, columns = ['PC_val'])
		PCA_df.sort_values(by=['PC_val'], inplace=True)
		self.data = PCA_df.reset_index(drop=True)

		# gets unit cost
		unit_cost = self.get_unit_cost()


		print('\n dynamically constructing clustering \n')
		D = np.zeros([self.k+1, self.data.shape[0]+1]) # actual values to minimize - D[m][i] = min val of m clusters and i data points
		B = np.zeros([self.k, self.data.shape[0]]) # argmins - keep track of ends of clusters - B[m][i]


		for m in range(self.k+1):
			for i in range(self.data.shape[0]):
				#print('iteration for ' + str(m) + ' clusters and ' + str(i) + ' observations')
				# base cases - more/same amount clusters as data points
				if m >= i:
					D[m][i] = 0
					if m == 0:
						B[0][0] = 0
					else:
						B[0][i] = i
				elif m == 0 or i == 0:
					D[m][i] = 0
				else:
					# less clusters than data points
					prob_vals = []
					for j in range(m, i):
						#print('observing data range ' + str(m) + ':' + str(j))
						u_cost = unit_cost[j][i]
						cost = D[m-1][j-1] + u_cost
						#print('\n cost to minimize is ' + str(cost) + '\n')
						prob_vals += [cost]

					#iterated on all points ahead of i, find the minimum of D[m-1][j-1] + unit[j][i]
					min_val = min(prob_vals)
					arg_min = prob_vals.index(min_val)

					D[m][i] = min_val
					B[m-1][i] = arg_min



		print('\n 1D clustering optimal cost is: ' + str(D[self.k][self.data.shape[0]-1]) + '\n') 

		sys.exit()


	def get_unit_cost(self):
		# finds unit cost for all pairs
		# DYNAMIC PROGRAMMING BABY
		# Pandas is too slow to do this, need faster access so mapping data to list
		print('\n getting unit cost \n')
		data = []
		for x in range(self.data.shape[0]):
			data += [self.data.iloc[x]['PC_val']]
	

		# unit cost(i, j) = ||x_j - 1/(j-i+1)*sum(x_i...x_j)||2^2

		unit_cost_array = [] # array of arrays
		for i in range(len(data)):
			cost_array = [] # array of costs per iteration
			for j in range(len(data)):
				if j < i:
					cost = 0
				else:
					if j == i:
						cost = data[j] # if initial step of iteration, unit cost is simply data
					else:
						cost = cost_array[j-i-1] + data[j] # else use past data to figure out the sum
					cost = cost / (j - i + 1) # divide by interval size to get 
					cost = (data[j] - cost)**2
				cost_array += [cost]

			unit_cost_array += [cost_array]

		return unit_cost_array

	def find_nearest_centroid_distance(self, movie_num, centroids):
		# returns the distance to the nearest centroid
		distances = []
		for centroid_num in centroids:
			movie = self.data.iloc[movie_num, :]
			centroid = self.data.iloc[centroid_num, :]
			distances += [self.get_distance(movie, centroid)]

		return min(distances)

	def assign_labels(self):
		# assigns labels to closets centroids
		print('\n assigning labels \n')
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			distances = []
			for j in range(self.centroids.shape[0]):
				centroid = self.centroids.iloc[j, :]
				distances += [self.get_distance(movie, centroid)]

			self.data.at[i, 'label'] = distances.index(min(distances))

	def get_cost(self):
		# Finds the total cost
		print('\n getting cost \n')
		cost = 0
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			centroid = self.centroids.loc[movie['label'], :]
			cost += self.get_distance(movie, centroid)


		return cost

	def get_distance(self, movie, othermovie):
		# Calls the function distance
		return distance(self.real_values, movie, othermovie)



def distance(values, movie, other_movie):
	distance = 0
	for value in values:
		distance += (movie[value] - other_movie[value])**2

	return distance





# This class implements KMEANS with categories



"""
class KMEANS:
	def __init__(self, k, init, data):
		self.k = k
		self.init = init
		self.data = data
		self.real_values = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
		self.categories = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
		self.centroids = []


	def run(self):
		if self.init == 'random':
			# random centrodis
			self.random_init()
		#elif self.init == 'kmeans++':
			#kmeans++ initialization

		self.assign_labels()
		cost = self.get_cost()
		prev_cost = cost + 100
		convergence_condition = 1
		iteration_counter = 0
		cost_array = []
		cost_array += [cost]
		convergence_condition < (prev_cost - cost)

		while(iteration_counter < 12):
			print('\n iteration number: ' + str(iteration_counter))
			prev_cost = cost
			self.average_centroids() # makes centroid the average of all its labels
			cost = self.get_cost()
			cost_array += [cost]
			print(' previous cost: ' + str(prev_cost))
			print(' current cost: ' + str(cost))
			iteration_counter += 1
			#if cost > prev_cost: break
			self.assign_labels() # reassigns labels to movies

		print cost_array


	def average_centroids(self):
		# For each centroid, averages out 
		self.data.sort_values(by=['label'], inplace=True)
		print('\n averaging centroids\n')
		j = 0

		for i in range(self.centroids.shape[0]):
			print('\n current centroid: ' + str(i))
			movie_list = pd.DataFrame(columns=(self.data.columns))
			centroid = self.centroids.iloc[i, :]
			
			if j >= self.data.shape[0]-1:
				# Ran out of data, just die already
				break

			# checks to make sure there are elements for centroid i
			if self.data.iloc[j, :]['label'] > i:
				#means current centroid represents an empty cluster
				continue

			# For each centroid, add to movie_list all movies with the same label
			while(self.data.iloc[j, :]['label'] == i):
				#print('Current movie: ' + str(j) + ' has label ' + str(self.data.iloc[j, :]['label']) + ' assigned to centroid ' + str(self.centroids.iloc[i, :]['label']))

				movie_list.loc[j] = self.data.iloc[j, :]
				j += 1
				
				#makes sure we dont go out of bounds
				if j == self.data.shape[0]:
					break
			
			
			if movie_list.empty:
				# Empty clusters from here on
				return

			# assign average to centroid

			#Keep track of the number of movies in each cluster
			self.centroids.at[i, 'num_assigned'] = movie_list.shape[0]

			for value in self.real_values:
				self.centroids.at[i, value] = movie_list[value].mean()

			
			for category in self.categories:
				# For each category, the mean is the most common category
				avg_length = int(round(sum([len(x) for x in movie_list[category]]) / movie_list.shape[0]))

				if avg_length > 0:
					val_counter = {}
					for category_list in movie_list[category]:
						for section in category_list:
							if section not in val_counter:
								val_counter[section] = 1
							else:
								val_counter[section] += 1


					val_counts = Counter(val_counter)
					val_array = val_counts.most_common(avg_length)
					vals = [val_array[z][0] for z in range(len(val_array))]
					self.centroids.at[i, category] = vals
				
				else:
					self.centroids.at[i, category] = {}
			

		for i in range(self.k):
			print('\n Cluster ' + str(i) + ' has ' + str(self.centroids.iloc[i]['num_assigned']) + ' movies assigned \n')


	def random_init(self):
		# initialize random centroids
		print('\n initializing random centroids \n')
		blank_centroid = self.data.iloc[0, :].copy(deep=True)
		self.centroids = pd.DataFrame(columns=(self.data.columns))

		real_ranges = {}
		category_ranges = {}

		for value in self.real_values:
			#real_ranges[value] = []
			blank_centroid[value] = 0
			#max_val = max(self.data[value])
			#min_val = min(self.data[value])
			#real_ranges[value] = [min_val, max_val]



		for category in self.categories:
			blank_centroid[category] = {}
			category_ranges[category] = []
			for i in range(self.data.shape[0]):
				movie = self.data.iloc[i, :]
				for entry in movie[category]:
					category_ranges[category] += [entry] # this results in weighted uniform random pick


		centroids = []
		for i in range(self.k):
			#init k centroids
			centroid = blank_centroid.copy(deep=True)
			centroid['label'] = i
			for value in self.real_values:
				centroid[value] = random.uniform(0, 1)#(real_ranges[value][0], real_ranges[value][1])


			for category in self.categories:
				if category == 'genres' or category == 'production_companies':
					centroid[category] = [random.choice(category_ranges[category])]
					centroid[category] += [random.choice(category_ranges[category])]
					centroid[category] += [random.choice(category_ranges[category])]
				else:
					centroid[category] = [random.choice(category_ranges[category])]

			centroids += [centroid]
			self.centroids.loc[i] = centroid

						
	def assign_labels(self):
		# assigns labels to closets centroids
		print('\n assigning labels \n')
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			distances = []
			for j in range(self.centroids.shape[0]):
				centroid = self.centroids.iloc[j, :]
				distances += [self.get_distance(movie, centroid)]

			self.data.at[i, 'label'] = distances.index(min(distances))

	def get_cost(self):
		# Finds the total cost
		print('\n getting cost \n')
		cost = 0
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			centroid = self.centroids.iloc[movie['label'], :]
			cost += self.get_distance(movie, centroid)

		return cost

	def get_distance(self, movie, othermovie):
		# Calls the function distance
		return distance(self.data.columns, movie, othermovie)

"""
"""
previous version of distance

def distance(columns, movie, other_movie):
	# computes distance between a pair of movies
	real_values = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
	categories = ['genres', 'production_companies', 'production_countries', 'spoken_languages']

	distance = 0


	for attribute in columns:
		m_properties = movie[attribute]
		om_properties = other_movie[attribute]


		if attribute in real_values: # Compute L2^2 normed distance

			distance += (m_properties - om_properties)**2

		elif attribute in categories:
			# Find the longest list, record its length
			len1 = len(m_properties)
			len2 = len(om_properties)
			if len1 > len2:
				max_len = len1
				max_list = m_properties
				min_list = om_properties
			else:
				max_len = len2
				max_list = om_properties
				min_list = m_properties

		
			# Assign distance 1 if not same, 0 if same
			for i in range(max_len):
				distance += (1 if max_list[i] not in min_list else 0)


	return distance


"""

"""

class KMEANS:
	def __init__(self, k, init, data):
		self.k = k
		self.init = init
		self.data = data

	def run(self):
		# Different versions of k means
		if self.init == 'random':
			self.run_random()

		cost = self.get_cost()
		past_cost = cost + 100
		iteration_counter = 1
		convergence_condition = 1

		while(convergence_condition < abs(past_cost - cost)):
			print('\n running iteration ' + str(iteration_counter) + '\n')
			past_cost = cost

			#TODO: calculate average of each cluster, assign average to centroid
			for i in range(self.k):
				j = 0
				print('computing average of centroid  ' + str(i) + '\n')
				current_movie_list = []
				if (j >= self.data.shape[0] - 1):
					if i < (self.k - 1):
						print('bad initialization')
						sys.exit()
					else:
						break

				while(self.data.iloc[j, :]['label'] < i):
					j += 1
					if (j >= self.data.shape[0] - 1):
						break

				while(self.data.iloc[j, :]['label'] == i):
					
					current_movie_list += [self.data.iloc[j, :]]
					j += 1
					if (j >= self.data.shape[0] - 1):
						break
	
				current_movie_list = pd.DataFrame(current_movie_list)
				self.centroids[i] = self.find_average(current_movie_list, self.centroids[i])

			iteration_counter += 1
			self.reassign_labels()
			cost = self.get_cost()
			print('previous cost was: ' + str(past_cost))
			print('\n current cost is: ' + str(cost) + '\n')

		print('\n optimization finished, recording results \n')
			
	def run_random(self):
		# Strictly for random init
		print('\n initializing random k means... \n')
		centroids = self.init_random() # randomly chooses centroids out of the possible range
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			distances = []
			for centroid in centroids:
				distances += [distance(self.data.columns, movie, centroid)]

			self.data.at[i, 'label'] = distances.index(min(distances))

		print('\n finished initial run, optimizing result until convergence \n')

		self.centroids = centroids
			
	def get_cost(self):
		# Computes cost as sum of L2^2 distances
		cost = 0

		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			centroid = self.centroids[movie['label']]
			cost += distance(self.data.columns, movie, centroid)

		return cost

	def reassign_labels(self):
		# Takes centroids, reassigns labels to each movie
		print('\n Reassigning labels \n')
		for i in range(self.data.shape[0]):
			movie = self.data.iloc[i, :]
			distances = []
			for centroid in self.centroids:
				distances += [distance(self.data.columns, movie, centroid)]

			self.data.at[i, 'label'] = distances.index(min(distances))


	def find_average(self, current_movie_list, centroid):
		# Takes a list, updates the centroid to be the avg of that list
		real_values = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
		categories = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
		
		if current_movie_list.empty:
			return centroid

		for value in real_values:
			centroid[value] = current_movie_list[value].mean() #simply average out 

		for category in categories:
			avg_length = int(round(sum([len(x) for x in current_movie_list[category]]) / current_movie_list.shape[0]))
			if avg_length > 0:

				val_counter = {}
				for category_list in current_movie_list[category]:
					for category in category_list:
						if category not in val_counter:
							val_counter[category] = 1
						else:
							val_counter[category] += 1

				val_counts = Counter(val_counter)
				val_array = val_counts.most_common(avg_length)
				centroid[category] = [val_array[i][0] for i in range(len(val_array))]
		
			else:
				centroid[category] = {}

		return centroid





	def init_random(self):
		# Random centroid initialization
		real_ranges = []
		category_ranges = {}
		centroids = []
		real_values = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
		categories = ['genres', 'production_companies', 'production_countries', 'spoken_languages']

		blank_centroid = self.data.iloc[0, :].copy(deep=True)

		for column in self.data.columns:
			if (column not in real_values) and (column not in categories):
				blank_centroid[column] = ""

		# Find range of each category
		for value in real_values:
			max_val = max(self.data[value])
			min_val = min(self.data[value])
			real_ranges += [[min_val, max_val]]

		for category in categories:
			category_ranges[category] = []
			for i in range(self.data.shape[0]):
				movie = self.data.iloc[i, :]
				for entry in movie[category]:
					if entry not in category_ranges[category]:
						category_ranges[category] += [entry]




		# initialize k centroids
		for i in range(self.k):
			centroid = blank_centroid.copy(deep=True)
			centroid['label'] = i

			for value in real_values:
				centroid[value] = random.uniform(min_val, max_val) #for reals

			for category in categories:
				if category == 'genres' or category == 'production_companies': # start out with 3 genres/production companies - empirically seems to work
					centroid[category] = [random.choice(list(category_ranges[category])), random.choice(list(category_ranges[category])), random.choice(list(category_ranges[category]))]

				elif category == 'production_countries' or category == 'spoken_languages':
					centroid[category] = [random.choice(list(category_ranges[category])), random.choice(list(category_ranges[category]))]

			centroids += [centroid]

		return centroids




"""


 