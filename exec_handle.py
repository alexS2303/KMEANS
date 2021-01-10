import pandas as pd 
from KMEANS import KMEANS
from tqdm import tqdm


# Handles the execution process of KMEANS

class Executor:
	def __init__(self, data, params):
		self.data = data
		self.params = params

	def run(self):
		# Runs k means or gets disagreement distance
		if self.params.init == 'dist':
			kplus_data = self.data.copy(deep=True)
			d1_data = self.data.copy(deep=True)
			kplus = KMEANS(self.params.k, 'kmeans++', kplus_data)
			#d1 = KMEANS(self.params.k, '1D', d1) UNCOMMENT WHEN 1D is working
			d1 = KMEANS(self.params.k, 'random', d1_data) #comment this line when 1D is working
			kplus.run()
			d1.run()

			self.kplus_labels = kplus_data[['id', 'label']]
			self.d1_labels = d1_data[['id', 'label']]
			self.get_disagreement_distance()

		# conventional execution

		else:
			kmeans = KMEANS(self.params.k, self.params.init, self.data)
			kmeans.run()


	def get_disagreement_distance(self):
		# Finds clustering disagreement distance
		self.d1_labels.sort_values(by=['id'], inplace=True)
		self.kplus_labels.sort_values(by=['id'], inplace=True)
		distance = 0
		for i in tqdm(range(self.d1_labels.shape[0])):
			for j in tqdm(range(i+1, self.d1_labels.shape[0])):
				# checking if movie x and movie y are in the same cluster
				d1_label_x = self.d1_labels.iloc[i]['label']
				kplus_label_x = self.kplus_labels.iloc[i]['label']

				d1_label_y = self.d1_labels.iloc[j]['label']
				kplus_label_y = self.kplus_labels.iloc[j]['label']

				# if x y are not same in both or not different in both
				if not (((d1_label_x == d1_label_y) and (kplus_label_x == kplus_label_y)) or ((d1_label_x != d1_label_y) and (kplus_label_x != kplus_label_y))):
					distance += 1



		print('\n disagreement distance is: ' + str(distance) + '\n')







	def get_data(self):
		return self.data



