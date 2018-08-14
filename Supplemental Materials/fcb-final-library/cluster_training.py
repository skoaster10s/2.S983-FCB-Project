#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pickle
from sklearn.mixture import GaussianMixture
import csv
import numpy as np
import csv_to_features

##################
### Parameters ###
##################

DATA_DIRECTORY = 'data/All_La_Liga_Matches'
TEAM_ID_FILE = 'data/team_id.csv'
NUM_FEATURES = 21

BEST_K = 16				# Number of clusters to be used in model
BEST_PARAM = 'full'		# Parameterization method to be used when trainig model

data_from_pickle = True 	# Loading full seasons data from pickle if True


if __name__=='__main__':
	####################
	### Loading Data ###
	####################

	print('Loading data...')

	if data_from_pickle:
		with open('pickles/team_id_map.pickle', 'rb') as p:
			team_id_map = pickle.load(p)

		with open('pickles/team_name_map.pickle', 'rb') as p:
			team_name_map = pickle.load(p)

		with open('pickles/x_full.pickle', 'rb') as p:
			x_full = pickle.load(p)

		with open('pickles/attacks_full.pickle', 'rb') as p:
			attacks_full = pickle.load(p)

		with open('pickles/team_attacks.pickle', 'rb') as p:
			team_attacks = pickle.load(p)

		with open('pickles/team_attack_ranges.pickle', 'rb') as p:
			team_attacks_ranges = pickle.load(p)

	else:
		with open(TEAM_ID_FILE, mode='r') as team_csv:
			reader = csv.reader(team_csv)
			next(reader, None)
			team_id_map = {rows[0]:rows[1] for rows in reader}
			team_name_map = {team_id_map[i]:i for i in team_id_map}

		team_filenames = {}
		for team_id in team_id_map:
			team_filenames[team_id] = csv_to_features.get_filenames(DATA_DIRECTORY, team_id)


		team_attacks = {}
		team_attack_ranges = {}
		attacks_full = []
		x_full = np.empty((0, NUM_FEATURES))

		last_i = 0
		for i, team_id in enumerate(team_filenames):
			# Get attacks from files
			print 'Team ID:', team_id
			temp_attacks = csv_to_features.get_attacks(team_id, DATA_DIRECTORY, team_filenames[team_id])
			print 'Number of attacks:', len(temp_attacks)
			
			# Update team_attack_ranges
			team_attacks[team_id] = temp_attacks
			team_attack_ranges[team_id] = (last_i, last_i + len(temp_attacks) - 1)
			last_i += len(temp_attacks)
			
			for j, attack in enumerate(temp_attacks):
				# Add attack dict to attacks_full
				attacks_full.append(attack)
				
				# Add feature vector of attack to x_full
				x_full = np.vstack((x_full, csv_to_features.dict_to_feature_vector(attack)))

		# Save data into pickles
		with open('pickles/team_id_map.pickle', 'wb') as p:
			pickle.dump(team_id_map, p)

		with open('pickles/team_name_map.pickle', 'wb') as p:
			pickle.dump(team_name_map, p)

		with open('pickles/x_full.pickle', 'wb') as p:
			pickle.dump(x_full, p)

		with open('pickles/attacks_full.pickle', 'wb') as p:
			pickle.dump(attacks_full, p)

		with open('pickles/team_attacks.pickle', 'wb') as p:
			pickle.dump(team_attacks, p)

		with open('pickles/team_attack_ranges.pickle', 'wb') as p:
			pickle.dump(team_attack_ranges, p)


	print 'Total number of attacks: ', len(x_full)
	print 'Finished loading data.'
	print

	#################################
	### Training Clustering Model ###
	#################################

	print 'Training clustering model...'

	# Train and save new model
	best_model = GaussianMixture(n_components=BEST_K, covariance_type=BEST_PARAM, init_params='kmeans').fit(x_full)

	with open('pickles/best_model_16.pickle', 'wb') as p:
		pickle.dump(best_model, p)

	print 'Finished training clustering model.'

