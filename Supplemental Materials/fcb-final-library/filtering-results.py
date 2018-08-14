#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pickle
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import csv_to_features
import cluster_results

DATA_DIRECTORY = 'data/All_La_Liga_Matches'
BEST_K = 16
NUM_FEATURES = 21

ATTACKING_TEAM = 'Real Madrid'
OPPONENTS = ['Barcelona', 'Atletico de Madrid', 'Valencia CF', 'Sevilla']
HOME_FLAG = True
AWAY_FLAG = True

if __name__=='__main__':
	# Load necessary pickle files
	print('Loading necessary pickles...')

	with open('pickles/team_name_map.pickle', 'rb') as p:
		team_name_map = pickle.load(p)

	with open('pickles/best_model_16.pickle', 'rb') as p:
		best_model = pickle.load(p)

	# Get files and attacks from parameters
	print('Getting specified attacks...')

	attacking_id = team_name_map[ATTACKING_TEAM]

	opponent_ids = [team_name_map[opp] for opp in OPPONENTS]

	team_filenames = csv_to_features.get_filenames(DATA_DIRECTORY, attacking_id, opponents=opponent_ids, home=HOME_FLAG, away=AWAY_FLAG)

	x_filtered = np.empty((0, NUM_FEATURES))

	# Get attacks from files
	temp_attacks = csv_to_features.get_attacks(attacking_id, DATA_DIRECTORY, team_filenames)
	
	for attack in temp_attacks:
		# Add feature vector of attack to x_full
		x_filtered = np.vstack((x_filtered, csv_to_features.dict_to_feature_vector(attack)))

	print 'Total number of attacks:', len(x_filtered)
	print

	# Predict those labels using the model
	print('Predicting labels...')

	labels = best_model.predict(x_filtered)

	# Generate breakdown pie charts
	print('Generating breakdown chart...')

	filtered_breakdown = cluster_results.get_occurences(labels)
	plt.pie(list(zip(*filtered_breakdown)[0]), autopct='%1.1f%%', labels=list(range(BEST_K)))
	plt.title(ATTACKING_TEAM + " vs. Top Teams at Away", fontsize=20)
	plt.tight_layout()
	plt.savefig(ATTACKING_TEAM, dpi=100)
	plt.close()

	

	