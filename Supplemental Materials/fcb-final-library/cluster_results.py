#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pickle
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import json

##################
### Parameters ###
##################
NUM_FEATURES = 21	# Number of features used in model
BEST_K = 16			# Number of clusters to used in model
TOP_N = 20			# Number of top attacks from each cluster to interpret clusters from

REP_DIREC = 'most-representative-attacks'
VID_DIREC = 'representative-video-attacks'
DIST_DIREC = 'team-attack-distributions'

META = 'home_team,away_team,attacking_team,start_time,end_time,'
HEADER = 'time_duration,num_passes,total_vertical,vertical_horizontal_ratio,avg_attacking_speed,num_long_passes,time_per_pass,arrival,ABAC,ABAB,ABCA,ABCB,ABCD,pctg_time_back,pctg_time_midfield,pctg_time_last_third,pctg_time_left,pctg_time_center,pctg_time_right,possesion_loss_x,possesion_loss_y'

# List of tuples of matches we have video for
video_matches = [('Sevilla', 'Real Madrid'),
				 ('Athletic Club', 'Atlético de Madrid'),
				 ('Eibar', 'Barcelona'),
				 ('Real Madrid', 'Real Sociedad'),
				 ('Barcelona', 'Athletic Club'),
				 ('Las Palmas', 'Sevilla'),
				 ('Sevilla', 'Eibar'),
				 ('Real Sociedad', 'Athletic Club'),
				 ('Athletic Club', 'Real Madrid'),
				 ('Atlético de Madrid', 'Sevilla')]


#################
### Functions ###
#################

def is_attack_in_video(attack):
	if attack['home']:
		return (attack['team'], attack['opponent']) in video_match_ids
	else:
		return (attack['opponent'], attack['team']) in video_match_ids

def get_occurences(data):
	total = float(len(data))
	occurences = [(0,0)] * BEST_K
	for x in data:
		new_count = occurences[x][1]+1
		occurences[x] = (new_count/total, new_count)
	return occurences

def get_label_indices(full_labels):
	cluster_ixs = [[] for _ in range(BEST_K)]
	for i, l in enumerate(full_labels):
		cluster_ixs[l].append(i)
	return cluster_ixs

def get_ordered_probabilities(full_probs, full_labels):
	all_probs = [[] for _ in range(BEST_K)]
	for i, l in enumerate(full_labels):
		all_probs[l].append((i, full_probs[i,l]))

	for i, tp in enumerate(all_probs):
		all_probs[i] = sorted(tp, key=lambda x: x[1], reverse=True)

	return all_probs

def get_most_representative_attacks(full_features, full_probs, full_labels):
	if not os.path.exists(REP_DIREC):
		os.makedirs(REP_DIREC)

	all_probs = get_ordered_probabilities(full_probs, full_labels)

	for i, tp in enumerate(all_probs):
		top_n_probs = list(zip(*tp[:TOP_N])[0])
		rep_filename = REP_DIREC + '/cluster-{}.csv'.format(i)
		temp_features = full_features[top_n_probs,:]
		np.savetxt(rep_filename, temp_features, delimiter=',', header=HEADER, comments='')


def get_most_representative_attacks_with_video(full_attacks, full_features, full_probs, full_labels):
	if not os.path.exists(VID_DIREC):
		os.makedirs(VID_DIREC)

	all_probs = get_ordered_probabilities(full_probs, full_labels)
	clustered_video_features = [np.empty((0, NUM_FEATURES+5)) for _ in range(BEST_K)]

	for i, attack_probs in enumerate(all_probs):
		for attack_i, attack_prob in attack_probs:
			if len(clustered_video_features[i]) == 10:
				break
			
			temp_attack = full_attacks[attack_i]
			if is_attack_in_video(temp_attack):
				home_id = int(temp_attack['team'] if temp_attack['home'] else temp_attack['opponent'])
				away_id = int(temp_attack['opponent'] if temp_attack['home'] else temp_attack['team'])
				attacking_id = int(temp_attack['team'])
				
				meta_attack_info = np.array([home_id, away_id, attacking_id] + temp_attack['time'])
				temp_video_features = np.hstack((meta_attack_info, full_features[attack_i,:]))
				clustered_video_features[i] = np.vstack((clustered_video_features[i], temp_video_features))

	for i, feats in enumerate(clustered_video_features):
		rep_filename = VID_DIREC + '/cluster-{}.csv'.format(i)
		np.savetxt(rep_filename, feats, delimiter=',', header=META+HEADER, comments='')

def get_team_avgs_stds(full_team_ranges, full_features):
	team_avgs = np.empty((0,NUM_FEATURES+1))
	team_stds = np.empty((0,NUM_FEATURES+1))

	for team_id in full_team_ranges:
		start, end = full_team_ranges[team_id]
		temp_features = full_features[start:end+1]
		team_avgs = np.vstack((team_avgs, np.hstack((int(team_id), np.mean(temp_features, axis=0)))))
		team_stds = np.vstack((team_stds, np.hstack((int(team_id), np.std(temp_features, axis=0)))))

	np.savetxt('team_avgs.csv', team_avgs, delimiter=',', header='team_id,'+HEADER, comments='')
	np.savetxt('team_stds.csv', team_stds, delimiter=',', header='team_id,'+HEADER, comments='')

def get_cluster_avgs_stds(full_labels, full_features):
	cluster_ixs = get_label_indices(full_labels)

	cluster_avgs = np.empty((0,NUM_FEATURES+1))
	cluster_stds = np.empty((0,NUM_FEATURES+1))

	for i, cluster_ix in enumerate(cluster_ixs):
		temp_features = full_features[cluster_ix]
		cluster_avgs = np.vstack((cluster_avgs, np.hstack((i, np.mean(temp_features, axis=0)))))
		cluster_stds = np.vstack((cluster_stds, np.hstack((i, np.std(temp_features, axis=0)))))

	np.savetxt('cluster_avgs.csv', cluster_avgs, delimiter=',', header='cluster,'+HEADER, comments='')
	np.savetxt('cluster_stds.csv', cluster_stds, delimiter=',', header='cluster,'+HEADER, comments='')	

def get_team_cluster_distributions(full_team_ranges, full_labels):
	if not os.path.exists(DIST_DIREC):
		os.makedirs(DIST_DIREC)

	team_results = {}
	for team_id in full_team_ranges:
		start, end = full_team_ranges[team_id]
		temp_labels = full_labels[start:end+1]
		team_results[team_id] = get_occurences(temp_labels)

	total_occurences = [0] * BEST_K
	for team_id in team_results:
		team_occurences = list(zip(*team_results[team_id])[1])
		for i in range(BEST_K):
			total_occurences[i] += team_occurences[i]

		plt.pie(list(zip(*team_results[team_id])[0]), autopct='%1.1f%%', labels=list(range(BEST_K)))
		plt.title(team_id_map[team_id], fontsize=30)
		
		plt.tight_layout()
		plt.savefig(DIST_DIREC + '/' + team_id_map[team_id], dpi=100)
		plt.close()
	
	pct_occurences = [float(o)/sum(total_occurences) for o in total_occurences]
	plt.pie(pct_occurences, autopct='%1.1f%%', labels=list(range(BEST_K)))
	plt.title('League-Wide Breakdown', fontsize=30)
	plt.tight_layout()
	plt.savefig(DIST_DIREC + '/0-League-Wide', dpi=100)
	plt.close()

	team_results = {team:list(zip(*team_results[team])[0]) for team in team_results}
	get_most_similar_teams(team_results)
	team_breakdowns_to_csv(team_results)


def get_most_similar_teams(team_breakdowns):
	most_similar_teams = {}

	for team in team_breakdowns:
		temp_distances = []
		for other_team in team_breakdowns:
			if team == other_team:
				continue

			temp_dist = distance_between_breakdowns(team_breakdowns[team], team_breakdowns[other_team])
			temp_distances.append((other_team, temp_dist))
		
		temp_distances = sorted(temp_distances, key=lambda x: x[1])
		most_similar_teams[team] = list(zip(*temp_distances)[0])

	with open('0-most_similar_teams.json', 'w') as f:
		json.dump(most_similar_teams, f)

def distance_between_breakdowns(breakdown1, breakdown2):
	return sum([(breakdown2[i]-breakdown1[i])**2 for i in range(BEST_K)])**0.5

def team_breakdowns_to_csv(team_breakdowns):
	BREAKDOWN_HEADER = 'team_id,' + ",".join(['cluster %d' % i for i in range(BEST_K)])
	breakdown_filename = DIST_DIREC + '/full_breakdown.csv'

	breakdown_np = []
	for team_id in team_breakdowns:
		breakdown_np.append([int(team_id)] + team_breakdowns[team_id])
	breakdown_np = np.array(breakdown_np)
	np.savetxt(breakdown_filename, breakdown_np, delimiter=',', header=BREAKDOWN_HEADER, comments='')


if __name__ == '__main__':
	##############################
	### Loading Data and Model ###
	##############################

	print('Loading necessary pickles...')

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

	with open('pickles/best_model_16.pickle', 'rb') as p:
		best_model = pickle.load(p)

	######################
	### Saving Results ###
	######################

	print('Getting results...')

	labels = best_model.predict(x_full)
	x_probs = best_model.predict_proba(x_full)


	# Saving most representative attacks
	print('Saving most representative attacks...')
	get_most_representative_attacks(x_full, x_probs, labels)

	# Saving most representative attacks with video
	print('Saving most representative attacks with video...')
	video_match_ids = set([(team_name_map[home], team_name_map[away]) for home,away in video_matches])
	get_most_representative_attacks_with_video(attacks_full, x_full, x_probs, labels)

	# Saving team averages and standard deviations of features
	print('Saving team averages and standard deviations of features...')
	get_team_avgs_stds(team_attacks_ranges, x_full)

	# Saving cluster averages and standard deviations of features
	print('Saving cluster averages and standard deviations of features...')
	get_cluster_avgs_stds(labels, x_full)

	# Saving team breakdowns of clusters
	print('Saving team breakdowns of clusters...')
	get_team_cluster_distributions(team_attacks_ranges, labels)
	