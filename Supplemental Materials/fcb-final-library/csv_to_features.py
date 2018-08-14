#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np

CSV_DIRECTORY = 'data/All_La_Liga_Matches'


def dict_to_feature_vector(attack_dict):
	v = np.zeros((1,21))
 
	v[0,0] = time_duration(attack_dict)
	v[0,1] = min(number_of_passes(attack_dict)/5,4)
	v[0,2] = total_vertical(attack_dict)/140
	v[0,3] = min(vertical_horizontal_ratio(attack_dict)/2,4)
	v[0,4] = average_attacking_speed(attack_dict)/1000
	v[0,5] = min(3*number_long(attack_dict),4)
	# v[0,6] = min(number_short(attack_dict)/4,4)
	v[0,6] = 15*time_duration(attack_dict)/max(number_of_passes(attack_dict), 1)
	v[0,7] = if_arrival(attack_dict)

	f = flow_motif_occurences(attack_dict)
	f_sum = max(sum([f[_]for _ in f]),1)
	v[0,8] = 2*f['ABAC']/f_sum
	v[0,9] = 2*f['ABAB']/f_sum
	v[0,10] = 2*f['ABCA']/f_sum
	v[0,11] = 2*f['ABCB']/f_sum
	v[0,12] = 2*f['ABCD']/f_sum

	v[0,13] = 4*pctg_time_back(attack_dict)
	v[0,14] = 4*pctg_time_midfield(attack_dict)
	v[0,15] = 4*pctg_time_last_third(attack_dict)
	v[0,16] = 4*pctg_time_left(attack_dict)
	v[0,17] = 4*pctg_time_center(attack_dict)
	v[0,18] = 4*pctg_time_right(attack_dict)
	# v[0,19] = end_box(attack_dict)/10
	x,y = possession_loss(attack_dict)
	v[0,19] = x
	v[0,20] = y
	return v

def dict_to_spatial_sequence(attack_dict):
	current_cell=None
	sequence=[]

	for e in attack_dict['events']:
		start_x,start_y=e['origin_position']

		grid_x = int(start_x//10) if 0<start_x<100 else 0 if start_x<=0 else 9
		grid_y = int(start_y//20) if 0<start_y<100 else 0 if start_y<=0 else 4
		cell = '{}{}'.format(grid_y,grid_x)

		if cell!=current_cell:
			sequence.append(cell)
			current_cell=cell

		end_x,end_y=e['origin_position']

		grid_x = int(end_x//10) if 0<end_x<100 else 0 if end_x<=0 else 9
		grid_y = int(end_y//20) if 0<end_y<100 else 0 if end_y<=0 else 4
		cell = '{}{}'.format(grid_y,grid_x)

		if cell!=current_cell:
			sequence.append(cell)
			current_cell=cell
	return sequence

def get_filenames(directory, attacking_team, opponents=None, home=True, away=True):

	filenames=[]
	folder=os.listdir(directory)

	for f in folder: 

		if f[-4:]!='.csv':
			continue

		g=f.replace('-','.').split('.')
		h= g[0]
		a= g[1]

		if home:
			if h == attacking_team:
				if opponents == None or a in opponents:
					filenames.append(f)

		if away:
			if a == attacking_team:
				if opponents == None or h in opponents:
					filenames.append(f)

	return filenames



def get_attacks(attacking_team,directory,filenames,min_time=0.08):
	all_attacks=[]
	long_attacks=[]
	for fname in filenames:
		split_fname = fname.replace('-','.').split('.')

		hteam=split_fname[0]
		ateam=split_fname[1]
		if hteam==attacking_team:
			home=True
			away=False
			defending_team=ateam
		elif ateam==attacking_team:
			home=False
			away=True
			defending_team=hteam
		else:
			continue

		margin=0
		current_number=None
		current_attack=None
		current_period = None
		firstline=True

		with open(directory + '/' + fname) as csvfile:
			f=csv.reader(csvfile,delimiter=',', quotechar='"')

			current_number=None
			for l in f:
				if firstline:
					firstline=False
					team_index = l.index('team_id')
					type_index = l.index('event')
					success_index = l.index('outcome')
					x_index = l.index('x')
					y_index = l.index('y')
					player_index = l.index('player_id')
					min_index = l.index('min')
					sec_index = l.index('sec')
					
					try:
						min_index_2 = l.index('end_min')
						sec_index_2 = l.index('end_sec')
					except:
						min_index_2 = min_index
						sec_index_2 = sec_index
					
					end_x_index = l.index('Pass End X-passing')
					end_y_index = l.index('Pass End Y-passing')
					attack_index = l.index('attack')
					distance_index = l.index('Length-passing')
					period_index = l.index('period_id')
					continue
				event_team = l[team_index]
				event_type = l[type_index]

				if event_type == 'Goal':
					if event_team == attacking_team:
						margin+=1
					else:
						margin -=1
				if event_team != attacking_team:
					continue

				period = l[period_index]
				attack_number = l[attack_index]
				event_dict={}
				event_dict['type'] = l[type_index]
				event_dict['successful'] = True if l[success_index]=='1' else False
				event_dict['origin_position']= (float(l[x_index]),float(l[y_index]))
				x=event_dict['origin_position'][0]
				if l[end_x_index]!='' and l[end_y_index]!='':
					event_dict['destination_position']=float(l[end_x_index]),float(l[end_y_index])
				else:
					event_dict['destination_position']=(float(l[x_index]),float(l[y_index]))
				event_dict['player'] = l[player_index]
				event_dict['time'] = float(l[min_index])+float(l[sec_index])/60
				event_dict['endtime'] = float(l[min_index_2])+float(l[sec_index_2])/60
				event_dict['startbox'] = event_dict['origin_position'][0] // (100*(1/3 + 0.01)) + 3*(event_dict['origin_position'][1] // (100*(1/3 + 0.01))) + 1
				event_dict['endbox'] = event_dict['destination_position'][0] // (100*(1/3 + 0.01)) + 3*(event_dict['destination_position'][1] // (100*(1/3 + 0.01))) + 1
				event_dict['time_duration'] = event_dict['endtime'] - event_dict['time']

				try: 
					event_dict['distance'] = float(l[distance_index])
				except:
					event_dict['distance'] = 0

				if event_dict['startbox'] == event_dict['endbox']:
					event_dict['box_equal'] = 1
				else:
					event_dict['box_equal'] = 0
				for o in range(1,10):
					event_dict['time_in_box' + str(o)] = 0
				if event_dict['box_equal'] == 1:
					event_dict['time_in_box' + str(int(event_dict['startbox']))] = event_dict['time_duration']
				else:
					event_dict['time_in_box' + str(int(event_dict['startbox']))] = 0.5*event_dict['time_duration']
					event_dict['time_in_box' + str(int(event_dict['endbox']))] = 0.5*event_dict['time_duration']

				if attack_number == current_number and period == current_period:

					current_attack['events'].append(event_dict)
					current_attack['time'][1]=event_dict['time']
				else:
					if current_attack is not None:
						all_attacks.append(current_attack)
					current_number=attack_number
					current_period=period
					current_attack={}
					current_attack['team']=attacking_team
					current_attack['opponent']=defending_team
					current_attack['home']=home
					current_attack['away']=away
					current_attack['score_margin']=margin
					current_attack['time']=[event_dict['time'],event_dict['time']]
					current_attack['events']=[event_dict]
					current_attack['origin_position']=event_dict['origin_position']
					current_attack['arrival']=False
				
				if x>67:
					current_attack['arrival']=True
					
	for attack in all_attacks:
			if attack['time'][1]-attack['time'][0]>=min_time:
				long_attacks.append(attack)

	return long_attacks


##### Functions that generate features
def time_duration(attack_dict):
	return attack_dict['time'][1]-attack_dict['time'][0]
	
def number_of_passes(attack_dict):
	return len([1 for e in attack_dict['events'] if (e['type']=='Pass' and e['successful'])])

def total_vertical(attack_dict):
	v=0
	x=attack_dict['events'][0]['origin_position'][0]
	for e in attack_dict['events']:
		start_x,start_y=e['origin_position']
		end_x,end_y=e['destination_position']
		v+=abs(start_x-x)
		v+=abs(end_x-start_x)
		x=end_x
	return v

def total_horizontal(attack_dict):
	h=0
	y=attack_dict['events'][0]['origin_position'][1]
	for e in attack_dict['events']:
		start_x,start_y=e['origin_position']
		end_x,end_y=e['destination_position']
		h+=abs(start_y-y)
		h+=abs(end_y-start_y)
		y=end_y
	return h

def end_box(attack_dict):
	return(attack_dict['events'][-1]['endbox'])

def if_arrival(attack_dict):
	return(attack_dict['arrival'])

def flow_motif_occurences(attack_dict):
	flow_motif_occurences = {'ABAB': 0,
							 'ABAC': 0,
							 'ABCB': 0,
							 'ABCA': 0,
							 'ABCD': 0}

	pattern_chars = ['A', 'B', 'C', 'D']
	length = 4
	
	passes = []
	for e in attack_dict['events']:
		if e['type'] == 'Pass' and e['successful']:
			passes.append(e['player'])

	for i in range(len(passes) + 1 - 4):
		pattern_str = ''
		temp_map = {}
		temp_i = 0
		for j in range(i, i+length):
			player = passes[j]
			if player in temp_map:
				pattern_str += temp_map[player]
			else:
				pattern_str += pattern_chars[temp_i]
				temp_map[player] = pattern_chars[temp_i]
				temp_i += 1

		# Got weird sequence of events from same player twice in a row
		if pattern_str in flow_motif_occurences:
			flow_motif_occurences[pattern_str] += 1 
	
	return flow_motif_occurences

def total_distance(attack_dict):
	d=0
	x,y=attack_dict['events'][0]['origin_position']
	for e in attack_dict['events']:
		start_x,start_y=e['origin_position']
		end_x,end_y=e['destination_position']
		d+=((start_x-x)**2+(start_y-y)**2)**.5
		d+=((end_x-start_x)**2+(end_y-start_y)**2)**.5
		x,y=end_x,end_y
	return d

def average_attacking_speed(attack_dict):
	return total_distance(attack_dict)/time_duration(attack_dict)

def number_long(attack_dict):
	l =0
	for e in attack_dict['events']:
		if e['type']=='Pass' and e['successful']:
			if e['distance']>=40:
				l+=1
	return l

def number_short(attack_dict):
	s =0
	for e in attack_dict['events']:
		if e['type']=='Pass' and e['successful']:
			if e['distance']<=15:
				s+=1
	return s 

def vertical_horizontal_ratio(attack_dict):
	return total_vertical(attack_dict)/(total_horizontal(attack_dict)+.00001) 

def pctg_time_back(attack_dict):
	# assumes each event occurs instantaneously
	# all time is spent in between attacks

	time_back=0
	x,y = attack_dict['events'][0]['origin_position']
	t = attack_dict['events'][0]['time']

	for e in attack_dict['events']:
		e_time = e['time']
		start_x,start_y = e['origin_position']
		end_x,end_y = e['destination_position']

		if x<=33 and start_x<=33:   # all movement in back
			time_back+=e_time-t     # add entire time to time_back
		elif (x<=33 and start_x>33) or (x>33 and start_x<=33): # 1/2 events in back
			time_back+=(e_time-t)/2

		x,y=end_x,end_y
		t=e_time
	return time_back/time_duration(attack_dict)

def pctg_time_midfield(attack_dict):
	# assumes each event occurs instantaneously
	# all time is spent in between attacks

	time_midfield=0
	x,y = attack_dict['events'][0]['origin_position']
	t = attack_dict['events'][0]['time']

	for e in attack_dict['events']:
		e_time = e['time']
		start_x,start_y = e['origin_position']
		end_x,end_y = e['destination_position']

		if (x>=33 and x<=66) and (start_x>=33 and start_x<=66) :   # all movement in midfield
			time_midfield+=e_time-t     # add entire time to time_midfield
		elif ((x<33 or x>66) and (start_x>=33 and start_x <=66)) or ((x>=33 and x<=66) and (start_x<33 or start_x>66)): # 1/2 events in back
			time_midfield+=(e_time-t)/2

		x,y=end_x,end_y
		t=e_time
	return time_midfield/time_duration(attack_dict)

def pctg_time_last_third(attack_dict):
	# assumes each event occurs instantaneously
	# all time is spent in between attacks

	time_last_third=0
	x,y = attack_dict['events'][0]['origin_position']
	t = attack_dict['events'][0]['time']

	for e in attack_dict['events']:
		e_time = e['time']
		start_x,start_y = e['origin_position']
		end_x,end_y = e['destination_position']

		if x>=66 and start_x>=66:   # all movement in back
			time_last_third+=e_time-t     # add entire time to time_back
		elif (x>=66 and start_x<66) or (x<66 and start_x>=66): # 1/2 events in back
			time_last_third+=(e_time-t)/2

		x,y=end_x,end_y
		t=e_time
	return time_last_third/time_duration(attack_dict)

def pctg_time_left(attack_dict):
	# assumes each event occurs instantaneously
	# all time is spent in between attacks

	time_left=0
	x,y = attack_dict['events'][0]['origin_position']
	t = attack_dict['events'][0]['time']

	for e in attack_dict['events']:
		e_time = e['time']
		start_x,start_y = e['origin_position']
		end_x,end_y = e['destination_position']

		if y>=66 and start_y>=66:   # all movement in back
			time_left+=e_time-t     # add entire time to time_back
		elif (y>=66 and start_y<66) or (y<66 and start_y>=66): # 1/2 events in back
			time_left+=(e_time-t)/2

		x,y=end_x,end_y
		t=e_time
	return time_left/time_duration(attack_dict)

def pctg_time_center(attack_dict):
	# assumes each event occurs instantaneously
	# all time is spent in between attacks

	time_center=0
	x,y = attack_dict['events'][0]['origin_position']
	t = attack_dict['events'][0]['time']

	for e in attack_dict['events']:
		e_time = e['time']
		start_x,start_y = e['origin_position']
		end_x,end_y = e['destination_position']

		if (y>=33 and y<=66) and (start_y>=33 and start_y<=66) :   # all movement in midfield
			time_center+=e_time-t     # add entire time to time_midfield
		elif ((y<33 or y>66) and (start_y>=33 and start_y <=66)) or ((y>=33 and y<=66) and (start_y<33 or start_y>66)): # 1/2 events in back
			time_center+=(e_time-t)/2

		x,y=end_x,end_y
		t=e_time
	return time_center/time_duration(attack_dict)

def pctg_time_right(attack_dict):
	# assumes each event occurs instantaneously
	# all time is spent in between attacks

	time_right=0
	x,y = attack_dict['events'][0]['origin_position']
	t = attack_dict['events'][0]['time']

	for e in attack_dict['events']:
		e_time = e['time']
		start_x,start_y = e['origin_position']
		end_x,end_y = e['destination_position']

		if y<=33 and start_y<=33:   # all movement in back
			time_right+=e_time-t     # add entire time to time_back
		elif (y<=33 and start_y>33) or (y>33 and start_y<=33): # 1/2 events in back
			time_right+=(e_time-t)/2

		x,y=end_x,end_y
		t=e_time
	return time_right/time_duration(attack_dict)


def possession_loss(attack_dict):
	endbox=end_box(attack_dict)

	x,y=0,0

	if endbox in [7,8,9]: 	#left
		y=2
	elif endbox in [4,5,6]:	#center
		y=1
	elif endbox in [1,2,3]:	#right
		y=0

	if endbox in [1,4,7]: 	#back
		x=0
	elif endbox in [2,5,8]:	#midfield
		x=1
	elif endbox in [3,6,9]:	#attacking 3rd
		x=2
	return x,y



### FIX: time duration for all spatial percentage features

if __name__ == '__main__':
	filenames = get_filenames(CSV_DIRECTORY, "182", "177", home = True)
	attacks = get_attacks('182',CSV_DIRECTORY,filenames, min_time=0.2)

	print(len(attacks))
	print(dict_to_feature_vector(attacks[0]))
