#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

## Find the folder where xml event data files are saved (Please edit the directory depending on your situation)
folder = os.listdir('F24 LaLIGA 2016/')

## Obtain a list of filenames for parsing
xml_list = []
for i in range(len(folder)):
    if re.search("eventdetails.xml$", folder[i]) != None:
        xml_list.append(folder[i])
        
## External data to interpret qualifiers
qualifier_id = pd.read_csv("qualifier_id.csv")
qualifier_id["type_long"] = qualifier_id["event"] + "-" + qualifier_id["type"]
qualifier_id.qualifier_id = qualifier_id.qualifier_id.apply(str)

## The main parsing step for match r
for r in range(len(xml_list)):
    ## Load the xml data
    xmldoc = minidom.parse('F24 LaLIGA 2016/' + xml_list[r])
    ## Obtain the game information, parse it and save it to csv
    game = xmldoc.getElementsByTagName('Game')
    game_keys = game[0].attributes.keys()
    game_keys
    game_values = []
    ## Build a data frame to store game information
    for i in range(len(game_keys)):
        game_values.append(game[0].attributes[game_keys[i]].value)
    df_game = pd.DataFrame(columns = game_keys)
    df_game.loc[0, :] = game_values
    df_game
    ## Save the game information data as csv file
    df_game.to_csv("Python/" + str(int(df_game.home_team_id)) + "-" + str(int(df_game.away_team_id)) + '-gameinfo.csv')

    ## Get all event-related nodes
    events = xmldoc.getElementsByTagName('Event')
    ## Parse main features of event 0
    event_keys = events[0].attributes.keys()
    event_values = []
    ## Iterate through all events data and store them into a row in a data frame
    for i in range(len(event_keys)):
        event_values.append(events[0].attributes[event_keys[i]].value)
    df_events = pd.DataFrame(columns = event_keys)
    df_events.loc[0, :] = event_values
    df_events

    ## Add qualifiers for event 0
    ## Get all the nodes of qualifiers (for event 0) and its keys
    qualifiers = events[0].getElementsByTagName('Q')
    qualifier_keys = qualifiers[0].attributes.keys()
    
    ## Parse the first qualifier nodes and store the data into a data frame
    if 'value' in qualifiers[0].attributes.keys():
        qualifier_values = []
        for i in range(len(qualifier_keys)):
            qualifier_values.append(qualifiers[0].attributes[qualifier_keys[i]].value)
        qualifier_values

        df_qualifiers = pd.DataFrame(columns = qualifier_keys)
        df_qualifiers.loc[0, :] = qualifier_values
    
    ## Parse all the rest of qualifier nodes of event 0 and store them into the same data frame by concatenation
    for j in range(1, len(qualifiers)):
        qualifier_keys = qualifiers[j].attributes.keys()
        if 'value' in qualifiers[j].attributes.keys(): 
            qualifier_values = []
            for k in range(len(qualifier_keys)):
                qualifier_values.append(qualifiers[j].attributes[qualifier_keys[k]].value)
            qualifier_values

            df_qualifiers1 = pd.DataFrame(columns = qualifier_keys)
            df_qualifiers1.loc[0, :] = qualifier_values
            df_qualifiers1

        df_qualifiers = pd.concat([df_qualifiers, df_qualifiers1])
        
    ## Since the data frame usually has multiple rows, we would like to flatten it into one single row with qualifier type on the columns, in order to concatenate it with existing main events features.
    if "value" in df_qualifiers.columns:
        df_qualifiers_new = df_qualifiers.merge(qualifier_id, left_on = 'qualifier_id', right_on = 'qualifier_id')
        df_qualifiers_new = df_qualifiers_new[["value", "type_long"]]
        df_qualifiers_new_sub = pd.DataFrame(columns = df_qualifiers_new.type_long)
        df_qualifiers_new_sub.loc[0, :] = np.array(df_qualifiers_new.value)
        df_qualifiers_new_sub

    events_new = pd.concat([df_events, df_qualifiers_new_sub], axis = 1)
    
    ## Parse the 2nd till the mth event and concatenate into a bigger data set (an event per iteration)
    for m in xrange(1, len(events)):
        ## Parse event m (same procedure as parsing event 0)
        event_keys = events[m].attributes.keys()
        event_values = []
        for i in range(len(event_keys)):
            event_values.append(events[m].attributes[event_keys[i]].value)
        df_events = pd.DataFrame(columns = event_keys)
        df_events.loc[0, :] = event_values
        df_events

        ## Add qualifiers for event m (same procedure as parsing qualifiers in the event 0)
        qualifiers = events[m].getElementsByTagName('Q')
        
        if len(qualifiers) > 0:
            qualifier_keys = qualifiers[0].attributes.keys()
            ## We make sure we only parse those qualifiers with values
            if 'value' in qualifiers[0].attributes.keys():
                qualifier_values = []
                for i in range(len(qualifier_keys)):
                    qualifier_values.append(qualifiers[0].attributes[qualifier_keys[i]].value)
                qualifier_values

                df_qualifiers = pd.DataFrame(columns = qualifier_keys)
                df_qualifiers.loc[0, :] = qualifier_values

            for j in range(1, len(qualifiers)):
                qualifier_keys = qualifiers[j].attributes.keys()
                if 'value' in qualifiers[j].attributes.keys(): 
                    qualifier_values = []
                    for k in range(len(qualifier_keys)):
                        qualifier_values.append(qualifiers[j].attributes[qualifier_keys[k]].value)
                    qualifier_values

                    df_qualifiers1 = pd.DataFrame(columns = qualifier_keys)
                    df_qualifiers1.loc[0, :] = qualifier_values
                    df_qualifiers1

                df_qualifiers = pd.concat([df_qualifiers, df_qualifiers1])
                
            if "value" in df_qualifiers.columns:
                df_qualifiers_new = df_qualifiers.merge(qualifier_id, left_on = 'qualifier_id', right_on = 'qualifier_id')
                df_qualifiers_new = df_qualifiers_new[["value", "type_long"]]
                df_qualifiers_new_sub = pd.DataFrame(columns = df_qualifiers_new.type_long)
                df_qualifiers_new_sub.loc[0, :] = np.array(df_qualifiers_new.value)
                df_qualifiers_new_sub = df_qualifiers_new_sub.loc[:, ~df_qualifiers_new_sub.columns.duplicated()]
                df_qualifiers_new_sub
            ## We concatenate by column to obtain a big single event row for event m 
            events_new1 = pd.concat([df_events, df_qualifiers_new_sub], axis = 1)
            ## We concatenate by row to obtain a series of event data ordered chronologically
            events_new = pd.concat([events_new, events_new1])
        ## Print to check the status while parsing ~1,500 events for a single match r
        if m%200 == 0:
            print("The " + str(r) + "th file is at the " + str(m) + "th line.")
            
    ## Merge the event_type dataset for classification of attacks
    event_type['type_id'] = event_type['type_id'].astype(str)
    events_new['min'] = events_new['min'].astype(int)
    events_new['sec'] = events_new['sec'].astype(int)
    events_new['outcome'] = events_new['outcome'].astype(int)
    event_type_game = events_new.merge(event_type, left_on = 'type_id', right_on = 'type_id')
    event_type_game = event_type_game.sort_values(['min', 'sec'])
    event_type_game['action'] = np.array(range(event_type_game.shape[0]))
    event_type_game = event_type_game[event_type_game.special != 0]
    double_event = ['Aerial', 'Tackle']
    ## Remove events that would add challenges to correct attacks classification
    for j in range(len(double_event)):
        event_type_game = event_type_game[(event_type_game.event != double_event[j]) | (event_type_game.outcome != 0)]

    event_type_game["attack"] = 0
    event_type_game = event_type_game[event_type_game.event != "Save"]
    event_type_game = event_type_game[(event_type_game.event != "Out") | (event_type_game.outcome != 1)]
    event_type_game = event_type_game[(event_type_game.event != "Foul") | (event_type_game.outcome != 0)]
    event_type_game = event_type_game[(event_type_game.event != "Corner Awarded") | (event_type_game.outcome != 0)]
    event_type_game.index = range(len(event_type_game))
    ## Add a column indicating the nth attack
    for i in range(1, len(event_type_game)):
        try: 
            if(event_type_game.loc[i - 1, "team_id"] != event_type_game.loc[i, "team_id"]):
                event_type_game.loc[i, "attack"] = event_type_game.loc[i - 1, "attack"] + 1
            elif event_type_game.event[i - 1] == "Corner Awarded":
                event_type_game.loc[i, "attack"] = event_type_game.loc[i - 1, "attack"] + 1
            elif event_type_game.event[i - 1] == "Foul":
                event_type_game.loc[i, "attack"] = event_type_game.loc[i - 1, "attack"] + 1
            else:
                event_type_game.loc[i, "attack"] = event_type_game.loc[i - 1, "attack"]
        except KeyError:
            event_type_game.loc[i, "attack"] = event_type_game.loc[i - 1, "attack"]

    ## Save the parsed event data for match r
    events_new.to_csv('Python/' + str(int(df_game.home_team_id)) + "-" + str(int(df_game.away_team_id)) + '.csv')
    ## Print to check the status while parsing 380 La Liga matches for season 2016-2017
    print("The " + str(r) + "th file is ready, with shape: " + str(events_new.shape[0]) + " and " + str(events_new.shape[1]))