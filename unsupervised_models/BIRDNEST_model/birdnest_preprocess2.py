from __future__ import print_function
import numpy as np
import operator
import pickle
import math
import csv
from sklearn import mixture


TIME_LOG_BASE = 5


def load_yelp_data(use_products):
	fin = open('./data/birdnest_input/yelp/yelp_network_CA.csv', 'r')
	ratings = {} # {user: [(prod, time, rating)]}
	for line in fin:
		temp = line[:-1].split(',')
		user, prod, rating, time = [temp[0], temp[1], int(float( temp[2])), int( temp[3])]
		if use_products: user, prod = prod, user	
		if user not in ratings: 
			ratings[user] = []
		ratings[user].append((prod, time, rating))
	return ratings, None


def process_data(ratings, dataname, use_products):

	keyword = 'prod' if use_products else 'user'
	rating_arr = []
	iat_arr = []
	ids = []
	max_time_diff = -1
	for user in ratings:
		cur_ratings = sorted(ratings[user], key=operator.itemgetter(1))
		for i in range(1, len(cur_ratings)):
			time_diff = cur_ratings[i][1] - cur_ratings[i-1][1]
			max_time_diff = max(max_time_diff, time_diff)

	S = int(1 + math.floor(math.log(1 + max_time_diff, TIME_LOG_BASE)))
	for user in ratings:
		if len(ratings[user]) <= 1: continue
		rating_counts = [0] * 5
		iat_counts = [0] * S
		cur_ratings = sorted(ratings[user], key=operator.itemgetter(1))
		rating_counts[cur_ratings[0][2] - 1] += 1
		for i in range(1, len(cur_ratings)):
			time_diff = cur_ratings[i][1] - cur_ratings[i-1][1]
			iat_bucket = int(math.floor(math.log(1 + time_diff, TIME_LOG_BASE)))
			rating_counts[cur_ratings[i][2] - 1] += 1
			iat_counts[iat_bucket] += 1
		rating_arr.append(rating_counts)
		iat_arr.append(iat_counts)
		ids.append(user)

	rating_arr = np.array(rating_arr)
	iat_arr = np.array(iat_arr)
	return (rating_arr, iat_arr, ids)
