from __future__ import division, print_function
from collections import Counter
import math
from operator import itemgetter
import operator
import random
import numpy as np

import pandas as pd 
import sklearn
from sklearn.cluster import KMeans
from scipy.special import gammaln, psi
import pickle
import matplotlib.pyplot as plt
from birdnest_preprocess2 import *
from birdnest_detect2 import *
import pdb

#np.set_printoptions(threshold='nan')
#np.set_printoptions(linewidth=160)

USE_PRODUCTS = 0
USE_TIMES = 1
keyword = 'prod' if USE_PRODUCTS else 'user'


dataname = 'yelp'
ratings, usermap = load_yelp_data(USE_PRODUCTS)
(rating_arr, iat_arr, ids) = process_data(ratings, dataname, USE_PRODUCTS)
(rating_arr, iat_arr) = (np.array(rating_arr), np.array(iat_arr))

(rating_arr, iat_arr) = (rating_arr[0:5000], iat_arr[0:5000])

# Detect suspicious users given matrices containing ratings and  inter-arrival times. 
# USE_TIMES is a boolean for whether the inter-arrival times should be used. The last parameter is the number of clusters to use. 
USE_TIMES = 1
suspn = detect(rating_arr, iat_arr, USE_TIMES, 4)


# OUTPUT RESULTS TO FILE: it considers the top (NUM_TO_OUTPUT) most suspicious users and stores their user ids, scores, ratings and IATs in separate files.
NUM_TO_OUTPUT = 500 # number of suspicious users to output to file
susp_sorted = np.array([(x[0]) for x in sorted(enumerate(suspn), key=itemgetter(1), reverse=True)])
most_susp = susp_sorted[range(1000)]
with open('./data/birdnest_output/%s/top%d%s_ids.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as outfile:
	with open('./data/birdnest_output/%s/top%d%s_scores.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as out_scores:
		with open('./data/birdnest_output/%s/top%d%s_ratings.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as out_rating:
			with open('./data/birdnest_output/%s/top%d%s_iat.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as out_iat:
				for i in most_susp:
					if usermap == None:
						print ( '%s' % (ids[i],), file=outfile)
					else:
						print ('%s %s' % (ids[i], usermap[ids[i]]), file = outfile)
					print ( '%s %f' % (ids[i], suspn[i]), file =out_scores)
					print ( rating_arr[i,:], file=out_rating)
					print ( iat_arr[i,:], file =out_iat)
