import pandas as pd 
import numpy as np 
import networkx as nx
from SpEagle import *
import pickle
import os 
import gc
import sys
sys.path.append(os.path.abspath(os.path.join('.')))
from graph_creation_helpers import sqlite_to_table
from unsupervised_models.helpers.generate_reviewer_priors import *



# I assume there is no knowledge of the reviewer and the restaurant so their ground truth is set as 0 -->honest
def graph_features_data(df: pd.DataFrame) ->pd.DataFrame:
    # initialize the dictionaries
    reviewer_data = {}
    restaurant_data = {}

    for _, review_info in df.iterrows():
        u_id = review_info['reviewerID']
        r_id = review_info['restaurantID']
        rating = review_info['rating']
        label=0
        date = review_info['date']

        # update the dictionaries
        if u_id not in reviewer_data:
            reviewer_data[u_id] = []
        reviewer_data[u_id].append((r_id, rating, label, date))

        if r_id not in restaurant_data:
            restaurant_data[r_id] = []
        restaurant_data[r_id].append((u_id, rating, label, date))
    
    return reviewer_data, restaurant_data


def graph_from_dict(reviewer_data: dict) ->nx.Graph:
    graph_dict = dict()
    for key, value in reviewer_data.items():
        graph_dict[key] = dict()
        for line in value:
            graph_dict[key][line[0]] = {'rating': line[1], 'label': line[2], 'date': line[3]}

    G = nx.Graph(graph_dict)
    return G


def add_attributes_on_graph(G: nx.Graph, reviewer_priors: dict, reviewer_data: dict, restaurant_data: dict) ->nx.Graph:

    # priors for reviewer
    node_attr = dict()
    for k, v in reviewer_priors.items():
        node_attr[k] = {'prior': v, 'types': 'reviewer', 'label': 0}
    # add nodes new attributes to the graph
    add_attribute_to_graph(graph=G, attribute=node_attr, adding_type='node')

    # priors for restaurants (default value 0.5)
    node_attr = dict()
    for k in restaurant_data.keys():
        node_attr[k] = {'prior': 0.5, 'types': 'restaurant'}
    # add nodes' new attributes to the graph
    add_attribute_to_graph(graph=G, attribute=node_attr, adding_type='node')

    # adge attributes (priors of review, default 0.5)
    edge_attr = dict()
    for key, value in reviewer_data.items():
        for line in value:
            edge_attr[(key, line[0])] = {'prior': 0.5, 'types': 'review'}
    add_attribute_to_graph(graph=G, attribute=edge_attr, adding_type='edge')

    return G


def main(state: str):

    print('Loading the data.')
    # load the restauran data
    restaurant_df = sqlite_to_table('labeled_datasets/yelpResData', 'restaurant')
    restaurant_df.columns = ['restaurantID'] + [f'restaurant_{col}' for col in restaurant_df.columns if col!='restaurantID']
    restaurant_df['restaurant_state'] = [loc.split(',')[-1].strip() for loc in restaurant_df['restaurant_location']]

    # keep data only relavant to one state
    if state in set(restaurant_df['restaurant_state']):
        restaurant_df = restaurant_df[restaurant_df['restaurant_state'] == state]
    else:
        raise ValueError(f'{state} is invalid, as it does not appear in the dataset')

    # load the review dataset
    review_df = pd.read_csv('labeled_datasets/review.csv')

    # merge the information 
    joined_df = review_df.merge(restaurant_df[['restaurantID', 'restaurant_rating', 'restaurant_state']], how='left', on='restaurantID')

    df = joined_df.loc[['Update' not in date for date in joined_df['date']], 
                    ['reviewerID', 'restaurantID', 'rating', 'date']]

    # use integer values of timestamp
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].apply(lambda dt: int(dt.timestamp()/86400))

    # clean up
    restaurant_df, review_df, joined_df = None, None, None
    gc.collect()

    # load the priors for reviewers or calculate if they do not exist
    if os.path.exists(f'data/speagle_input/reviewer_priors_{state}.pkl'):
        print('Priors for user, restaurant and review suspiciousness have already been calculated')
        run_script = input('Do you wish to overwrite the file? ')
        if run_script in ('Y', 'YES', 'Yes', 'yes', 'y'):
            print('Calculating prior probabilities of suspiciousness for yelp reviewers')
            calc_reviewer_priors(state)
    else:
        calc_reviewer_priors(state)
    # after the priors are calculated and saved load them
    print('Loading user prior probabilities of suspiciousness')
    with open(f'data/speagle_input/reviewer_priors_{state}.pkl', 'rb') as handle:
        reviewer_priors = pickle.load(handle)

    print('Adding features to nodes of the graph')
    # For every node (reviewer, restaurant) the rating, the timestamp of the and the label is add (fraud Vs honest, default=honest)
    reviewer_data, restaurant_data = graph_features_data(df)

    # create the graph and add attributes on nodes and edges
    G = graph_from_dict(reviewer_data)
    G = add_attributes_on_graph(G, reviewer_priors, reviewer_data, restaurant_data)

    print('Running SpEagle. The process takes some time. Please be patient.')
    # run SpEagle
    # input parameters: numerical_eps, eps, num_iters, stop_threshold
    numerical_eps = 1e-5
    eps = 0.1
    user_review_potential = np.log(np.array([[1 - numerical_eps, numerical_eps], [numerical_eps, 1 - numerical_eps]]))
    review_product_potential = np.log(np.array([[1 - eps, eps], [eps, 1 - eps]]))
    potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
                  'r_p': review_product_potential, 'p_r': review_product_potential}
    stop_threshold = 1e-3

    model = SpEagle(G, potentials, message=None, max_iters=4)

    # new runbp func
    model.schedule(schedule_type='bfs')

    iter = 0
    num_bp_iters = 2
    model.run_bp(start_iter=iter, max_iters=num_bp_iters, tol=stop_threshold)

    _, reviewBelief, _ = model.classify()

    reviewer_beliefs = sorted([belief[0][0] for belief in reviewBelief.items()], key=lambda x: x[1], reverse=True)
    speagle_flagged_reviewers = []
    for reviewer in reviewer_beliefs:
        if reviewer not in speagle_flagged_reviewers:
            speagle_flagged_reviewers.append(reviewer)

    with open(f'data/speagle_output/speagle_user_ids_{state}.pkl', 'wb') as file:
        pickle.dump(speagle_flagged_reviewers, file)


if __name__ == '__main__':

    # states with many entries: CA, NY, FL, MI, TX
    state = str(input('Please input the state for which you want to run SpEagle: '))

    if os.path.exists(f'data/speagle_output/speagle_user_ids_{state}.pkl'):
        run_script = input(f'SpEagle has already run for {state}. Do you wish to continue (rewrite the file)? ')
        if run_script in ('Y', 'YES', 'Yes', 'yes', 'y'):
            print('Starting the process.')
            main(state)
        else:
            print('Aborting script.')
    else:
        main(state)
