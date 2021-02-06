import pandas as pd
import numpy as np 
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join('.')))
from graph_creation_helpers import sqlite_to_table
import warnings 
warnings.filterwarnings('ignore')


def calc_max_reviews_in_day(df: pd.DataFrame) ->pd.DataFrame:

    reviews_df = df[['Update' not in x for x in df['date']]].groupby(['reviewerID','date'])['reviewID'].count().reset_index().rename(columns={'reviewID': 'n_reviews'})
    max_rev_df = reviews_df.groupby('reviewerID')['n_reviews'].max().reset_index().rename(columns={'n_reviews': 'max_reviews_in_day'})
    
    return max_rev_df


def cal_review_ratios(df: pd.DataFrame) ->pd.DataFrame:

    # add a value of 1 to be summed while the data are pivoted
    df['val'] = 1

    # pivot the data 
    rating_df = pd.pivot_table(df, index='reviewerID', columns=['rating'], values='val', aggfunc=np.sum).fillna(0)

    # find positive review (4,5 star reviews) and negative review (1,2 stars) ratio per user
    rating_df['tot_reviews'] = rating_df[1] + rating_df[2] + rating_df[3] + rating_df[4] + rating_df[5]
    rating_df['pos_rev_ratio'] = (rating_df[4] + rating_df[5])/rating_df['tot_reviews'] 
    rating_df['neg_rev_ratio'] = (rating_df[1] + rating_df[2])/rating_df['tot_reviews'] 

    return rating_df


def calc_rating_deviation(df: pd.DataFrame, weighted: bool=True) ->pd.DataFrame:

    # remove columns where the review is an update
    rating_dev_df = df[['Update' not in x for x in df['date']]][['reviewerID', 'restaurantID', 'date', 'rating', 'restaurant_rating']]

    # absolute rating deviation = |rating r{i,j} - mean(r{j})|
    rating_dev_df['abs_rating_deviation'] = abs(rating_dev_df['restaurant_rating'] - rating_dev_df['rating'])

    # calculate average rating deviation
    rating_dev_group_df = rating_dev_df.groupby('reviewerID')['abs_rating_deviation'].mean().reset_index().rename(columns={'abs_rating_deviation': 'avg_rating_deviation'})

    # merge back on the original data
    rating_dev_df = rating_dev_df.merge(rating_dev_group_df, on='reviewerID', how='left')

    # weighted rating deviation
    if weighted:
        # make date a datetime object
        rating_dev_df['date'] = pd.to_datetime(rating_dev_df['date'])

        # find the ranking of dates when restaurants are reviewed
        date_rank_series = rating_dev_df.sort_values(['restaurantID','date']).groupby('restaurantID')['date'].rank()
        date_rank_series.name = 'date_rank'

        # add back to the data
        rating_dev_df = pd.concat([rating_dev_df, date_rank_series], axis=1)

        # add the date rank weight
        rating_dev_df['date_rank_weight'] = rating_dev_df['date_rank'].apply(lambda x: 1/pow(x, 1.5))

        rating_dev_df['weighted_abs_rating_deviation'] = (rating_dev_df['abs_rating_deviation'] * rating_dev_df['date_rank_weight'])

        # calculate the weighted deviation per reviewer and merge back to original data
        grouped_weighted_dev_df = rating_dev_df.groupby('reviewerID')[['weighted_abs_rating_deviation', 'date_rank_weight']].sum().reset_index().rename(columns={'weighted_abs_rating_deviation': 'sum_weighted_abs_rating_deviation', 'date_rank_weight': 'sum_date_rank_weight'})

        grouped_weighted_dev_df['weighted_rating_deviation'] = grouped_weighted_dev_df['sum_weighted_abs_rating_deviation']/grouped_weighted_dev_df['sum_date_rank_weight']

        rating_dev_df = rating_dev_df.merge(grouped_weighted_dev_df[['reviewerID', 'weighted_rating_deviation']], on='reviewerID', how='left')

    return rating_dev_df


def calc_rating_entropies(df: pd.DataFrame) ->pd.DataFrame:
    # add a value of 1 to be summed while the data are pivoted
    df['val'] = 1

    # pivot the data 
    rating_df = pd.pivot_table(df, index='reviewerID', columns=['rating'], values='val', aggfunc=np.sum).fillna(0)

    # find positive review (4,5 star reviews) and negative review (1,2 stars) ratio per user
    rating_df['tot_reviews'] = rating_df[1] + rating_df[2] + rating_df[3] + rating_df[4] + rating_df[5]

    # calculate entropies of observed rating distribution of reviewer
    for n in range(1,6):
        rating_df[f'entropy_{n}'] = -(rating_df[n]/rating_df['tot_reviews']) * np.log2(rating_df[n]/rating_df['tot_reviews'])

    entropies_df  = pd.melt(rating_df.reset_index()[['reviewerID', 'entropy_1', 'entropy_2', 'entropy_3', 'entropy_4', 'entropy_5']],
                            id_vars=['reviewerID'],
                            value_vars=['entropy_1', 'entropy_2', 'entropy_3', 'entropy_4', 'entropy_5']).dropna()
    # overwrite rating 
    entropies_df['rating'] = entropies_df['rating'].apply(lambda x: int(x.split('_')[1]))
    entropies_df.rename(columns={'value': 'entropy'}, inplace=True)

    return entropies_df


def calc_from_cdf(df: pd.DataFrame, features_list: list, susp_when: str='high') ->pd.DataFrame:

    if susp_when not in ['high', 'low']:
        raise ValueError(f'{susp_when} is invalid for susp_when. \n Select one from "high", "low"')

    N =df.shape[0]
    for feature in features_list:
        # find the commulative distribution from the observed data
        cdf = df[feature].tolist()
        # find the probability of the value in the feature (where it is positioned in the distribution)
        if susp_when == 'high':
            df[f'f_{feature}'] = df[feature].apply(lambda x: 1 - sum([xi <=x for xi in cdf])/N)
        elif susp_when == 'low':
            df[f'f_{feature}'] = df[feature].apply(lambda x: sum([xi <=x for xi in cdf])/N)
        
    return df 


def calc_reviewer_priors(state: str):

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
    df = joined_df.loc[joined_df['restaurant_state']==state, 
                        ['reviewerID', 'restaurantID', 'reviewID', 'date', 'rating', 'restaurant_rating']]

    # calculate max reviews
    max_rev_df = calc_max_reviews_in_day(df)

    # pos/neg review ratios
    rating_df = cal_review_ratios(df)

    # rating deviations (including weighted version)
    rating_dev_df = calc_rating_deviation(df, True)

    # entropies of reviews
    entropies_df = calc_rating_entropies(df)

    # Add distributions
    max_rev_df = calc_from_cdf(max_rev_df, ['max_reviews_in_day'], 'high')
    rating_df = calc_from_cdf(rating_df, ['pos_rev_ratio', 'neg_rev_ratio'], 'high')
    entropies_df = calc_from_cdf(entropies_df, ['entropy'], 'low')
    # change the rating_deviation_df to be per user
    rating_deviation_df = rating_dev_df.groupby('reviewerID')[['avg_rating_deviation', 'weighted_rating_deviation']].max().reset_index()
    rating_deviation_df = calc_from_cdf(rating_deviation_df, ['avg_rating_deviation', 'weighted_rating_deviation'], 'high')

    # Cumpute suspiciousness score
    suspiciousness_features_df = max_rev_df[['reviewerID', 'f_max_reviews_in_day']].merge(rating_df.reset_index()[['reviewerID', 'f_pos_rev_ratio', 'f_neg_rev_ratio']], on='reviewerID', how='left').merge(rating_deviation_df[['reviewerID', 'f_avg_rating_deviation', 'f_weighted_rating_deviation']], on='reviewerID', how='left').merge(entropies_df[['reviewerID', 'f_entropy']], on='reviewerID', how='left')

    N = suspiciousness_features_df.shape[1]-1
    suspiciousness_features_df['reviewer_suspiciousness'] = 1 - np.sqrt(suspiciousness_features_df[[col for col in suspiciousness_features_df.columns if col != 'reviewerID']].pow(2).sum(axis=1)/N)

    # save the information in a dictionary 
    reviewer_priors = {reviewer: suspiciousness for reviewer, suspiciousness in zip(suspiciousness_features_df['reviewerID'].tolist(), suspiciousness_features_df['reviewer_suspiciousness'].tolist())}
    print('Saving the calculated priors of reviewer suspiciousness')
    with open(f'data/speagle_input/reviewer_priors_{state}.pkl', 'wb') as handle:
        pickle.dump(reviewer_priors, handle)


if __name__ == '__main__':
    # states with many entries: CA, NY, FL, MI, TX
    state = str(input('Please input the state for which restaurant reviewers you want to calculate the priors: '))

    if os.path.exists(f'labeled_datasets/reviewer_priors_{state}.pkl'):
        run_script = input('The file you want to create already exists. Do you wish to continue (rewrite the file)? ')
        if run_script in ('Y', 'YES', 'Yes', 'yes', 'y'):
            print('Starting the process.')
            calc_reviewer_priors(state)
        else:
            print('Aborting script.')
    else:
        print('Starting the process.')
        calc_reviewer_priors(state)
