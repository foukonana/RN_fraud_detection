import pandas as pd 
from tqdm import tqdm
import pickle
import os 


def birdnest_review_file(df: pd.DataFrame, write_file: bool=True, state: str='') -> pd.DataFrame:

    if 'date' in df.columns:
        # birdnest uses integer values of timestampt to calculate differencies 
        df['date'] = pd.to_datetime(df['date'])
        df['int_date'] = df['date'].apply(lambda dt: int(dt.timestamp()/86400))
        df.drop(columns='date', inplace=True)
    else:
        raise KeyError('Column date is not the dataset columns.')

    if write_file:
        df.to_csv(f'data/birdnest_input/yelp/yelp_network_{state}.csv', header=False, index=False)

    return df


def birdnest_review_dict(df: pd.DataFrame, write_pkl: bool=True, state: str='') -> dict:
    # initialize empty dictionary
    birdnest_input_dict = {}

    # for every reviewer keep a list of tuples (restaurantID, date_of_review, review_rating)
    for reviewer_of_interest in  tqdm(set(df['reviewerID'])):
        reviewer_list = []
        for _, row in df[[reviewer == reviewer_of_interest for reviewer in df['reviewerID']]].iterrows():
            reviewer_list.append((row['restaurantID'], row['date'], row['rating']))

        birdnest_input_dict[reviewer_of_interest] = reviewer_list
    
    if write_pkl:
        # write the data into python2 readable pickle format
        with open(f'data/{state}_reviews.pkl', 'wb') as handle:
            pickle.dump(birdnest_input_dict, handle, protocol=2)
    
    return birdnest_input_dict


def main(state: str, n_entries: int):

    # load the restauran data
    restaurant_df = load_restaurant_data('labeled_datasets/yelpResData')

    # keep data only relavant to one state
    if state in set(restaurant_df['restaurant_state']):
        restaurant_df = restaurant_df[restaurant_df['restaurant_state'] == state]
    else:
        raise ValueError(f'{state} is invalid, as it does not appear in the dataset')


    print(f'The review network of restaurants in {state} state is being constructed..')

    # load the review dataset
    review_df = pd.read_csv('labeled_datasets/review.csv')

    # load reviewer data 
    reviewer_df = load_reviewer_data('labeled_datasets/yelpResData')

    # join the dataframes
    df = review_df.merge(restaurant_df, how='inner', on='restaurantID')
    df = df.merge(reviewer_df, how='inner', on='reviewerID')

    # save the data for ease of access
    # df.to_csv('labeled_datasets/enriched_reviews.csv', index=False)

    # keep only relevant columns for birdnest
    df = df[['reviewerID', 'restaurantID', 'rating', 'date']]

    # there are updates in the date column --> remove for consistency 
    df = df[['Update' not in review_date for review_date in df['date']]]

    # keep reviewers, restaurants with all least N reviews
    df = keep_top_entries(df, n_entries)

    # save the data into csv
    df = birdnest_review_file(df, True, state)

    # print('Constructing input dictionary for birdnest..')
    # birdnest_dict = birdnest_review_dict(df, True, state)


if __name__ == '__main__':
    # states with many entries: CA, NY, FL, MI, TX
    state = str(input('Please input the state for which you want to create the graph: '))
    n_reviews = int(input('Please input the ninimum number of reviews for reviewer and restaurant: '))

    if os.path.exists(f'data/birdnest_input/yelp/yelp_network_{state}.csv'):
        run_script = input('The file you want to create already exists. Do you wish to continue (rewrite the file)? ')
        if run_script in ('Y', 'YES', 'Yes', 'yes', 'y'):
            print('Starting the process.')
            main(state, n_reviews)
        else:
            print('Aborting script.')
    else:
        main(state, n_reviews)
