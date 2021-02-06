import sqlalchemy 
import pandas as pd 
import networkx as nx
from networkx.algorithms import bipartite
import pickle


def sqlite_to_table(db_file: str, table_name: str) ->pd.DataFrame:
    
    e = sqlalchemy.create_engine(f'sqlite:///{db_file}.db')
    df = pd.read_sql_table(table_name, e)
    
    return df


def review_network_from_pandas(df: pd.DataFrame) ->nx.MultiGraph:
        
    if all(name in df.columns for name in ['reviewerID', 'restaurantID', 'rating']):
        # initialize the graph object 
        B = nx.MultiGraph()        
        # add the nodes of the two sets
        B.add_nodes_from(df['reviewerID'].unique().tolist(), bipartite=0, label='reviewer')
        B.add_nodes_from(df['restaurantID'].unique().tolist(), bipartite=1, label='restaurant')
        # define adge and weight list and add them on the network
        edges = [tuple(x) for x in df[['reviewerID','restaurantID','rating']].values.tolist()]
        B.add_weighted_edges_from(edges, weight='rating')

    else:
        raise KeyError('The dataframe does not contain the correct variable to create a review network')

    return B


# function that connects to the database and fetches the appropriate table 
def sqlite_to_table(db_file: str, table_name: str) ->pd.DataFrame:
    
    e = sqlalchemy.create_engine(f'sqlite:///{db_file}.db')
    df = pd.read_sql_table(table_name, e)
    
    return df


def load_restaurant_data(db_file: str) -> pd.DataFrame:

    # load the restauran data from the sqlite database
    df = sqlite_to_table(db_file, 'restaurant')

    # add the state for the restaurant
    df['state'] = [loc.split(',')[-1].strip() for loc in df['location']]

    # rename the dataset to indicate its source table
    df.columns = ['restaurantID'] + [f'restaurant_{col_name}' for col_name in df.columns if col_name != 'restaurantID']

    return df


def load_reviewer_data(db_file: str) -> pd.DataFrame:

    # load reviewer data from sqlite database
    df = sqlite_to_table(db_file, 'reviewer')

    # remove reviewer's location as it is irrelevant of the review
    df.drop(columns='location', inplace=True)

    # rename the dataset to indicate its source table
    df.columns = ['reviewerID'] + [f'reviewer_{col_name}' for col_name in df.columns if col_name != 'reviewerID']

    return df


# function that keeps reviewers and restaurants with at least N reviews
def keep_top_entries(df: pd.DataFrame, N: int) -> pd.DataFrame:
    
    orig_shape = df.shape[0]

    n_rating_per_restaurant=df.groupby('restaurantID')['rating'].count().rename('n_reviews').reset_index()
    restaurants_of_interest = n_rating_per_restaurant[n_rating_per_restaurant['n_reviews']>=N]['restaurantID'].tolist()
    df = df[df['restaurantID'].isin(restaurants_of_interest)]

    n_rating_per_user=df.groupby('reviewerID')['rating'].count().rename('n_reviews').reset_index()
    reviewers_of_interest = n_rating_per_user[n_rating_per_user['n_reviews']>=N]['reviewerID'].tolist()
    df[df['reviewerID'].isin(reviewers_of_interest)]

    print(f'Keeping reviewers and restaurants with at least {N} reviews, we keep {df.shape[0]/orig_shape*100:.2f}% of the data')

    return df
