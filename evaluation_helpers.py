import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')


def load_review_data(data_path: str, data_file: str) ->pd.DataFrame:
    # read the .csv file
    df = pd.read_csv(data_path + '/' + data_file)
    return df  


def flagged_data_pivot(df: pd.DataFrame, mode: str) ->pd.DataFrame:

    if mode == 'reviewer':
        var = 'reviewerID'
    elif mode == 'restaurant':
        var = 'restaurantID'
    else:
        raise ValueError('mode can be one of reviewer or restaurant')

    # keep only information of restaurant and flagged review
    flagged_df = df[[var, 'flagged']]

    # reviews are flagged with Y, N meaning fake or not fake review
    flag_mapper = {'Y': 'Y', 'YR': 'Y', 
                   'N': 'N', 'NR': 'N',
                  }
    flagged_df['flagged'].replace(flag_mapper, inplace=True)

    # add the value of 1 to a column to pivot on its entries
    flagged_df['ones'] = [1]*flagged_df.shape[0]

    # pivot the data 
    pivoted_df = pd.pivot_table(data=flagged_df, 
                            values='ones',
                            index=var,
                            columns='flagged',
                            aggfunc=np.sum,
                            fill_value=0)
    
    # add fake review percentage and boolean values 
    pivoted_df['N_review_percent'] = round(pivoted_df['N']/(pivoted_df['N']+pivoted_df['Y']), 4)
    pivoted_df['Y_review_percent'] = 1 - pivoted_df['N_review_percent'] 

    pivoted_df['N_boolean'] = pivoted_df['N']>0
    pivoted_df['Y_boolean'] = pivoted_df['Y']>0  

    return pivoted_df


def read_birdnest_results(top_n: int, output_type: str) ->pd.DataFrame:

    # path of the birdnest results is on data/birdnest_out
    acceptable_types = ['scores', 'ids', 'iat', 'ratings']
    if output_type in acceptable_types:
        df = pd.read_csv(f'data/birdnest_output/yelp/top{top_n}user_{output_type}.txt', header=None)
        if output_type == 'ids':
            df.columns = ['birdnest_flagged_users']
        elif output_type == 'scores':
            df_columns = ['birdnest_flagged_users', 'NEST_scores']
        else:
            df.columns = [f'birdnest_flagged_user_{output_type}']
    else:
        raise ValueError(f'output_type shoud be one of: {", ".join(acceptable_types)}')

    return df


def get_flagged_users_by_percent(pivoted_data: pd.DataFrame, flag_percent: int) ->list:

    # select subset of user that are flagged at least by flag_percent of their reviews
    print(f'Reviewers are considered as fraud if at least {flag_percent*100:.2f}% of their reviews is flagged as fake')
    flagged_users_by_percent = pivoted_data[pivoted_data['Y_review_percent']>flag_percent].reset_index()['reviewerID'].tolist()

    return flagged_users_by_percent


def get_results_accuracy(flagged_users_df: pd.DataFrame, reted_users: pd.DataFrame, flag_percents: list, top_n: list,
                         save_plot: bool=False, print_results: bool=False) -> pd.DataFrame:

    # initialize with empty dictionary to fill with results                    
    results_dict = {}
    for flag_percent in flag_percents:

        yelp_flagged_users = get_flagged_users_by_percent(flagged_users_df, flag_percent)

        for top_n_users in top_n:
            n_yelp_flagged = reted_users.iloc[:top_n_users].isin(yelp_flagged_users).sum()[0]/top_n_users
            
            # update the values of the list
            if 'flag_percent' in results_dict.keys():
                results_dict['flag_percent'].append(f'{int(flag_percent*100)}%')
                results_dict['top_n_users'].append(top_n_users)
                results_dict['perc_correctly_flagged'].append(n_yelp_flagged)
            else:
                results_dict['flag_percent'] = [f'{int(flag_percent*100)}%']
                results_dict['top_n_users'] = [top_n_users]
                results_dict['perc_correctly_flagged'] = [n_yelp_flagged]
    
    # add the values in the dataframe
    df = pd.DataFrame(results_dict)

    # create a stached barchart
    plt.figure(figsize=(8, 5))
    sns.barplot(x = 'top_n_users', y = 'perc_correctly_flagged', hue='flag_percent', data = df, palette='mako')
    plt.legend(loc='lower left')
    plt.xlabel('Top n users rated by birdnest as fraudulent')
    plt.ylabel('% of actual fraud Vs predicted users')
    plt.title('% of users correctly classified as fraud Vs percent of reviews \n needed to classify user as fraud')
    
    if save_plot:
        plt.savefig(f'results.png', bbox_inches='tight')
    plt.show()

    if print_results:
        print(df)
    
    return df


def print_review_score_stats(pivoted_data: pd.DataFrame) ->None:
    
    print(f'Around {sum(pivoted_data["Y_review_percent"]>0)/pivoted_data.shape[0]*100:.2f}% of yelp users'
          f' are considered to have posted at least 1 fraud review')
    
    for n_percent in [0.3, 0.5, 0.7]:
        print(f'Around {sum(pivoted_data["Y_review_percent"]>n_percent)/pivoted_data.shape[0]*100:.2f}% of yelp users'
             f' have over {n_percent*100}% of their reviews, evaluated as fraud reviews')
