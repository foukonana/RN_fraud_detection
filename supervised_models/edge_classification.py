import pandas as pd 
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')


def evaluate_model(y_test, prediction, labels_dict, model_name) ->pd.DataFrame:
    # metrics
    accuracy = accuracy_score(y_test, prediction)
    recall = recall_score(y_test, prediction, average='macro')
    precision = precision_score(y_test, prediction, average='macro')
    f1 = f1_score(y_test, prediction, average='macro')

    print('Accuracy score: {:.3f}'.format(accuracy))
    print('Recall score: {:.3f}'.format(recall))
    print('Precision score: {:.3f}'.format(precision))
    print('F1 score: {:.3f}'.format(f1))

    print('Classification report:\n{}'.format(classification_report(y_test, prediction, target_names=list(labels_dict.values()))))

    # confusion matrix
    cm = confusion_matrix(y_test, prediction)

    df_cm = pd.DataFrame(cm, index=list(labels_dict.values()), columns=list(labels_dict.values()))
    plt.figure(figsize=(16,7))
    plt.title(f'Confusion matrix of {model_name} results')
    sns.set(font_scale=1)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.0f')

    #plt.savefig('_matrix.png'.format(title))
    plt.show()

    df = pd.DataFrame(data={'metric': ['accuracy', 'recall', 'precision', 'f1'],
                            'score': [accuracy*100, recall*100, precision*100, f1*100],
                            'model': model_name}) 
    return df 


def review_graph_to_features(selected_location: str) ->pd.DataFrame:

    # load the data from their respective tables
    review_df = pd.read_csv('labeled_datasets/review.csv', 
                             usecols=['reviewerID', 'restaurantID', 'rating','usefulCount', 'coolCount', 'funnyCount', 'flagged'])
    reviewer_df = pd.read_csv('labeled_datasets/reviewer.csv',
                               usecols=['reviewerID', 'reviewer_friendCount', 'reviewer_reviewCount','reviewer_usefulCount', 
                                        'reviewer_coolCount', 'reviewer_funnyCount', 'reviewer_complimentCount', 
                                        'reviewer_tipCount', 'reviewer_fanCount'])
    restaurant_df = pd.read_csv('labeled_datasets/restaurant.csv',
                                 usecols=['restaurantID', 'restaurant_rating',  'restaurant_PriceRange', 'restaurant_state'])
    restaurant_df = restaurant_df[restaurant_df['restaurant_state'] == selected_location].reset_index(drop=True)

    df = pd.merge(review_df, restaurant_df, on='restaurantID', how='inner')
    df = pd.merge(df, reviewer_df, on='reviewerID', how='inner')

    # add reviewer centralities (we are not using centralities for the restaurant)
    with open('graph_data/reviewer_centralities_CA.pkl', 'rb') as f:
        rev_cen = pickle.load(f)

    rev_cen_df = pd.DataFrame(rev_cen).reset_index()
    rev_cen_df.columns = ['reviewerID', 'reviewer_degree_cen', 'reviewer_betweenneess_cen', 'reviewer_closeness_cen']

    df = pd.merge(df, rev_cen_df, on='reviewerID', how='left')
    df.drop(columns=['reviewerID', 'restaurantID', 'restaurant_state'], inplace=True)

    return df


def encode_variables(df :pd.DataFrame) ->pd.DataFrame:

    # encode the ordinal variable
    df.loc[df['restaurant_PriceRange'].isnull(), 'restaurant_PriceRange'] = 'Not Specified'
    price_mapper = {'Not Specified': 0,
                '$': 1, 
                '$$': 2,
                '$$$': 3,
                '$$$$': 4,
                }
    df['restaurant_PriceRange'].replace(price_mapper, inplace=True)

    # reviews are flagged with Y, YR when the review is fake, N, NR when the review is considered honest
    flag_mapper = {'Y': 1, 
                   'YR': 1,
                   'N': 0, 
                   'NR': 0, 
                   }
    df['flagged'].replace(flag_mapper, inplace=True)
    df.rename(columns={'flagged': 'label'}, inplace=True)

    return df


def get_correlation_heatmap(df :pd.DataFrame) ->None:

    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), cmap=plt.cm.Reds, annot=True)
    plt.title('Heatmap displaying the relationship between\nthe features of the data',
              fontsize=13)
    plt.show()


def supervised_models_comparison(df: pd.DataFrame) ->pd.DataFrame:

    X, y = df.drop(columns='label'), df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119, stratify=y)

    # define the models along with the respective grid search parameters
    # logistic regression
    log_reg_clf = LogisticRegression()
    log_reg_params = {}
    # random forest
    rf_clf = RandomForestClassifier()
    rf_params = {'n_estimators': range(20, 36)}
    # naive bayes
    nb_clf = BernoulliNB()
    nb_params = {'alpha': [x * 0.005 for x in range(1, 10)]}


    # calculate metrics for various combinations 
    metrics_df = pd.DataFrame()
    for model_name, model, model_params in zip(['Logistic Regression', 'Random Forest', 'Naive Bayes'],
                                            [log_reg_clf, rf_clf, nb_clf],
                                            [log_reg_params, rf_params, nb_params]):
        print(f'Calculating results of {model_name} classifier')
        scaler = MinMaxScaler()
        edge_pipe = Pipeline([('scaling', scaler),
                            ('classifier', model)])
        # grid search 
        grid = GridSearchCV(model,
                        param_grid=model_params,
                        cv=5,
                        n_jobs=-1)
        grid.fit(X_train, y_train)
        # change the classifier in the pipe to the best classifier after frid search 
        edge_pipe = Pipeline([('scaling', scaler),
                            ('classifier',  grid.best_estimator_)])

        edge_pipe.fit(X_train, y_train)

        # results
        y_pred = edge_pipe.predict(X_test)
        results_df = evaluate_model(y_test, y_pred, labels_dict={0: 'Honest', 1: 'Fraud'}, model_name=model_name)
        metrics_df = metrics_df.append(results_df)

    return metrics_df
    

def main():

    # load the dataset used for the edge classification task 
    df = review_graph_to_features(selected_location='CA')
    # process the categorical data
    df = encode_variables(df)

    metrics_df = supervised_models_comparison(df)

    # plot all the scores
    plt.figure(figsize=(8,6))
    sns.barplot(x='metric', y='score', hue='model', data=metrics_df, palette='ocean')
    plt.title('Resulting metrics of different models')
    plt.show()


if __name__ == '__main__':
    main()
