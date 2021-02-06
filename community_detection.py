import community
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import numpy as np
import glob
import ujson
from collections import defaultdict
import itertools
import sys
import pickle5 as pickle
from sklearn.cluster import SpectralClustering
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from graph_stats import get_dense_graph, set_degree_distributions
from graph_per_state import review_network_from_pandas
from create_birdnest_input import load_restaurant_data


def compute_louvain(unGraph):
    print('Louvain')
    print('Computing parition.....')
    t = datetime.now()
    partition = community.best_partition(unGraph)
    time_taken = datetime.now() - t

    print('Converting results.......')
    louvain_results = defaultdict(list)
    for node, comm in sorted(partition.items()):
        louvain_results[comm].append(node)

    print('Computing modularity.....')
    modularity = nx.algorithms.community.modularity(unGraph, louvain_results.values())

    extract_results('louvain', unGraph, louvain_results, time_taken, modularity)

    print('Calculate color values.....')
    values = [partition.get(node) for node in unGraph.nodes()]

    visualize('Louvain', unGraph, values)

    print('Louvain Finished.....')


def compute_label_propagation(unGraph):
    print('Label Propagation')
    print('Computing partition.....')
    t = datetime.now()
    lbcomms = nx.algorithms.community.label_propagation.label_propagation_communities(unGraph)
    time_taken = datetime.now() - t

    print('Converting results.....')
    ress = {}
    res = list(lbcomms)
    count = 0
    for item in res:
        ress[str(count)] = list(item)
        count += 1

    print('Computing modularity.....')
    modularity = nx.algorithms.community.modularity(unGraph, ress.values())

    extract_results('label_propagation', unGraph, ress, time_taken, modularity)

    values = calc_color_values(unGraph, ress)

    visualize('Label Propagation', unGraph, values)

    print('Label Propagation Finished.....')


def compute_k_clique(unGraph, k):
    print('K-Clique')
    print('Computing parition for ' + str(k) + '-sized cliques.....')
    t = datetime.now()
    comms = list(nx.algorithms.community.k_clique_communities(unGraph, k))
    time_taken = datetime.now() - t

    print('Converting results.....')
    count = 0
    ress = dict()
    for item in comms:
        ress[str(count)] = list(item)
        count += 1

    modularity = None
    print('Computing modularity.....')
    try:
        modularity = nx.algorithms.community.modularity(unGraph, ress.values())
    except Exception as e:
        print(repr(e))

    extract_results('kClique_' + str(k), unGraph, ress, time_taken, modularity)

    values = []
    for i in unGraph.nodes():
        for key, value in ress.items():
            if i in value:
                values.append(int(key))
                break

    proc = False
    info = ''
    if len(values) != len(unGraph.nodes()):
        diff = len(unGraph.nodes()) - len(values)
        color = len(ress.values()) + 1
        non_exist = [color for i in range(diff)]
        new_vals = values + non_exist
        info = 'Nodes without a community: ' + str(diff) + ' with label: ' + str(color) + '\n'
        proc = True

    if proc:
        print('Colors added')
        values = new_vals

    print(len(values))
    extract_results('kClique_' + str(k), unGraph, ress, time_taken, modularity, extra_text=info)

    visualize('K-Clique for ' + str(k) + ' sized clusters', unGraph, values)

    print('K-Clique Finished.....')


def compute_clauset_newman_moore(unGraph):
    print('Clauset-Newman-Moore')
    print('Computing partition.....')
    t = datetime.now()
    c = list(nx.algorithms.community.greedy_modularity_communities(unGraph))
    time_taken = datetime.now() - t

    print('Converting results.....')
    ress = dict()
    count = 0
    for item in (c):
        ress[str(count)] = list(item)
        count += 1

    print('Computing modularity.....')
    modularity = nx.algorithms.community.modularity(unGraph, ress.values())

    extract_results('CNM', unGraph, ress, time_taken, modularity)

    values = calc_color_values(unGraph, ress)

    visualize('Clauset-Newman-Moore', unGraph, values)

    print('Clauset-Newman-Moore finished.....')


def compute_spectral_clustering(unGraph :nx.Graph, n_clusters: int, plot_results: bool=False) ->None:
    # np.set_printoptions(threshold=sys.maxsize)

    print(f'Computing spectral Clustering for {n_clusters} clusters....')

    A = nx.convert_matrix.to_numpy_matrix(unGraph)

    nodes = list(unGraph.nodes())
    nodes.sort()

    starting_time = datetime.now()
    clusters = SpectralClustering(affinity='precomputed',
                                  assign_labels="kmeans", random_state=0, n_clusters=n_clusters).fit_predict(A)
    time_taken = datetime.now() - starting_time

    if plot_results:
        visualize(f'Spectral Clustering {n_clusters}', unGraph, clusters)

    print('Converting results....')
    results = defaultdict(list)
    for node, cluster in zip(nodes, clusters):
        results[str(cluster)].append(node)

    print('Computing modularity....')
    modularity = None
    try:
        modularity = nx.algorithms.community.modularity(unGraph, results.values())
    except Exception as e2:
        print(repr(e2))

    extract_results(f'spectral_clustering_{n_clusters}', unGraph, results, time_taken, modularity)

    print('Spectral Clustering finished....')


def calc_color_values(graph, data):
    print('Calculate color values.....')
    values = []
    for i in graph.nodes():
        for k, v in data.items():
            for e in v:
                if i == e:
                    values.append(int(k))
    return values


def extract_results(alg_name, graph, data, time, modularity, **kwargs):
    _max_l = list()
    _min_l = list()
    error = False
    if modularity is None:
        error = True
    extra_text = kwargs.get('extra_text', None)

    print('Writing results.....')
    with open(alg_name + '_results.txt', 'a') as f:
        f.write('Communities Computed: \n')
        f.write(ujson.dumps(data))
        f.write('================================================================\n')
        f.write('Information about computed partition: \n')
        try:
            if error:
                f.write('modularity= ' + str(modularity) + '\n')
            else:
                f.write('modularity= ' + str('{:f}'.format(modularity)) + ' or ' + str(modularity) + '\n')
        except Exception as e3:
            print(repr(e3))
        f.write('time taken to compute: ' + str(time) + '\n')
        cnt = 0
        d = 0
        for item in data:
            cnt += 1
            f.write('For community ' + str(item) + ':\n')
            subG = graph.subgraph(data[item])
            f.write('NumOfEdges ' + str(len(subG.edges())) + '\n')
            f.write('NumOfNodes ' + str(len(subG.nodes())) + '\n')
            density = nx.classes.function.density(subG)
            d += density
            f.write('Density of community = ' + str(density) + '\n')
            try:
                _max = max(dict(subG.degree()).items(), key=lambda x: x[1])
                _min = min(dict(subG.degree()).items(), key=lambda x: x[1])
                _max_l.append(_max[1])
                _min_l.append(_min[1])
                f.write('Max (node, degree) -> ' + str(_max) + '\n')
                f.write('Min (node, degree) -> ' + str(_min) + '\n')
            except Exception as e:
                print(repr(e))
            if len(data.keys()) != cnt:
                f.write('Next community info===============================================\n')

        f.write('Information after computing.....\n')
        try:
            f.write('Max degree = ' + str(max(_max_l)) + '\n')
            f.write('Min degree = ' + str(min(_min_l)) + '\n')
        except Exception as e:
            print(repr(e))
        f.write('Average Density = ' + str(d / len(data)) + '\n')
        if extra_text is not None:
            print(extra_text)
            f.write('Extra Info: \n')
            f.write(extra_text)


def visualize(alg_name, graph, data):
    print('Visualizing results....')
    # commented out the following part as it is slow
    # nx.draw_spring(graph, cmap=plt.get_cmap('jet'), node_color=data,
    #                node_size=10, with_labels=False)
    # plt.show()
    # nx.draw_circular(graph, cmap=plt.get_cmap('jet'), node_color=data,
    #                  node_size=10, with_labels=False)
    # plt.show()

    nodes = list(graph.nodes())
    nodes.sort()

    plt.scatter(nodes, data, c=data, s=10, cmap='jet')
    plt.margins(x=0)
    plt.title(alg_name)
    plt.xlabel('Nodes')
    plt.ylabel('Community ID')
    plt.show()


def main():

    # load the review table
    review_df = pd.read_csv('data_no_share/review.csv')
    # load restaurant data and add state information  
    restaurant_df = pd.read_csv('data_no_share/restaurant.csv')
    # join the dataframes and keep only information regarding CA
    joined_df = review_df.merge(restaurant_df[['restaurantID', 'restaurant_state']], how='left', on='restaurantID')
    reviews_of_interest_df = joined_df.loc[joined_df['restaurant_state']=='CA', 
                                           ['reviewerID', 'restaurantID', 'restaurant_state', 'rating', 'flagged']]
    # create the bipartite graph 
    B = review_network_from_pandas(reviews_of_interest_df)
    G = nx.Graph(B)

    reviewers = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    restaurants = set(G) - reviewers
    print(round(bipartite.density(G, reviewers), 2))

    print(f'There are {len(reviewers)} distinct reviewers in the dataset and {len(restaurants)} distinct restaurants')

    # plot degree distributions of reviewers and restaurants
    node_set_names = ['reviewers', 'restaurants']
    reviewers_degrees, restaurants_degrees = set_degree_distributions(G, reviewers, restaurants, node_set_names, True)
    print(f'The average degree of a reviewer is {np.mean(reviewers_degrees):.2f}'
          f' while the average degree of a restaurant {np.mean(restaurants_degrees):.2f}')

    # get a network where every reviewer ahs at least 5 reviews and every restaurant is reviewed by at least 5 reviewers
    G = get_dense_graph(G, reviewers, reviewers_degrees, 5, restaurants, restaurants_degrees, 5,
                        node_set_names, new_network_name='graph_data/review_graph', save_new_graph=False)
    
    # find the reviewer and restaurant nodes again as many may have been removed
    reviewers = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    restaurants = set(G) - reviewers

    #print(round(bipartite.density(G, restaurants), 2))
    D_reviewers = bipartite.projected_graph(G, reviewers)
    D_restaurants = bipartite.projected_graph(G, restaurants)
    # nx.write_gpickle(D, 'dense_new_york.gpickle')

    #k_values = [10, 9, 8, 7, 6]
    spectral_values = [5, 10, 15]

    # compute_louvain(D_reviewers)
    #
    # compute_label_propagation(D_reviewers)
    #
    # compute_clauset_newman_moore(D_reviewers)

    # for k in k_values:
    #     compute_k_clique(D_reviewers, k)

    for n_clusters in spectral_values:
        compute_spectral_clustering(D_restaurants, n_clusters, True)


if __name__ == '__main__':
    main()
