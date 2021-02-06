import pandas as pd
import numpy as np 
import scipy.sparse
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm import tqdm
import pickle
import gc


def general_info(review_bipart_network: nx.MultiGraph, network_name: str) -> None:

    if bipartite.is_bipartite(review_bipart_network):
        bip= 'is'
    else:
        bip = 'is not'
    print(f'The {network_name} {bip} bipartite')

    if nx.is_connected(review_bipart_network):
        bip= 'is'
    else:
        bip = 'is not'
    print(f'The {network_name} {bip} connected')


def get_node_sets(review_bipart_network: nx.MultiGraph) -> (list, list):
    
    top_node_set = set(n for n,d in review_bipart_network.nodes(data=True) if d['bipartite']==0)
    bottom_node_set = set(n for n,d in review_bipart_network.nodes(data=True) if d['bipartite']==1)

    return top_node_set, bottom_node_set


def get_degree_centrality(G: nx.Graph) ->dict:

    return nx.degree_centrality(G)


def get_betweenness_centrality(G: nx.Graph, approximation: bool=False) ->dict:
    
    if approximation:
        betweenness_cen = nx.betweenness_centrality(G, k=2000)
    else:
        betweenness_cen = nx.betweenness_centrality(G)

    return   betweenness_cen


def get_closeness_centrality(G: nx.Graph, approximation: bool=False) ->dict:

    if approximation:

        node_list = np.array(G.nodes())
        A = nx.adjacency_matrix(G).tolil()
        D = scipy.sparse.csgraph.floyd_warshall(A, directed=False, unweighted=False)

        n = D.shape[0]
        closeness_cenrtality = {}

        for r in tqdm(range(n)):
            cc = 0.0

            possible_paths = list(enumerate(D[r, :]))
            shortest_paths = dict(filter(lambda x: not x[1] == np.inf, possible_paths))

            total = sum(shortest_paths.values())
            n_shortest_paths = len(shortest_paths) - 1

            if total > 0 and n > 1:
                s = n_shortest_paths/(n-1)
                cc = (n_shortest_paths/total)*s
            
            closeness_cenrtality[r] = cc
        
        closeness_cen = {}
        for n, node in enumerate(node_list):
            closeness_cen[node] = closeness_cenrtality[n]
    
    else:
        closeness_cen = nx.closeness_centrality(G, wf_improved = True)
    
    return closeness_cen


def measures_for_centrality(G: nx.Graph) ->(dict, dict):

    # calculate node sets
    G_top_node_set = set(n for n,d in G.nodes(data=True) if d['bipartite']==0)
    G_bottom_node_set = set(G) - G_top_node_set

    print('Calculating degree centrality')
    degree_cen = get_degree_centrality(G)

    if len(G_top_node_set) < 2000:
        print('Calculating betweenness centrality. Please wait..')
        betweenness_cen = get_betweenness_centrality(G)
    else:
        # approximation of betweenness centrality
        print('Calculating approximation of betweenness centrality. Please wait..')
        betweenness_cen = get_betweenness_centrality(G, approximation=True)

    if len(G_top_node_set) < 2000:
        print('Calculating closeness centrality. Please wait..')
        closeness_cen = get_closeness_centrality(G)
    else:
        print('Calculating approximation of closeness centrality. Please wait..')
        # add approximation using floyd-warshal method for adjastency matrix 
        # https://medium.com/@pasdan/closeness-centrality-via-networkx-is-taking-too-long-1a58e648f5ce

        closeness_cen = get_closeness_centrality(G, approximation=True)        

    #page_rank = nx.pagerank(G, alpha=0.8)
    #hubs, authorities = nx.hits(review_bipart_network)

    # keep centrality measures per node set
    top_node_set_centralities = {}
    bottom_node_set_centralities = {}

    # for centrality_dict, centrality_name in zip([degree_cen, betweenness_cen, closeness_cen, page_rank], 
    #                                             ['Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality', 'Page_Rank']):
    for centrality_dict, centrality_name in zip([degree_cen, betweenness_cen, closeness_cen], 
                                                ['Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']):

        # centrality measures of nodes in top node set
        top_node_centralities = {node_id: centrality for node_id, centrality in centrality_dict.items() if node_id in G_top_node_set}
        # centrality measures of nodes in bottom node set
        bottom_node_centralities = {node_id: centrality for node_id, centrality in centrality_dict.items() if node_id in G_bottom_node_set}
        centrality_dict = None
        gc.collect()

        # save in distinct dictionaries
        top_node_set_centralities.update({centrality_name: top_node_centralities})
        bottom_node_set_centralities.update({centrality_name: bottom_node_centralities})

    return top_node_set_centralities, bottom_node_set_centralities


def plot_centralities(node_set_centralities: dict, node_set_name: str,
                      split_centralities_on_flag: bool=False, flagged_ids: list=None,
                      savefig: bool=False, figname: str=None) -> None:
    
    if not split_centralities_on_flag:
        fig, axs = plt.subplots(nrows=1, ncols=len(node_set_centralities), sharey=True, sharex=False, figsize=(14,4))

        for plot_counter, centrality_measure in enumerate(node_set_centralities.keys()):
            sns.histplot(list(node_set_centralities[centrality_measure].values()), bins=30, ax=axs[plot_counter])
            axs[plot_counter].set_xlabel(centrality_measure)
            plt.yscale('log', basey=10)

        plt.suptitle(f'Histograms of centrality measures of {node_set_name} node set', size=12)
        axs[0].set_ylabel('Number of nodes')
        plt.tight_layout()

    
    else:
        if not flagged_ids:
            raise NameError('flagged_ids list is not defined')

        # transform the data into a pandas dataframe
        df = get_dataframe_from_centralities_dict(node_set_centralities, node_set_name)
        # add flag column 
        df[f'flagged_{node_set_name}'] = df[node_set_name].isin(flagged_ids)

        fig, axs = plt.subplots(nrows=1, ncols=len(node_set_centralities), sharey=True, sharex=False, figsize=(14,4))

        for plot_counter, centrality_measure in enumerate(node_set_centralities.keys()):
            sns.histplot(df[~df[f'flagged_{node_set_name}']][centrality_measure], 
                         bins=30, color='steelblue', alpha=0.5, label='Real', ax=axs[plot_counter])
            sns.histplot(df[df[f'flagged_{node_set_name}']][centrality_measure], 
                         bins=30, color='firebrick', alpha=0.5, label='Fake', ax=axs[plot_counter])
            axs[plot_counter].set_xlabel(centrality_measure)
            axs[plot_counter].legend(loc='upper right')
            plt.yscale('log', basey=10)

        plt.suptitle(f'Histograms of centrality measures of {node_set_name} node set per flagged group', size=12)
        axs[0].set_ylabel('Number of nodes')
        #axs[0].tight_layout()

    if savefig:
        plt.savefig(f'graph_plots/{figname}.png', bbox_inches='tight')

    plt.show()


def get_dataframe_from_centralities_dict(node_set_centralities: dict, node_set_name:str) ->pd.DataFrame:
    # the dictionary contains dictionaries of centralities
    # the keys of these dictionaries are the same between all of them 
    centrality_name = list(node_set_centralities.keys())[0]
    # initialize 
    centralities_df = pd.DataFrame({node_set_name: list(node_set_centralities[centrality_name].keys())})

    for centrality_name in node_set_centralities.keys():
        df = pd.DataFrame.from_dict(data=node_set_centralities[centrality_name], orient='index')
        df = df.reset_index()
        df.columns=[node_set_name, centrality_name]

        # merge the new information 
        centralities_df = centralities_df.merge(df, on=node_set_name, how='inner')

    return centralities_df


def common_neighbors_plot(G: nx.Graph, node_set: set, node_set_name: str, other_node_set_name: str, 
                          savefig: bool=False, figname: str=None) ->None:

    # initialize a second node set                       
    G_set = node_set
    G_n_neighbors = []
    print('Calculating common neighbors. Be patient..')
    for u in tqdm(node_set):
        # remove an item from the second node set with each iteration
        G_set = G_set - {u}
        if len(G_set) > 0:
            G_n_neighbors.append([len(list(nx.common_neighbors(G, u, v))) for v in G_set])
    # flaten the list
    G_n_neighbors = [item for sublist in G_n_neighbors for item in sublist]

    # plot the resutls
    plt.figure(figsize=(6, 5))
    sns.histplot(G_n_neighbors)
    plt.yscale('log', basey=10)
    plt.xscale('log', basex=10)
    plt.title(f'Log-log distribution plot of {node_set_name}\' common {other_node_set_name}')

    if savefig:
        plt.savefig(f'graph_plots/{figname}.png', bbox_inches='tight')

    plt.show()


def main():

    G = nx.read_gpickle('graph_data/review_graph.gpickle')
    B = nx.Graph(G)

    # get centrality values for every node set in the network
    reviewer_centralities, restaurant_centralities = measures_for_centrality(B)
    # save the centrality dictionaries
    for centrality_dict, dict_name in zip([reviewer_centralities, restaurant_centralities], ['reviewer_centralities', 'restaurant_centralities']):
        with open(f'graph_data/{dict_name}.pkl', 'wb') as f:
            pickle.dump(centrality_dict, f)

    # plot the centrality values distribution 
    plot_centralities(reviewer_centralities, 'reviewer', True, 'reviewer_centralities')
    plot_centralities(restaurant_centralities, 'restaurants', True, 'restaurant_centralities')

    # # plot number of common reviewers per restaurant
    # restaurant_nodes = set(n for n,d in B.nodes(data=True) if d['bipartite']==1)
    # common_neighbors_plot(B, restaurant_nodes, 'restaurants', 'reviewers', True, 'common_neighbors')


if __name__ == '__main__':
    main()
