'''
Created on 22 Jan 2017

@author: af
'''
import networkx as nx
import numpy as np
import pdb
import gzip
import csv
import pandas as pd
import os
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict as dd, OrderedDict
from haversine import haversine
import kdtree
import sys
from sklearn.neighbors import NearestNeighbors
#from networkx.algorithms.bipartite.projection import weighted_projected_graph
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def projected_graph(B, nodes, multigraph=False):
    if B.is_multigraph():
        raise nx.NetworkXError("not defined for multigraphs")
    if B.is_directed():
        directed=True
        if multigraph:
            G=nx.MultiDiGraph()
        else:
            G=nx.DiGraph()
    else:
        directed=False
        if multigraph:
            G=nx.MultiGraph()
        else:
            G=nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n,B.node[n]) for n in nodes)
    i = 0
    nodes = set(nodes)
    tenpercent = len(nodes) / 10
    for u in nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  
        nbrs2=set((v for nbr in B[u] for v in B[nbr])) & nodes - set([u])
        if multigraph:
            for n in nbrs2:
                if directed:
                    links=set(B[u]) & set(B.pred[n])
                else:
                    links=set(B[u]) & set(B[n])
                for l in links:
                    if not G.has_edge(u,n,l):
                        G.add_edge(u,n,key=l)
        else:
            G.add_edges_from((u,n) for n in nbrs2)
    return G

def efficient_projected_graph(B, nodes):
    g = nx.Graph()
    nodes = set(nodes)
    g.add_nodes_from(nodes)
    b_nodes = set(B.nodes())
    i = 0
    nodes = set(nodes)
    tenpercent = len(b_nodes) / 10
    for n in b_nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  
        nbrs = list(set([nbr for nbr in B[n]]) & nodes - set([n]))
        if n in nodes:
            for nbr in nbrs:
                if not g.has_edge(n, nbr):
                    g.add_edge(n, nbr)
        for nbr1 in nbrs:
            for nbr2 in nbrs:
                if nbr1 < nbr2:
                    if not g.has_edge(nbr1, nbr2):
                        g.add_edge(nbr1, nbr2)
        del nbrs
            
    return g        
    

def collaboration_weighted_projected_graph(B, nodes):
    if B.is_multigraph():
        raise nx.NetworkXError("not defined for multigraphs")
    if B.is_directed():
        pred=B.pred
        G=nx.DiGraph()
    else:
        pred=B.adj
        G=nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n,B.node[n]) for n in nodes)
    i = 0
    nodes = set(nodes)
    tenpercent = len(nodes) / 10
    for u in nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  
        unbrs = set(B[u])
        nbrs2 = set((n for nbr in unbrs for n in B[nbr])) & nodes - set([u])
        for v in nbrs2:
            vnbrs = set(pred[v])
            common = unbrs & vnbrs
            weight = sum([1.0/(len(B[n]) - 1) for n in common if len(B[n])>1])
            G.add_edge(u,v,w=weight)
    return G

def efficient_collaboration_weighted_projected_graph(B, nodes):
    r"""Newman's weighted projection of B onto one of its node sets.

    The collaboration weighted projection is the projection of the
    bipartite network B onto the specified nodes with weights assigned
    using Newman's collaboration model [1]_:

    .. math::
        
        w_{v,u} = \sum_k \frac{\delta_{v}^{w} \delta_{w}^{k}}{k_w - 1}

    where `v` and `u` are nodes from the same bipartite node set,
    and `w` is a node of the opposite node set. 
    The value `k_w` is the degree of node `w` in the bipartite
    network and `\delta_{v}^{w}` is 1 if node `v` is
    linked to node `w` in the original bipartite graph or 0 otherwise.
 
    The nodes retain their attributes and are connected in the resulting
    graph if have an edge to a common node in the original bipartite
    graph.

    Parameters
    ----------
    B : NetworkX graph 
      The input graph should be bipartite. 

    nodes : list or iterable
      Nodes to project onto (the "bottom" nodes).

    Returns
    -------
    Graph : NetworkX graph 
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(5)
    >>> B.add_edge(1,5)
    >>> G = bipartite.collaboration_weighted_projected_graph(B, [0, 2, 4, 5])
    >>> print(G.nodes())
    [0, 2, 4, 5]
    >>> for edge in G.edges(data=True): print(edge)
    ... 
    (0, 2, {'weight': 0.5})
    (0, 5, {'weight': 0.5})
    (2, 4, {'weight': 1.0})
    (2, 5, {'weight': 0.5})
    
    Notes
    ------
    No attempt is made to verify that the input graph B is bipartite.
    The graph and node properties are (shallow) copied to the projected graph.

    See Also
    --------
    is_bipartite, 
    is_bipartite_node_set, 
    sets, 
    weighted_projected_graph,
    overlap_weighted_projected_graph,
    generic_weighted_projected_graph,
    projected_graph 

    References
    ----------
    .. [1] Scientific collaboration networks: II. 
        Shortest paths, weighted networks, and centrality, 
        M. E. J. Newman, Phys. Rev. E 64, 016132 (2001).
    """
    nodes = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    all_nodes = set(B.nodes())
    i = 0
    tenpercent = len(all_nodes) / 10
    for m in all_nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  

        nbrs = B[m]
        target_nbrs = [t for t in nbrs if t in nodes]
        #if len(nbrs) < 2:
        #    continue
        if m in nodes:
            for n in target_nbrs:
                if m < n:
                    n_nbrs = len(B[n])
                    if n_nbrs > 1:
                        w_n = 1.0 / (n_nbrs - 1)
                    else:
                        w_n = 0
                    w = 1.0 / (len(nbrs) - 1) + w_n
                    if G.has_edge(m, n):
                        G[m][n]['w'] += w
                    else:
                        G.add_edge(m, n, w=w)
        
        for n1 in target_nbrs:
            for n2 in target_nbrs:
                if n1 < n2:
                    w = 1.0 / (len(nbrs) - 1)
                    if G.has_edge(n1, n2):
                        G[n1][n2]['w'] += w
                    else:
                        G.add_edge(n1, n2, w=w)
        
    return G

def efficient_collaboration_weighted_projected_graph2(B, nodes):
    nodes = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    all_nodes = set(B.nodes())
    i = 0
    tenpercent = len(all_nodes) / 10
    for m in all_nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1  

        nbrs = B[m]
        target_nbrs = [t for t in nbrs if t in nodes]
        if m in nodes:
            for n in target_nbrs:
                if m < n:
                    if not G.has_edge(m, n):
                        G.add_edge(m, n)
        for n1 in target_nbrs:
            for n2 in target_nbrs:
                if n1 < n2:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
    logging.info('the graph is written in na.txt.gz')
    return G

class DataLoader():
    def __init__(self, data_home, bucket_size=50, encoding='utf-8', celebrity_threshold=10, one_hot_labels=False, mindf=10):
        self.data_home = data_home
        self.bucket_size = bucket_size
        self.encoding = encoding
        self.celebrity_threshold = celebrity_threshold
        self.one_host_labels = one_hot_labels
        self.mindf = mindf
    def load_data(self):
        logging.info('loading the dataset from %s' %self.data_home)
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')
        
        df_train = pd.read_csv(train_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_test = pd.read_csv(test_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_train.dropna(inplace=True)
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)
        df_train['user'] = df_train['user'].apply(lambda x: str(x).lower())
        df_train.drop_duplicates(['user'], inplace=True, keep='last')
        df_train.set_index(['user'], drop=True, append=False, inplace=True)
        df_train.sort_index(inplace=True)
        df_dev['user'] = df_dev['user'].apply(lambda x: str(x).lower())
        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)
        df_test['user'] = df_test['user'].apply(lambda x: str(x).lower())
        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)
        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test
        

    def get_graph(self):
        g = nx.Graph()
        nodes = set(self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + len(self.df_test), 'duplicate target node'
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist()
        node_id = {node:id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        logging.info('adding the train graph')
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_train.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        logging.info('adding the dev graph')
        for i in range(len(self.df_dev)):
            user = self.df_dev.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_dev.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)        
        logging.info('adding the test graph')
        for i in range(len(self.df_test)):
            user = self.df_test.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_test.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)    
        celebrities = []
        for i in xrange(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)
        logging.info('removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)
            
        logging.info('projecting the graph')
        g = efficient_collaboration_weighted_projected_graph2(g, range(len(nodes_list)))
        logging.info('#nodes: %d, #edges: %d' %(nx.number_of_nodes(g), nx.number_of_edges(g)))
        self.graph = g

        
    def tfidf(self):
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)[^@]\b\w\w+\b', use_idf=True, norm='l2', binary=True
                                     , sublinear_tf=False, min_df=self.mindf, max_df=0.2, ngram_range=(1, 1), stop_words='english', 
                                     vocabulary=None, encoding=self.encoding, dtype='float32')
        
        self.X_train = vectorizer.fit_transform(self.df_train.text.values)
        self.X_dev = vectorizer.transform(self.df_dev.text.values)
        self.X_test = vectorizer.transform(self.df_test.text.values)
        logging.info("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        logging.info("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        logging.info("test        n_samples: %d, n_features: %d" % self.X_test.shape)
    
    def assignClasses(self):
        clusterer = kdtree.KDTreeClustering(bucket_size=self.bucket_size)
        train_locs = self.df_train[['lat', 'lon']].values
        clusterer.fit(train_locs)
        clusters = clusterer.get_clusters()
        cluster_points = dd(list)
        for i, cluster in enumerate(clusters):
            cluster_points[cluster].append(train_locs[i])
        logging.info('#labels: %d' %len(cluster_points))
        self.cluster_median = OrderedDict()
        for cluster in sorted(cluster_points):
            points = cluster_points[cluster]
            median_lat = np.median([p[0] for p in points])
            median_lon = np.median([p[1] for p in points]) 
            self.cluster_median[cluster] = (median_lat, median_lon)
        dev_locs = self.df_dev[['lat', 'lon']].values
        test_locs = self.df_dev[['lat', 'lon']].values
        nnbr = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size=1, metric=haversine, n_jobs=4)
        nnbr.fit(np.array(self.cluster_median.values()))
        self.dev_classes = nnbr.kneighbors(dev_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.test_classes = nnbr.kneighbors(test_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.train_classes = clusters
        if self.one_host_labels:
            num_labels = np.max(self.train_classes) + 1
            y_train = np.zeros((len(self.train_classes), num_labels), dtype=np.float32)
            y_train[np.arange(len(self.train_classes)), self.train_classes] = 1
            y_dev = np.zeros((len(self.dev_classes), num_labels), dtype=np.float32)
            y_dev[np.arange(len(self.dev_classes)), self.dev_classes] = 1
            y_test = np.zeros((len(self.test_classes), num_labels), dtype=np.float32)
            y_test[np.arange(len(self.test_classes)), self.test_classes] = 1
            self.train_classes = y_train
            self.dev_classes = y_dev
            self.test_classes = y_test


if __name__ == '__main__':
    data_loader = DataLoader(data_home='./data/', dataset='cmu')
    data_loader.load_data()
    data_loader.get_graph()
    data_loader.tfidf()
    data_loader.assignClasses()
    pdb.set_trace()
    
