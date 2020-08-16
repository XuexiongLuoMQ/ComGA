import networkx as nx
import igraph as ig
from networkx.algorithms import community
import numpy as np
from scipy.io import loadmat


class Graph:
    def __init__(self):
        self.network = None
        self.labels = None
        self.clusters = None
        self.original_vec = None
        self.clusters_num = -1
        self.node_num = -1
    def load_graph_from_weighted_edgelist(self, file_name):
        self.network = nx.read_weighted_edgelist(file_name)
        self.node_num = self.network.number_of_nodes()

    def load_graph_from_edgelist(self, file_name):
        self.network = nx.Graph()
        with open(file_name, 'r') as file:
            for line in file:
                node1, node2 = line.strip().split()
                self.network.add_edge(int(node1), int(node2))
        self.node_num = self.network.number_of_nodes()

    def load_label_from_mat(self, file_name, variable_name):
        data = loadmat(file_name)
        value = data[variable_name]
        if isinstance(value, np.ndarray):
            self.labels = {idx: label for (idx, label) in enumerate(value.flatten().tolist())}
        else:
            nonzero_list = value.nonzero()
            label_dict = dict(zip(nonzero_list[0], nonzero_list[1]))
            self.labels = label_dict

    def load_graph_from_mat(self, file_name, variable_name):
        data = loadmat(file_name)
        mat_matrix = data[variable_name]
        if isinstance(mat_matrix, np.ndarray):
            self.network = nx.from_numpy_matrix(mat_matrix)
            if nx.is_directed(self.network):
                self.network.to_undirected()
        else:
            self.network = nx.from_scipy_sparse_matrix(mat_matrix)
            if nx.is_directed(self.network):
                self.network.to_undirected()
        for node in self.network.nodes_with_selfloops():
            self.network.remove_edge(node, node)
        self.node_num = self.network.number_of_nodes()

    def lpa(self):
        communities_generator = community.label_propagation_communities(self.network)
        print(communities_generator,'ZZZZZZZZZZZZZZZ')
        node_cluster_pair = [(node, cluster_id) for (cluster_id, node_set) in enumerate(communities_generator)
                             for node in list(node_set)]
        self.clusters = dict(node_cluster_pair)
        self.clusters_num = max(self.clusters.values())+1
        print('{} clusters'.format(self.clusters_num))

    def infomap(self):
        g = ig.Graph(list(self.network.edges()))
        infomap_cluster = g.community_infomap()
        self.clusters = {node: cluster_id for cluster_id, nodes in enumerate(infomap_cluster) for node in nodes}
        self.clusters_num = max(self.clusters.values())+1
        print('{} clusters'.format(self.clusters_num))

    def output_data(self):
        A = np.asarray(nx.adjacency_matrix(self.network, nodelist=None, weight='None').todense())
        x = A
        y = np.array([self.clusters[node] for node in sorted(self.clusters.keys())])
        return x, y

    def read_graph(self,edgeFile):
        print('loading graph...')

        G = nx.read_edgelist(edgeFile, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        # if not FLAGS.directed:
        #     G = G.to_undirected()

        return G