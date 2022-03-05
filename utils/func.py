import networkx as nx
import numpy as np
import pandas as pd


def gen_M(file_path):
    raw_data = pd.read_csv(file_path)
    origins = raw_data['From']
    destinations = raw_data['To']
    if origins.min() == 0 or destinations.min() == 0:
        origins += 1
        destinations += 1
    n_node = max(origins.max(), destinations.max())
    n_link = raw_data.shape[0]

    M = np.zeros((n_node, n_link))
    for i in range(n_link):
        M[origins[i] - 1, i] = 1
        M[destinations[i] - 1, i] = -1
    mu = np.array(raw_data['Cost']).reshape(-1, 1)
    return M, mu


def gen_M2nxG(M, weight, is_multi=True):
    G = nx.MultiDiGraph() if is_multi else nx.DiGraph()
    for i in range(M.shape[1]):
        start = np.where(M[:, i] == 1)[0].item()
        end = np.where(M[:, i] == -1)[0].item()
        G.add_edge(start, end, weight=weight[i].item(), index=1)
    return G


class MapInfo:
    def __init__(self, file_path="../maps/SiouxFalls/sioux_network.csv", is_multi=True):
        self.M, self.mu = gen_M(file_path)
        self.G = gen_M2nxG(self.M, self.mu, is_multi)
        self.n_node = self.M.shape[0]
        self.n_edge = self.M.shape[1]

    def get_next_nodes(self, node):
        return [node] + list(map(lambda x: x[1]+1, self.G.edges(node-1)))

    def get_edges(self, node):
        return [[node, node]] + list(map(lambda x: [x[0]+1, x[1]+1], self.G.edges(node-1)))

    def get_edge_weight(self, edge):
        return self.G.get_edge_data(edge[0]-1, edge[1]-1)[0]["weight"]

    def get_edge_index(self, edge):
        return self.G.get_edge_data(edge[0]-1, edge[1]-1)[0]["index"]
