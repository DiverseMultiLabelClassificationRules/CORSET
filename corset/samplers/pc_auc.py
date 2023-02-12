"""
an DFS-based algorithm to enumerate alpha-probably cliques in an uncertain graph
"""
import networkx as nx
import numpy as np
import functools
from .base import ReducedSpaceSampler
from scipy import sparse as sp
from ..graph import construct_confidence_graph

from logzero import logger

def enumerate_probable_cliques(g, alpha, proba_key='proba', return_proba=True):
    """
    g: the connectivity graph of an uncertain graph
    proba: clique probability of S
    """
    pc_list = []
    dfs(tuple(), g, 1.0, alpha, pc_list, proba_key, return_proba)
    return pc_list[1:]  # remove empty tuple

def dfs(S: tuple, g: nx.Graph, proba: float, alpha: float, pc_list: [], proba_key:str, return_proba: bool):
    """
    S: current clique
    g: underlying connectivity graph of the uncertain graph
    proba: clique probability of S
    alpha: probability threshold
    pc_list: list of alpha-probabile cliques
    proba_key: key name to access edge probability
    return_proba: whether clique probability is returned together with the clique
    """
    if return_proba:
        pc_list.append((S, proba))
    else:
        pc_list.append(S)

    if len(S) == 0:
        # 1-clique has probability 1.0 by default
        for u in g.nodes():
            dfs((u, ), g, 1.0, alpha, pc_list, proba_key, return_proba)
    else:
        nbr_sets = (set(g.neighbors(v)) for v in S)
        shared_nbrs = functools.reduce(lambda a, b: a.intersection(b),
                                       nbr_sets)
        
        for u in shared_nbrs:
            multiplier = np.prod([g[u][v][proba_key] for v in S])
            if u > S[-1]:
                new_proba = proba * multiplier
                if new_proba >= alpha:
                    dfs(S + (u, ), g, new_proba, alpha, pc_list, proba_key, return_proba)


def convert_to_connectivity_graph(g: nx.DiGraph, proba_key='proba'):
    """
    convert an uncertain graph (directed) to a connectivty graph (undirected)

    g: an directed uncertain graph
       requirementment: edges are bidirectional, if (u, v) exists, then (v, u) exists as well
    """
    conn_g = nx.Graph()
    conn_g.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if u < v:
            p = 1 - (1-g[u][v][proba_key]) * (1-g[v][u][proba_key])
            conn_g.add_edge(u, v, proba=p)
    return conn_g

class FrequencySamplerMixin:
    """sample according to frequency, assuming the sample space is reduced

    the sample space is obtained via `self._sample_space`
    
    the valid itemsets contained by each data record is stored in `self.new_rows`, 
    where each row is a list of itemset ids
    """
    def _assign_row_weights_uncovered_area(self):

        #self.row_weights = np.array(list(map(len, self.new_rows)))
        # new rows contain the sample ids for a given rows . We need to take the corresponding sample and 
        # add to the weight the size of the uncovered labels in that sample 
        self.row_weights = np.zeros(len(self.new_rows)) 
        for i in range(len(self.new_rows)): 
            this_weight = 0 
            sample_ids_this_row = self.new_rows[i] 
            for sample_id in sample_ids_this_row:
                sample = self._sample_space[sample_id] 
                this_weight += len([lab for lab in sample if lab not in self.covered[i]])
                
            self.row_weights[i] = this_weight    
        

    def _sample_itemset_from_row(self, row_id):
        # for the sampled row, sample a clique
        # from its associated clique samples uniformly randomly
        clique_ids = self.new_rows[row_id]
        sampled_clique_id = np.random.choice(clique_ids)
        return self._sample_space[sampled_clique_id]    



class PCFrequencySampler(FrequencySamplerMixin, ReducedSpaceSampler):
    """sample space = alpha-probable cliques, which are node sets with clique probability >= some threshold"""

    def __init__(self,
    		covered
                 random_state=12345,
                 min_proba=0.05):
        kwargs = locals()
        del kwargs['self']

        self.__dict__.update(kwargs)

        self._g = None
        
        self.covered = copy.deepcopy(covered) 
        
        

    def _build_sample_space(self):
        np.random.seed(self.random_state)

        self._sample_space = list(
            map(
                set,
                enumerate_probable_cliques(self._g,
                                           alpha=self.min_proba,
                                           return_proba=False
                                      )))
        logger.debug('sample space size: {}'.format(len(self._sample_space)))
        # print('size of reduced sample space: {}'.format(len(self._sample_space)))
        # print(self._sample_space)

    def _build_graph(self, Y: sp.csr_matrix):
        ug = construct_confidence_graph(Y.tocsc())
        self._g = convert_to_connectivity_graph(ug, proba_key='proba')

    def fit(self, Y: sp.csr_matrix):
        self._build_graph(Y)
        super().fit(Y, target_distribution="uncovered area")
        
        
    
    
