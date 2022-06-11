# an DFS-based algorithm to enumerate alpha-probably cliques in an uncertain graph

import networkx as nx
import numpy as np
import functools
from scipy import sparse as sp
from time import time
from .base import Sampler
from .surs import ReducedSpaceSampler
from .assignment import TrieSampleAssignmentMixin
from .freq import SURSFrequencySamplerMixin

from ..graph import construct_confidence_graph, convert_to_connectivity_graph
from ..utils import array_product

from logzero import logger

def enumerate_probable_cliques(g, alpha, proba_key='proba', return_proba=True, backend='dfs'):
    """
    g: the connectivity graph of an uncertain graph
    proba: clique probability of S
    """
    if backend == 'dfs':
        pc_list = []
        # adj_mat = nx.to_scipy_sparse_matrix(g, weight=proba_key, dtype=float, format='csr')
        shared_nbrs = set(g.nodes())  # initially all nodes are considered
        dfs(tuple(), g, 1.0, shared_nbrs, alpha, pc_list, proba_key, return_proba)
        return pc_list[1:]  # remove empty tuple
    elif backend == 'dfs_v2':
        from .dfs import dfs_v2
        pc_list = []
        # adj_mat = nx.to_scipy_sparse_matrix(g, weight=proba_key, dtype=float, format='csr')
        dfs_v2(g, alpha, pc_list, proba_key, return_proba)
        # pc_list = list(map(tuple, pc_list))  # map each list to a tuple
        return pc_list[1:]  # remove empty tuple
    else:
        raise ValueError('unsupported backend "{}"'.format(backend))


# @profile
def dfs(S: tuple, g: nx.Graph, proba: float, shared_nbrs: set, alpha: float, pc_list: [], proba_key:str, return_proba: bool):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    USE .dfs.dfs_v2 FOR BETTER PERFORMANCE!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
            dfs((u, ), g, 1.0, set(g.neighbors(u)),
                alpha, pc_list, proba_key, return_proba)
    else:
        for u in shared_nbrs:
            if u > S[-1]:
                # incrementally determine if go down the recursion
                success = True
                new_proba = proba
                for v in S:
                    new_proba *= g._adj[u][v][proba_key]
                    if new_proba < alpha:
                        success = False
                        break                    
                if success:
                    new_shared_nbrs = shared_nbrs.intersection(set(g.neighbors(u)))  # incremental update
                    dfs(S + (u, ), g, new_proba, new_shared_nbrs, alpha, pc_list, proba_key, return_proba)

                    

def prune_edges(g: nx.Graph, min_proba: float, proba_key='proba'):
    """
    make a copy of the input graph with the difference that 
    edges, whose weight (specified by `proba` ) are under min_proba, are removed
    """
    gp = nx.Graph()
    gp.add_nodes_from(g.nodes())

    for u, v in g.edges():
        if g[u][v][proba_key] >= min_proba:
            gp.add_edge(u, v)
            gp[u][v][proba_key] = g[u][v][proba_key]
    return gp


class PCSampleSpaceConstructor:
    """
    a class providing methods to construct sample space using alpha-probable clique enumeration algorithm
    """
    def _build_sample_space(self):
        np.random.seed(self.random_state)

        if self.do_prune_edges:
            g = prune_edges(self._g, self.min_proba, proba_key='proba')
            logger.debug('prune edges from the graph: done')
            logger.debug("number of nodes: {}".format(g.number_of_nodes()))
            logger.debug("number of edges change: {} -> {}".format(
                self._g.number_of_edges(),
                g.number_of_edges()))
        else:
            g = self._g

        s = time()
        sample_space = enumerate_probable_cliques(g,
                                                  alpha=self.min_proba,
                                                  return_proba=False,
                                                  backend=self.dfs_backend
                                                  )
        logger.debug('running dfs takes: {}'.format(time() - s))
        self._sample_space = list(map(set, sample_space))
        logger.debug('sample space size: {}'.format(len(self._sample_space)))
        # print('size of reduced sample space: {}'.format(len(self._sample_space)))
        # print(self._sample_space)

    # @profile
    def _build_graph(self, Y: sp.csr_matrix):
        ug = construct_confidence_graph(Y.tocsc())
        logger.debug('confidence graph construction: done')
        self._g = convert_to_connectivity_graph(ug, proba_key='proba')
        logger.debug('converting to connectivity graph: done')

class PCFrequencySampler(TrieSampleAssignmentMixin, PCSampleSpaceConstructor, SURSFrequencySamplerMixin, ReducedSpaceSampler):
    """sample space = alpha-probable cliques, which are node sets with clique probability >= some threshold"""

    def __init__(self,
                 random_state=12345,
                 min_proba=0.05,
                 do_prune_edges=True,
                 dfs_backend='dfs_v2'):
        kwargs = locals()
        del kwargs['self']

        self.__dict__.update(kwargs)

        self._g = None

    def fit(self, Y: sp.csr_matrix):
        self._build_graph(Y)

        super().fit(Y)
