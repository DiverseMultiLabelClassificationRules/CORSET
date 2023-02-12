import networkx as nx
import numpy as np

from scipy import sparse as sp
from .utils import flatten

from itertools import permutations

# @profile
def construct_confidence_graph(Y: sp.csc_matrix, label_names=None):
    """
    there is an edge from $u$ to $v$ if $u$ and $v$ co-occur in at least one training example
    - the weight (called "proba") from $u$ to $v$ is defined as
      - $c(u, v) = \frac{|X(u) \cap X(v)|}{|X(u)|}$, where $X$ is the support function
      - it can be interpreted as `Pr[v appears | u appears]`
    """
    label_freq = flatten(Y.sum(axis=0))

    nlabels = Y.shape[1]
    label2rows = {}
    for l in range(nlabels):
        label2rows[l] = set(Y[:, l].indices)

    cor_M = Y.T.tocsr().astype(float) @ Y.astype(float)

    rows, cols = cor_M.nonzero()

    g = nx.DiGraph()
    g.add_nodes_from(range(nlabels))  # make sure node ids are zero-indexed

    for u, v in zip(rows, cols):
        numer = len(label2rows[u].intersection(label2rows[v]))
        denum = len(label2rows[u])
        g.add_edge(u, v, proba=(numer / denum))

    g.add_nodes_from(np.arange(Y.shape[1]))
    for i in range(Y.shape[1]):
        g.nodes[i]['freq'] = label_freq[i]
        if label_names is not None:
            g.nodes[i]['name'] = label_names[i]
        else:
            g.nodes[i]['name'] = i

    # remove self-loops
    g.remove_edges_from(nx.selfloop_edges(g))
    return g

# @profile
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

def show_subgraph(g, nodes):
    sg = g.subgraph(nodes)
    for u, v, w in sg.edges(data='proba'):
        print('{} ({}) -> {} ({}): {:.2f}'.format(g.nodes[u]['name'],
                                                  g.nodes[u]['freq'],
                                                  g.nodes[v]['name'],
                                                  g.nodes[v]['freq'], w))


def forms_clique(g: nx.DiGraph, S: set, v):
    # returns True if v is connected to every node in S
    for s in S:
        if not g.has_edge(s, v):
            return False
    return True


def sample_clique(g: nx.DiGraph,
                  seed: int = None,
                  start: int = None,
                  verbose: bool = False):
    """
     a simple heuristic which samples a clique from a graph `g`

    it does the following:

    1. randomly select a node u to start; put u to an empty set S
    2. for each out edges (excluding self-loops), put it in O with probability equal to its weight. 
    2. If O is empty, exit, otherwise, for pick e=(u, v) among O with probability proportional to its weight
    3. check if v is connected with every node in S, if not, exit the loop, otherwise add v to S, set u=v, and repeat 2 - 3
    4. return S
    """
    np.random.seed(seed)
    if start is None:
        u = np.random.choice(g.nodes())
    else:
        u = start
    S = {u}
    while True:
        if verbose:
            print(f'at {u}'.format(u))
        out_nbrs = np.array(list(g[u].keys()))
        if len(out_nbrs) == 0:
            break
        probas = np.array([g[u][v]['proba'] for v in out_nbrs])
        mask = (np.random.rand(len(probas)) < probas)

        nz = mask.nonzero()[0]
        candidates = out_nbrs[nz]
        cand_probas = probas[nz]
        if verbose:
            print(f'candidates: {candidates}')
            print(f'cand_probas: {cand_probas}')

        if len(candidates) == 0:
            break

        v = np.random.choice(candidates, p=cand_probas / cand_probas.sum())
        if forms_clique(g, S, v):
            if verbose:
                print(f'add {v}')
            S.add(v)
        else:
            break
        u = v
    return tuple(sorted(S))


def subgraph_stat(g, nodes):
    ret = {
        'size': len(nodes),
    }

    ret['density'] = np.sum(
        [g[u][v]['proba']
         for u, v in permutations(nodes, 2)]) / len(nodes)

    ret['is_clique'] = (g.subgraph(nodes).number_of_edges() == (
        len(nodes) * (len(nodes) - 1)))
    # ret['label_freq'] = [g.nodes[n]['freq'] for n in nodes]
    return ret
