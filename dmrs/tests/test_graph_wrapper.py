import numpy as np
import networkx as nx
from dmrs.samplers.dfs import Graph

def test_graph():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    key = 'proba'
    g.add_edges_from([
        (0, 1, {key: 1}),
        (1, 2, {key: 0.5}),
    ])

    mat = nx.to_scipy_sparse_matrix(g, dtype=float, weight=key)
    g_wrapped = Graph(mat.shape[0], mat.data, mat.indices, mat.indptr)    

    assert g_wrapped.neighbors(1) == {0, 2}
    assert g_wrapped.neighbors(0) == {1}

    assert g_wrapped.edge_weight(0, 1) == 1
    assert g_wrapped.edge_weight(1, 2) == 0.5
