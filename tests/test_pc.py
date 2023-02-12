import networkx as nx
import numpy as np

import pytest

from itertools import combinations
from corset.samplers.pc import enumerate_probable_cliques, convert_to_connectivity_graph, PCFrequencySampler, prune_edges
from .fixtures import toy_dataset as ds

@pytest.mark.parametrize('backend', ['dfs', 'dfs_v2'])
def test_enumerate_probable_cliques(backend):
    g = nx.Graph()
    g.add_nodes_from(range(4))
    for u, v in combinations(range(4), 2):
        g.add_edge(u, v, proba=0.5)

    print(enumerate_probable_cliques(g, 0.9, return_proba=True, backend=backend))
    actual_1 = set(enumerate_probable_cliques(g, 0.9, return_proba=True, backend=backend))
    expected = {
        ((0,), 1.0),
        ((1,), 1.0),
        ((2,), 1.0),
        ((3,), 1.0)
    }
    assert actual_1 == expected

    actual_2 = set(enumerate_probable_cliques(g, 0.5, return_proba=True, backend=backend))
    expected = {
        ((0,), 1.0),
        ((0, 1), 0.5),
        ((0, 2), 0.5),
        ((0, 3), 0.5),
        ((1,), 1.0),
        ((1, 2), 0.5),
        ((1, 3), 0.5),
        ((2,), 1.0),
        ((2, 3), 0.5),
        ((3,), 1.0)
    }
    assert actual_1.issubset(actual_2)
    assert actual_2 == expected

    actual_3 = set(enumerate_probable_cliques(g, 0.125, return_proba=True, backend=backend))
    expected = {
        ((0,), 1.0),
        ((0, 1), 0.5),
        ((0, 1, 2), 0.125),
        ((0, 1, 3), 0.125),
        ((0, 2), 0.5),
        ((0, 2, 3), 0.125),
        ((0, 3), 0.5),
        ((1,), 1.0),
        ((1, 2), 0.5),
        ((1, 2, 3), 0.125),
        ((1, 3), 0.5),
        ((2,), 1.0),
        ((2, 3), 0.5),
        ((3,), 1.0)
    }
    assert actual_2.issubset(actual_3)
    assert actual_3 == expected

    actual_4 = set(enumerate_probable_cliques(g, 0.5**6, return_proba=True, backend=backend))
    assert actual_3.issubset(actual_4)
    assert len(actual_4) == (len(actual_3) + 1)


def test_convert_to_connectivity_graph():
    g = nx.DiGraph()
    g.add_nodes_from(range(3))
    g.add_edge(0, 1, proba=0.5)
    g.add_edge(1, 0, proba=0.7)
    g.add_edge(1, 2, proba=0.3)
    g.add_edge(2, 1, proba=0.4)

    conn_g = convert_to_connectivity_graph(g)
    np.testing.assert_allclose(conn_g[0][1]['proba'], 1 - (1 - 0.5) * (1 - 0.7))
    np.testing.assert_allclose(conn_g[1][2]['proba'], 1 - (1 - 0.3) * (1 - 0.4))


def test_prune_edges():
    g = nx.Graph()
    g.add_nodes_from(range(3))
    g.add_edge(0, 1, proba=0.1)
    g.add_edge(1, 2, proba=0.2)
    g.add_edge(0, 2, proba=0.3)

    gp = prune_edges(g, 0.1001, proba_key='proba')
    assert set(gp.edges()) == {(1, 2), (0, 2)}

    gp = prune_edges(g, 0.2001, proba_key='proba')
    assert set(gp.edges()) == {(0, 2)}

    gp = prune_edges(g, 0.3001, proba_key='proba')
    assert set(gp.edges()) == set()

@pytest.mark.parametrize('do_prune_edges', [True, False])    
def test_sampler_runnable(ds, do_prune_edges):
    sampler = PCFrequencySampler(random_state=12345, min_proba=0.0, do_prune_edges=do_prune_edges)

    sampler.fit(ds.trn_Y)
    # assert len(sampler._sample_space) <= num_samples
    assert set(sampler.new_rows[0]) == set(range(len(sampler._sample_space)))

    itemset = sampler.sample_once(12345)
    assert isinstance(itemset, set)


def test_pruning_equivalence(ds):
    sampler_no_pruning = PCFrequencySampler(random_state=12345, min_proba=0.5, do_prune_edges=False)
    sampler_no_pruning.fit(ds.trn_Y)

    sampler_with_pruning = PCFrequencySampler(random_state=12345, min_proba=0.5, do_prune_edges=True)
    sampler_with_pruning.fit(ds.trn_Y)

    assert sampler_no_pruning._sample_space == sampler_with_pruning._sample_space    
