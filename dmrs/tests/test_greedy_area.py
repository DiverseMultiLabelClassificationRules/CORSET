import pytest

from dmrs.samplers.greedy_area import GreedyAreaSampler

from .fixtures import toy_dataset as ds

def test_runnable(ds):
    X, Y = ds.trn_X.toarray(), ds.trn_Y.toarray()
    sampler = GreedyAreaSampler(random_state=123)
    assert hasattr(sampler, 'random_state')
    sampler.fit(X, Y)
    itemset = sampler.sample_once(12345)
    assert isinstance(itemset, set)
    for i in itemset:
        assert 0 <= i < ds.ncls
