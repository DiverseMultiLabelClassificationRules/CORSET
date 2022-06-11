import numpy as np
from dmrs.samplers.dummy import DummyHeadSampler, DummyTailSampler
from .fixtures import random_dataset

def test_head_sampler_runnable(random_dataset):
    X = random_dataset.trn_X
    y = (np.random.rand(X.shape[0]) > 0.5)
    sampler = DummyHeadSampler()
    sampler.fit(X, y)

    itemset = sampler.sample_once()
    assert isinstance(itemset, set)
    assert len(itemset) == 1
    assert 0 <= list(itemset)[0] < X.shape[1]


def test_tail_sampler_runnable(random_dataset):
    Y = random_dataset.trn_Y
    sampler = DummyTailSampler()
    sampler.fit(Y)

    itemset = sampler.sample_once()
    assert isinstance(itemset, set)
    assert len(itemset) == 1
    assert 0 <= list(itemset)[0] < Y.shape[1]
    
