import pytest
import numpy as np
from dmrs.samplers.uncovered_area import UncoveredAreaMixin, UncoveredAreaSampler
from dmrs.utils import counter2proba
from dmrs.exceptions import NoMoreSamples
from .fixtures import random_dataset
from collections import Counter


class TestUncoveredAreaMixin:
    @pytest.fixture
    def sampler(self):
        sampler = UncoveredAreaMixin()
        sampler.new_rows = [[0, 1, 2], [0]]
        sampler._sample_space = [{0}, {1}, {0, 1}]
        return sampler

    def test__assign_row_weights(self, sampler):
        sampler._assign_row_weights()
        np.testing.assert_allclose(sampler.row_weights, [4, 1])
        np.testing.assert_allclose(sampler.row_probas, [0.8, 0.2])

    def test__build_element_counter(self, sampler):
        sampler._build_element_counter()
        assert dict(sampler.element_counter_per_row[0]) == {0: 2, 1: 2}
        assert dict(sampler.element_counter_per_row[1]) == {0: 1}

    @pytest.mark.parametrize('indices, new_set, new_weights', [
        ([0], {0, 1}, [0, 1]),
        ([0], {1}, [2, 1]),
        ([0, 1], {0, 1}, [0, 0])
    ])
    def test_update_row_weights(self, sampler, indices, new_set, new_weights):
        sampler._assign_row_weights()

        if sum(new_weights) == 0:
            new_probas = np.ones(len(new_weights)) / len(new_weights)
        else:
            new_probas = np.array(new_weights) / sum(new_weights)
        
        for i in range(10):  # we update 10 times, the results should be the same as updating once
            sampler.update_row_weights(indices, new_set)
        np.testing.assert_allclose(sampler.row_weights, new_weights)
        np.testing.assert_allclose(sampler.row_probas, new_probas)

    def test__sample_itemset_from_row(self, sampler):
        sampler._assign_row_weights()
        n_reps = 2000
        # sample from record 0
        samples = [tuple(sampler._sample_itemset_from_row(0))
                   for _ in range(n_reps)]
        proba = counter2proba(Counter(samples))

        assert len(proba) == 3
        np.testing.assert_allclose(proba[(0, )],    .25, rtol=1e-1)
        np.testing.assert_allclose(proba[(1, )],    .25, rtol=1e-1)
        np.testing.assert_allclose(proba[(0, 1, )], .50, rtol=1e-1)

        # sample from record 1
        samples = [tuple(sampler._sample_itemset_from_row(1))
                   for _ in range(n_reps)]
        proba = counter2proba(Counter(samples))

        assert len(proba) == 1
        np.testing.assert_allclose(proba[(0, )], 1.0, rtol=1e-2)

        # now we update weights 100 times for the same update
        # the effect should be the same as just update once
        for i in range(100):
            sampler.update_row_weights([0, 1], {0})
        
        samples = [tuple(sampler._sample_itemset_from_row(0))
                   for _ in range(n_reps)]
        proba = counter2proba(Counter(samples))
        assert len(proba) == 2
        np.testing.assert_allclose(proba[(0, 1, )], .5, rtol=1e-1)
        np.testing.assert_allclose(proba[(1, )], .5, rtol=1e-1)

        with pytest.raises(NoMoreSamples):
            sampler._sample_itemset_from_row(1)  # all samples have zero weight, no way to sample


class TestUncoveredAreaSampler:
    def test_runnable(self, random_dataset):
        Y = random_dataset.trn_Y
        sampler = UncoveredAreaSampler()
        sampler.fit(Y)

        sampler.sample_once()

        sampler.update_row_weights([0, 1], {2, 3})

        a_sample = sampler.sample_once()
        assert isinstance(a_sample, set)

        # samples = sampler.sample(5)
        # assert len(samples) == 5
        # for sample in samples:
        #     assert isinstance(sample, set)
        
