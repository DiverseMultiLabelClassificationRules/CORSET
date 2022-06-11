import pytest
import numpy as np
from scipy import sparse as sp
from collections import Counter

from dmrs.samplers.discrim import CFTPDiscriminativitySampler
from dmrs.samplers.cftp import CFTP


@pytest.fixture
def input_data():
    X = sp.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
            ]
        )
    )

    sample_space = {(1,), (2,), (3,), (4,), (1, 2), (2, 3)}
    new_rows = [[0, 1], [0], [2, 3, 5], [1, 2, 3, 5], [0, 1, 4], [2]]

    y = np.array([1] * 3 + [0] * 3)

    return X, y, sample_space, new_rows


@pytest.fixture
def ground_truth_proba_for_pairs():
    """mapping from a tuple of row ids to the sampling probability"""
    counts = {
        (0, 3): 1,
        (0, 4): 0,
        (0, 5): 2,
        (1, 3): 1,
        (1, 4): 0,
        (1, 5): 1,
        (2, 3): 0,
        (2, 4): 3,
        (2, 5): 2,
    }
    total = sum(counts.values())
    probas = {}
    for k, v in counts.items():
        probas[k] = v / total

    return probas


@pytest.fixture
def ground_truth_proba_for_samples():
    counts = {
        0: 4,
        1: 1,
        2: 1,
        3: 2,
        # 4: 0,
        5: 2,
    }
    total = sum(counts.values())
    probas = {}
    for k, v in counts.items():
        probas[k] = v / total
    return probas


@pytest.fixture
def toy_sampler(input_data):
    sampler = CFTPDiscriminativitySampler(random_state=12345)

    X, y, sample_space, new_rows = input_data
    N, D = X.shape

    # artificially inject sample space and new_rows
    sampler._sample_space = sample_space
    sampler.new_rows = new_rows

    return sampler


def test_preprocessing(input_data, toy_sampler):
    X, y, sample_space, new_rows = input_data
    sampler = toy_sampler
    sampler._check_data(X, y)

    # test _compute_pos_and_neg_ids
    sampler._compute_pos_and_neg_info(y)
    assert sampler._positives == set(range(3))
    assert sampler._negagives == set(range(3, 6))

    assert sampler._pos_list == [0, 1, 2]
    assert sampler._neg_list == [3, 4, 5]

    # test _compute_weight_ub
    sampler._compute_weight_dict(X, y)
    actual = sampler._weight_dict
    expected = {0: 2, 1: 1, 2: 3, 3: 1, 4: 1, 5: 1}
    assert actual == expected

    np.testing.assert_allclose(sampler._pos_weight, np.array([2, 1, 3]))

    sampler._construct_cftp()
    assert isinstance(sampler.cftp, CFTP)


def test_fit_and_sample_a_pair(input_data, toy_sampler):
    # assert isinstance(sample, tuple)
    X, y, _, _ = input_data
    sampler = toy_sampler

    sampler.fit(X, y)
    for _ in np.arange(10):
        pair = sampler.sample_a_pair()
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert y[pair[0]] == 1
        assert y[pair[1]] == 0


def test_ground_truth_proba_for_pairs(
    input_data, toy_sampler, ground_truth_proba_for_pairs
):
    X, y, _, _ = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    actual = sampler.get_ground_truth_proba_for_pairs()
    expected = ground_truth_proba_for_pairs
    assert actual == expected


def test_ground_truth_proba_for_samples(
    input_data, toy_sampler, ground_truth_proba_for_samples
):
    X, y, _, _ = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    actual = sampler.get_ground_truth_proba_for_samples()
    expected = ground_truth_proba_for_samples
    assert actual == expected


def test_sample_a_pair_correctness(
    input_data, toy_sampler, ground_truth_proba_for_pairs
):
    X, y, _, _ = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    sample_size = 10000
    pairs = [sampler.sample_a_pair() for _ in range(sample_size)]
    actual_counts = Counter(pairs)

    actual_proba = dict(actual_counts)
    assert sum(actual_counts.values()) == sample_size

    for k, v in actual_proba.items():
        actual_proba[k] = v / sample_size

    for k, v in actual_proba.items():
        np.testing.assert_allclose(
            actual_proba[k], ground_truth_proba_for_pairs[k], rtol=1e-1
        )


def test_sample_once_correctness(
    input_data, toy_sampler, ground_truth_proba_for_samples
):
    X, y, _, _ = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    sample_size = 10000
    samples = [sampler.sample_once(return_sample_index=True) for _ in range(sample_size)]

    actual_counts = Counter(samples)

    actual_proba = dict(actual_counts)
    assert sum(actual_counts.values()) == sample_size

    for k, v in actual_proba.items():
        actual_proba[k] = v / sample_size

    print("actual_proba: ", actual_proba)
    print("ground_truth_proba_for_samples: ", ground_truth_proba_for_samples)

    for k, v in actual_proba.items():
        np.testing.assert_allclose(
            actual_proba[k], ground_truth_proba_for_samples[k], rtol=1e-1
        )

def test_end2end(input_data):
    sampler = CFTPDiscriminativitySampler(random_state=12345)
    X, y = input_data[:2]
    sampler.fit(X, y)
    sample = sampler.sample_once(return_sample_index=False)
    assert isinstance(sample, set)
    assert len(sample) > 0
