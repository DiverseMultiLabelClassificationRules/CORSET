import pytest
import numpy as np
from scipy import sparse as sp
from collections import Counter

from corset.samplers.boley_discrim import BoleyCFTPDiscriminativitySampler
from corset.samplers.boley_cftp import BoleyCFTP
from corset.utils import counter2proba, convert_matrix_to_sets_v2


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

    y = np.array([1] * 3 + [0] * 3)

    return X, y


@pytest.fixture
def ground_truth_proba_for_pairs():
    """mapping from a tuple of row ids to the sampling probability"""
    counts = {
        (0, 3): 2,  # {0, 1}, {1, 2, 3} -> (2^1 - 1) 2^1 = 2 <- {0} and {1}
        (0, 4): 2,  # {0, 1}, {1, 2} -> (2^1 - 1) 2^1 = 2    <- {0} and {1}
        (0, 5): 3,  # {0, 1}, {2} -> (2^2 - 1)= 3            <- {0, 1} and {}
        (1, 3): 1,  # {0}, {1, 2, 3} -> (2^1 - 1) = 1        <- {0} and {}
        (1, 4): 1,  # {0}, {1, 2} -> (2^1 - 1) = 1           <- {0} and {}
        (1, 5): 1,  # {0}, {2} -> (2^1 - 1) = 1              <- {0} and {}
        # (2, 3): 0,  # {2, 3}, {1, 2, 3} -> (2^0 - 1) = 0   <- {} and {2, 3} drop it
        (2, 4): 2,  # {2, 3}, {1, 2} -> (2^1 - 1) 2^1 = 2    <- {3} and {2}
        (2, 5): 2,  # {2, 3}, {2} -> (2^1 - 1) 2^1 = 2       <- {3} and {2}
    }
    return counter2proba(counts)


@pytest.fixture
def ground_truth_proba_for_samples():
    # use the comments in ground_truth_proba_for_pairs to calculate the ground truth
    counts = {
        (0,): 6,
        (1,): 1,
        # (2, ): 0,
        (3,): 2,
        (0, 1): 3,
        # (1, 2): 0,
        (2, 3): 2,
        # (0, 3): 0,
        # (0, 1, 2): 0,
        # (1, 2, 3): 0,
        # (0, 2, 3): 0,
        # (0, 1, 3): 0,
        # (0, 1, 2, 3): 0,
    }
    return counter2proba(counts)


@pytest.fixture
def toy_sampler(input_data):
    sampler = BoleyCFTPDiscriminativitySampler(random_state=12345)

    X, y = input_data

    sampler._populate_rows(X)

    return sampler


def test_preprocessing(input_data, toy_sampler):
    X, y = input_data
    sampler = toy_sampler
    sampler._check_data(X, y)

    # test _compute_pos_and_neg_ids
    sampler._compute_pos_and_neg_info(y)
    assert sampler._positives == set(range(3))
    assert sampler._negatives == set(range(3, 6))

    assert sampler._pos_list == [0, 1, 2]
    assert sampler._neg_list == [3, 4, 5]

    # test _compute_weight_ub
    sampler._compute_weight_dict(X, y)
    actual = sampler._weight_dict
    expected = {0: 3, 1: 1, 2: 3, 3: 1, 4: 1, 5: 1}
    assert actual == expected

    np.testing.assert_allclose(sampler._pos_weight, np.array([3, 1, 3]))

    sampler._construct_cftp()
    assert isinstance(sampler.cftp, BoleyCFTP)


def test_fit_and_sample_a_pair(input_data, toy_sampler):
    # assert isinstance(sample, tuple)
    X, y = input_data
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
    X, y = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    actual = sampler.get_ground_truth_proba_for_pairs()
    expected = ground_truth_proba_for_pairs
    assert actual == expected


def test_ground_truth_proba_for_samples(
    input_data, toy_sampler, ground_truth_proba_for_samples
):
    X, y = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    actual = sampler.get_ground_truth_proba_for_samples()
    expected = ground_truth_proba_for_samples
    assert actual == expected


def test_sample_a_pair_correctness(
    input_data, toy_sampler, ground_truth_proba_for_pairs
):
    X, y = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    sample_size = 20000
    pairs = [sampler.sample_a_pair() for _ in range(sample_size)]
    actual_counts = Counter(pairs)

    actual_proba = counter2proba(actual_counts)

    assert set(actual_proba.keys()) == set(ground_truth_proba_for_pairs.keys())
    for k, v in actual_proba.items():
        np.testing.assert_allclose(
            actual_proba[k], ground_truth_proba_for_pairs[k], rtol=1e-1
        )


def test_sample_once_correctness(
    input_data, toy_sampler, ground_truth_proba_for_samples
):
    X, y = input_data
    sampler = toy_sampler
    sampler.fit(X, y)

    sample_size = 10000
    samples = [tuple(sorted(sampler.sample_once())) for _ in range(sample_size)]

    actual_counts = Counter(samples)

    assert sum(actual_counts.values()) == sample_size

    actual_proba = counter2proba(actual_counts)
    assert set(actual_proba.keys()) == set(ground_truth_proba_for_samples.keys())

    # print("actual_proba: ", actual_proba)
    # print("ground_truth_proba_for_samples: ", ground_truth_proba_for_samples)

    for k, v in actual_proba.items():
        np.testing.assert_allclose(
            actual_proba[k], ground_truth_proba_for_samples[k], rtol=1e-1
        )


def test_end2end(input_data):
    sampler = BoleyCFTPDiscriminativitySampler(random_state=12345)
    X, y = input_data
    sampler.fit(X, y)
    sample = sampler.sample_once()
    assert isinstance(sample, set)
    assert len(sample) > 0
