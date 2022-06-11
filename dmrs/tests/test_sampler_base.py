import pytest
import numpy as np
from scipy import sparse as sp

from dmrs.samplers.pc import PCFrequencySampler
from .fixtures import random_dataset
from dmrs.samplers.assignment import NaiveSampleAssignmentMixin


def test_sample_assignment_equality(random_dataset):
    # inherit from PCFrequencySampler to get the sample space
    class NaiveClass(NaiveSampleAssignmentMixin, PCFrequencySampler):
        pass

    TrieClass = PCFrequencySampler

    naive = NaiveClass(min_proba=0.3)
    trie_based = TrieClass(min_proba=0.3)

    def get_new_rows(obj):
        obj._build_graph(random_dataset.trn_Y)
        obj._build_sample_space()
        obj._generate_new_rows(random_dataset.trn_Y)

        return list(map(set, obj.new_rows))
    expected = get_new_rows(naive)
    actual = get_new_rows(trie_based)

    assert actual == expected
