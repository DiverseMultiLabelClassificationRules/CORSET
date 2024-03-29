import random
import numpy as np
from scipy import sparse as sp
from itertools import product
from collections import Counter
from typing import List

from .base import Sampler
from .assignment import TrieSampleAssignmentMixin, PRETTISampleAssignmentMixin
from .cftp import CFTP
from .pc import PCSampleSpaceConstructor


class CFTPDiscriminativitySampler(
    PCSampleSpaceConstructor,
    PRETTISampleAssignmentMixin,
    # TrieSampleAssignmentMixin,
    Sampler,
):
    """
    sampling according to discriminativity using the sampling
    under reduced sample space framework

    the sampling technique is based on "couping from the past" (CFTP)

    the proposal probability distribution in CFTP is based on Martino's modified version,
    which is tight as well
    """

    def __init__(self, do_prune_edges=True, min_proba=0.5, proposal_distribution="tight", random_state=None):
        assert proposal_distribution in ('tight', 'original')
        
        self.random_state = random_state
        self.dfs_backend = "dfs_v2"
        self.do_prune_edges = do_prune_edges
        self.min_proba = min_proba
        self.proposal_distribution = proposal_distribution
        
        # other attributes
        self._sample_space = None
        self._positives = None
        self.N = 0  # number of rows

    def __repr__(self):
        return "dmrs.samplers.discrim.CFTPDiscriminativitySampler(min_proba={})".format(
            self.min_proba
        )

    def _compute_weight_dict(self, X, y):
        """compute the weight of each data record, which is used to construct the weight upperbound (the proposal distribution)"""
        # arrays of weights
        self._weight_dict = {}

        # compute the weight upper bound for each data record, according to a specific proposal distribution
        if self.proposal_distribution == 'tight':
            for i in range(self.N):
                if i in self._positives:
                    self._weight_dict[i] = len(self.new_rows[i])
                else:  # no need to store
                    self._weight_dict[i] = 1

            # self._pos_weight = np.array([self._weight_dict[i] for i in self._pos_list])

        elif self.proposal_distribution == 'original':
            sample_space_size = len(self._sample_space)
            for i in range(self.N):
                if i in self._positives:
                    self._weight_dict[i] = np.sqrt(len(self.new_rows[i]))
                else:  # no need to store
                    self._weight_dict[i] = np.sqrt(sample_space_size - len(self.new_rows[i]))

    def _compute_pos_and_neg_info(self, y):
        self._positives, self._negagives = map(
            lambda arr: set(list(arr)), [(y > 0).nonzero()[0], (y == 0).nonzero()[0]]
        )
        self._pos_list = sorted(list(self._positives))
        self._neg_list = sorted(list(self._negagives))

    def _check_data(self, X, y):
        self.N, self.D = X.shape
        assert set(np.unique(y)) == {
            0,
            1,
        }, f"class labels in the data {np.unique(y)} != {0, 1}"

    def _construct_cftp(self):
        """construct the CFTP sampler"""
        self.cftp = CFTP(
            W_pos_dict=self._weight_dict,
            W_neg_dict=self._weight_dict,
            pos_list=self._pos_list,
            neg_list=self._neg_list,
            data_records=self.new_rows,
            max_iters=1024,
            random_state=None,
        )

    def build_sample_space_and_generate_rows(self, X: sp.csr_matrix):
        """materialize self._sample_space and self.new_rows"""
        # call the two methods below from `PCSampleSpaceConstructor`
        # which materializes the _sample_space field
        self._build_graph(X)
        self._build_sample_space()

        # call thw method below from `TrieSampleAssignmentMixin`
        # which materializes the new_rows field
        self._generate_new_rows(X)

    # @profile
    def fit(self, X: sp.csr_matrix, y: np.ndarray):
        """X: data records matrix
        y: a binary vector indicating the label of each row
        """

        self._check_data(X, y)

        # only build the sample space once
        if self._sample_space is None or self.new_rows is None:
            self.build_sample_space_and_generate_rows(X)

        self._compute_pos_and_neg_info(y)

        self._compute_weight_dict(X, y)

        self._construct_cftp()

    def sample_a_pair(self, return_details=False):
        """
        Sample tuple of data according to discriminativity using coupling from the past

        returns
        D  a tuple of data or None if failed within max_iters
        """
        return self.cftp.sample(return_details=return_details)

    def sample_once(self, return_sample_index=False, return_cftp_details=False):
        """if return_sample_index is True, return the sample index in the reduced sample space, instead of its content"""
        res = self.sample_a_pair(return_details=return_cftp_details)

        if return_cftp_details:
            pair, cftp_details = res
        else:
            pair = res

        if pair is None:
            return None
        else:
            pos, neg = pair
            dpos = set(self.new_rows[pos])  # samples in pos DR
            dneg = set(self.new_rows[neg])  # samples in neg DR

            sample = random.choice(
                list(dpos.difference(dneg))
            )  # true weight of this pair
            if return_sample_index:
                sample_result = sample
            else:
                sample_result = self._sample_space[sample]

        if return_cftp_details:
            return sample_result, cftp_details
        else:
            return sample_result

    def sample(self, k, return_sample_index=False, return_cftp_details=False):
        return [
            self.sample_once(return_sample_index, return_cftp_details=return_cftp_details)
            for _ in range(k)
        ]

    def get_ground_truth_proba_for_pairs(self):
        """
        get the ground truth proba for each positive and negative DR pair

        do not run on large data sets since time complexity: O(N^2)
        """
        proba = {}
        total = 0
        for pos, neg in product(self._pos_list, self._neg_list):
            cnt = len(set(self.new_rows[pos]) - set(self.new_rows[neg]))
            proba[(pos, neg)] = cnt
            total += cnt

        for k in proba:
            proba[k] /= total

        return proba

    def get_ground_truth_proba_for_samples(self):
        """get the truth probability for each sample

        do not run on large data sets since time complexity: O(N^2)"""
        cnt = Counter()
        for pos, neg in product(self._pos_list, self._neg_list):
            for sample in set(self.new_rows[pos]) - set(self.new_rows[neg]):
                cnt[sample] += 1

        total = sum(cnt.values())

        proba = {}
        for k in cnt:
            proba[k] = cnt[k] / total

        return proba
