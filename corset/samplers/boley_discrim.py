import random
import numpy as np
from scipy import sparse as sp
from itertools import product
from collections import Counter
from tqdm import tqdm
from .base import Sampler
from .assignment import TrieSampleAssignmentMixin, PRETTISampleAssignmentMixin
from .boley_cftp import BoleyCFTP
from ..utils import (
    convert_matrix_to_sets_v2,
    counter2proba,
    draw_bernoulli_elementwise,
    powerset,
)


class BoleyCFTPDiscriminativitySampler(Sampler):
    """
    sampling according to discriminativity using the sampling

    the sampling technique is based on "couping from the past" (CFTP)
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

        self._rows = None
        self._positives = None
        self.N = 0  # number of rows

    def __repr__(self):
        return "dmrs.samplers.discrim.BoleyCFTPDiscriminativitySampler(random_state={})".format(
            self.random_state
        )

    def _populate_rows(self, X):
        """assign the _rows to list of sets"""
        self._rows = convert_matrix_to_sets_v2(X)

    def _rescale_weights(self):
        """rescale the weights to reduce the risk of overflow
        to be implemented
        """
        # num_features = np.array(list(map(len, self._rows)), dtype=np.int64)
        # min_num_features, max_num_features = num_features.min(), num_features.max()

        # max_num_features_diff = max_num_features - min_num_features
        pass

    def _compute_weight_dict(self, X, y):
        """compute the weight of each data record, which is used to construct the weight upperbound"""
        # arrays of weights
        self._weight_dict = {}

        # compute the weight upper bound for each data record
        for i in range(self.N):
            if i in self._positives:
                self._weight_dict[i] = (
                    np.power(2, len(self._rows[i]), dtype=np.float64) - 1
                )
            else:  # no need to store
                self._weight_dict[i] = 1

        self._pos_weight = np.array([self._weight_dict[i] for i in self._pos_list])

    def _check_overflow(self):
        """check if numeric overflow is encountered, possibly caused by the large number of features"""
        if (np.array(list(self._weight_dict.values())) < 0).any():
            raise ValueError(
                "self._weight_dict contains negative values, possibly caused by numeric overflow. "
            )

    def _compute_pos_and_neg_info(self, y):
        """get the indices of positive and negative data records"""
        self._positives, self._negatives = map(
            lambda arr: set(list(arr)), [(y > 0).nonzero()[0], (y == 0).nonzero()[0]]
        )
        self._pos_list = sorted(list(self._positives))
        self._neg_list = sorted(list(self._negatives))

    def _check_data(self, X, y):
        self.N, self.D = X.shape
        assert set(np.unique(y)) == {
            0,
            1,
        }, f"{np.unique(y)} contains elements other than {0, 1}"

    def _construct_cftp(self):
        """construct the CFTP sampler"""
        self.cftp = BoleyCFTP(
            W_pos_dict=self._weight_dict,
            W_neg_dict=self._weight_dict,
            pos_list=self._pos_list,
            neg_list=self._neg_list,
            rows=self._rows,
            max_iters=1024,
            random_state=None,
        )

    # @profile
    def fit(self, X: sp.csr_matrix, y: np.ndarray):
        """X: data records matrix
        y: a binary vector indicating the label of each row
        """

        self._check_data(X, y)

        if self._rows is None:
            self._populate_rows(X)

        self._compute_pos_and_neg_info(y)

        self._compute_weight_dict(X, y)
        self._check_overflow()

        self._construct_cftp()

    def sample_a_pair(self):
        """
        Sample tuple of data according to discriminativity using coupling from the past

        returns
        D  a tuple of data or None if failed within max_iters
        """
        return self.cftp.sample()

    def sample_once(self):
        """sample a pattern (a set of features)"""
        res = self.sample_a_pair()
        if res is None:
            return None
        else:
            pos, neg = res
            row_pos = self._rows[pos]  # samples in pos DR
            row_neg = self._rows[neg]  # samples in neg DR

            p1 = draw_bernoulli_elementwise(row_pos - row_neg, exclude_empty=True)
            p2 = draw_bernoulli_elementwise(
                row_pos.intersection(row_neg), exclude_empty=False
            )
            sample_pattern = p1 | p2
            return sample_pattern

    def sample(self, k, show_progress=False):
        iter_obj = range(k)
        if show_progress:
            iter_obj = tqdm(iter_obj)
        return [self.sample_once() for _ in iter_obj]

    def get_ground_truth_proba_for_pairs(self):
        """
        get the ground truth proba for each positive and negative DR pair

        do not run on large data sets since time complexity: O(N^2)
        """
        weights = {}
        for pos, neg in product(self._pos_list, self._neg_list):
            wgt = (np.power(2, len(self._rows[pos] - self._rows[neg])) - 1) * np.power(
                2, len(self._rows[pos].intersection(self._rows[neg]))
            )
            if wgt > 0:
                weights[(pos, neg)] = wgt
        return counter2proba(weights)

    def get_ground_truth_proba_for_samples(self):
        """get the truth probability for each sample

        do not run on large data sets since time complexity: O(N^2)"""
        cnt = Counter()
        for pos, neg in product(self._pos_list, self._neg_list):
            # fix the thing below!
            diff = self._rows[pos] - self._rows[neg]
            intersect = self._rows[pos].intersection(self._rows[neg])
            p1_ps = powerset(diff, exclude_empty=True)
            p2_ps = powerset(intersect, exclude_empty=False)
            for p1, p2 in product(p1_ps, p2_ps):
                sample = tuple(sorted(p1 + p2))
                cnt[sample] += 1

        return counter2proba(cnt)
