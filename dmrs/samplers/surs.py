import numpy as np

from scipy import sparse as sp

from .base import Sampler


class ReducedSpaceSampler(Sampler):
    """sample from a reduced sample space using Boley's local sampling framework"""

    def _build_sample_space(self):
        raise NotImplementedError("self._sample_space should be assigned some value"
                                  "you may consider mixin using PCSampleSpaceConstructor")

    def _generate_new_rows(self, Y):
        raise NotImplementedError('should be implemented by some sample assignment mixin')

    def _assign_row_weights(self):
        """assign weights to each row"""
        raise NotImplementedError

    def _sample_itemset_from_row(self, row_id):
        """given a row id, sample a clique from that row"""
        raise NotImplementedError

    def fit(self, Y: sp.csr_matrix):
        """
        sample cliques, produce the new matrix, and update row weights
        """
        self._build_sample_space()

        self._generate_new_rows(Y)
        self._assign_row_weights()
        self.row_probas = self.row_weights / self.row_weights.sum()

    def sample_once(self, seed=None):
        """take one sample"""
        np.random.seed(seed)
        # sample a row with proba proportional to its weight
        row_id = np.random.choice(np.arange(len(self.new_rows)), p=self.row_probas)

        return self._sample_itemset_from_row(row_id)

