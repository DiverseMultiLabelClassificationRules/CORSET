import numpy as np
from scipy import sparse as sp

from .base import Sampler
from ..utils import convert_matrix_to_sets

class BoleyFrequencySampler(Sampler):

    def __init__(self, random_state=None):
        self.random_state = random_state
        
    def fit(self, Y: sp.csr_matrix):
        self.rows = convert_matrix_to_sets(Y)
        self.nrows = len(self.rows)
        # assign weights to each tuple
        weights = np.power(2, list(map(len, self.rows)))
        self.probas = weights / weights.sum()

    def sample_once(self, seed=None):
        np.random.seed(seed)

        # loop until a non-empty set is returned
        # without the loop, empty sets is likely to be returned for small tuple, e.g., size 1 or 2
        while True:
            sampled_tuple = tuple(self.rows[np.random.choice(np.arange(self.nrows), p=self.probas)])
            item_idx = (np.random.rand(len(sampled_tuple)) < 0.5).nonzero()[0]
            res = set([sampled_tuple[i] for i in item_idx])
            if len(res) > 0:
                return res


class SURSFrequencySamplerMixin:
    """sample according to frequency, assuming the sample space is reduced

    the sample space is obtained via `self._sample_space`

    the valid itemsets contained by each data record is stored in `self.new_rows`,
    where each row is a list of itemset ids
    """
    def _assign_row_weights(self):
        self.row_weights = np.array(list(map(len, self.new_rows)))

    def _sample_itemset_from_row(self, row_id):
        """
        requirement: self._sample_space
        """
        # for the sampled row, sample an itemset
        # then for the row, sample an itemset uniformly randomly
        itemset_ids = self.new_rows[row_id]

        sampled_itemset_id = np.random.choice(itemset_ids)
        return self._sample_space[sampled_itemset_id]            
