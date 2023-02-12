import numpy as np
from scipy import sparse as sp
from scipy.special import binom
from .base import Sampler
from ..utils import convert_matrix_to_sets, flatten

class BoleyAreaSampler(Sampler):
    """sampling according to area function under Boley's local sampling framework
    """
    def __init__(self, random_state=None):
        self.random_state = random_state
        
    def fit(self, Y: sp.csr_matrix):
        # filter out empty rows
        nnz_per_row = flatten(Y.sum(axis=1))
        Y = Y[nnz_per_row > 0]

        self.rows = convert_matrix_to_sets(Y)
        sizes = np.array(list(map(len, self.rows)))
                
        self.nrows = len(self.rows)

        assert (sizes > 0).all()
        # assign weights to each tuple        
        weights = np.power(2, sizes - 1) * sizes
        self.probas = weights / weights.sum()

    def sample_once(self, seed=None):
        np.random.seed(seed)

        sampled_tuple = tuple(self.rows[np.random.choice(np.arange(self.nrows), p=self.probas)])

        size = len(sampled_tuple)
        k_weights = np.array([binom(size, k) * k for k in range(1, size+1)])
        k = np.random.choice(np.arange(1, size+1), p=k_weights / k_weights.sum())
        itemset = set(np.random.permutation(sampled_tuple)[:k])
        assert len(itemset) == k
        return itemset
