import numpy as np
from scipy import sparse as sp

from collections import Counter
from tqdm import tqdm


class Sampler:
    def fit(self, *args):
        raise NotImplementedError

    def sample_once(self, seed=None):
        raise NotImplementedError

    def sample(self, size=1):
        np.random.seed(self.random_state)
        return [self.sample_once(np.random.randint(100000)) for _ in range(size)]
