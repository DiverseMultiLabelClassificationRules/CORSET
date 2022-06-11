import numpy as np
from .base import Sampler

class DummyHeadSampler(Sampler):
    """return a random feature
    """    
    def fit(self, X, y):
        self.D = X.shape[1]
        self.random_state = None

    def sample_once(self, *args):
        return {np.random.randint(0, self.D)}


class DummyTailSampler(Sampler):
    """return a random label
    """
    def fit(self, Y):
        self.L = Y.shape[1]
        self.random_state = None

    def sample_once(self, *args):
        return {np.random.randint(0, self.L)}

    def update_row_weights(self, *args):
        pass
