import pytest
import numpy as np
from scipy import sparse as sp
from dmrs.data import Dataset

@pytest.fixture
def Y():
    return sp.csr_matrix([[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

@pytest.fixture
def toy_dataset():
    ds = Dataset()

    trn_Y = sp.csr_matrix([[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    trn_X = sp.random(trn_Y.shape[0], 5, density=0.5).tocsr()

    tst_X = trn_X.copy()
    tst_Y = trn_Y.copy()
    ds.set_data(trn_X, trn_Y, tst_X, tst_Y)

    ds.build_confidence_graph()
    return ds


@pytest.fixture
def random_dataset():
    np.random.seed(12345)
    
    ds = Dataset()
    N, D, L = 10, 25, 30
    trn_Y_dense = np.random.choice([0, 1], size=(N * L,), p=[9/10., 1/10.]).reshape(N, L)
    trn_Y = sp.csr_matrix(trn_Y_dense)
    trn_X = sp.random(trn_Y.shape[0], D, density=0.5).tocsr()

    tst_X = trn_X.copy()
    tst_Y = trn_Y.copy()
    ds.set_data(trn_X, trn_Y, tst_X, tst_Y)

    ds.build_confidence_graph()
    print('random dataset generated')
    return ds
