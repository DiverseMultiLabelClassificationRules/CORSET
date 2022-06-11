import pytest
import numpy as np
from scipy import sparse as sp
from dmrs.data import Dataset

@pytest.fixture
def data():
    N, D, L = 10, 25, 30
    trn_Y = sp.csr_matrix(np.ones((N, L)))
    trn_X = sp.csr_matrix(np.ones((N, D)))

    tst_X = trn_X.copy()
    tst_Y = trn_Y.copy()
    return trn_X, trn_Y, tst_X, tst_Y

def test_split_train_True(data):
    ds = Dataset(split_train=True, train_ratio=0.5)
    trn_X, trn_Y, tst_X, tst_Y = data
    ds.set_data(trn_X, trn_Y, tst_X, tst_Y)

    assert ds.trn_X.shape[0] == ds.dev_X.shape[0]
    assert ds.trn_Y.shape[0] == ds.dev_Y.shape[0]
    assert ds.trn_X.shape[0] == ds.trn_Y.shape[0]
    assert ds.ndev == ds.dev_X.shape[0]
    
def test_split_train_False(data):
    ds = Dataset(split_train=False)
    trn_X, trn_Y, tst_X, tst_Y = data
    ds.set_data(trn_X, trn_Y, tst_X, tst_Y)
    
    assert ds.trn_X.shape[0] == ds.trn_Y.shape[0]
    assert not hasattr(ds, 'dev_X')
    assert not hasattr(ds, 'dev_Y')
    with pytest.raises(AttributeError):
        ds.ndev
