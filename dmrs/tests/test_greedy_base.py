import pytest
import pickle as pkl
import numpy as np
from scipy import sparse as sp
from dmrs.greedy.base import GreedyBase
from dmrs.rule import Rule

@pytest.fixture
def dataset():
    X = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])
    Y = np.array([
        [1, 1],
        [0, 1],
        [1, 0]
    ])
    return sp.csr_matrix(X), sp.csr_matrix(Y)

class TestGreedyBase():
    def test_pickle(self, dataset):
        X, Y = dataset
        rules = [Rule((0, ), (0, ))]
        for r in rules:
            r.bind_dataset(X.tocsc(), Y.tocsc())
            
        # init a classifier
        alg = GreedyBase()
        alg.selected_rules = rules
        alg.fit(X, Y)
        expected_preds = alg.predict(X)

        alg_data = pkl.dumps(alg)
        
        # save it
        alg_copy = pkl.loads(alg_data)
        assert alg_copy.selected_rules == alg.selected_rules
        actual_preds = alg_copy.predict(X)

        assert alg_copy.num_labels == 2
        assert alg_copy.num_features == 3
        np.testing.assert_allclose(expected_preds.todense(), actual_preds.todense())
        assert alg.get_params() == alg_copy.get_params()

        # just double check 
        abssent_attrs = ['X', 'Y', 'N', 'D', 'L']
        for r in alg_copy.selected_rules:
            for attr in abssent_attrs:
                assert not hasattr(r, attr)


    def test_get_params(self, dataset):
        X, Y = dataset
        alg = GreedyBase(random_state=123)
        alg.fit(X, Y)

        params = alg.get_params()
        assert params['random_state'] == 123
        absent_attrs = ['X', 'Y', 'X_csc', 'Y_csc', 'N', 'D', 'L']
        for attr in absent_attrs:
            assert attr not in params
