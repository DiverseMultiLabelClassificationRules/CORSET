import pytest
import numpy as np
from dmrs.samplers.cftp import CFTP
from collections import Counter
from copy import copy


class TestCFTP:
    @pytest.fixture
    def input_kwargs(self):
        W_pos_dict = {0: 2, 1: 2}
        W_neg_dict = {2: 1, 3: 1}
        data_records = [{0, 1}, {0, 1}, set(), set()]
        pos_list, neg_list = [0, 1], [2, 3]

        ret = locals()
        del ret["self"]
        return ret

    def test_init(self, input_kwargs):
        cftp = CFTP(**input_kwargs)
        np.testing.assert_allclose(cftp.pos_weights, [2, 2])
        np.testing.assert_allclose(cftp.neg_weights, [1, 1])

        np.testing.assert_allclose(cftp.pos_proba, [0.5, 0.5])
        np.testing.assert_allclose(cftp.neg_proba, [0.5, 0.5])

    def test_add_data(self, input_kwargs):
        cftp = CFTP(**input_kwargs)
        n_iters = 2
        cftp.add_data(n_iters)

        assert len(cftp.u_list) == n_iters
        assert len(cftp.C_pos_list) == n_iters
        assert len(cftp.C_neg_list) == n_iters

        u_list_prev = copy(cftp.u_list)
        C_pos_list_prev = copy(cftp.C_pos_list)
        C_neg_list_prev = copy(cftp.C_neg_list)

        n_iters = 4
        cftp.add_data(n_iters)
        assert len(cftp.u_list) == n_iters
        assert len(cftp.C_pos_list) == n_iters
        assert len(cftp.C_neg_list) == n_iters
        assert cftp.u_list[-2:] == u_list_prev
        assert cftp.C_pos_list[-2:] == C_pos_list_prev
        assert cftp.C_neg_list[-2:] == C_neg_list_prev

    def test_add_data_statistical_property(self, input_kwargs):
        cftp = CFTP(**input_kwargs)

        n_reps = 100000

        cftp.add_data(n_reps)
        pos_cnt, neg_cnt = Counter(cftp.C_pos_list), Counter(cftp.C_neg_list)

        for pos in {0, 1}:
            np.testing.assert_allclose(pos_cnt[pos] / n_reps, 0.5, rtol=1e-2)

        for neg in {0, 1}:
            np.testing.assert_allclose(pos_cnt[pos] / n_reps, 0.5, rtol=1e-2)

    def test_simulate_backwards(self, input_kwargs):
        cftp = CFTP(**input_kwargs)
        # simulate two iterations
        cftp.C_pos_list = [0, 1]
        cftp.C_neg_list = [2, 2]
        cftp.u_list = [0.5, 0.5]
        actual_D, actual_history = cftp.simulate_backwards(return_history=True)

        expected_history = [
            dict(
                pos=0,
                neg=2,
                u=0.5,
                D=None,
                W_C=2,
                W_D_bar=1,
                W_C_bar=2,
                W_D=1,
                ratio=1.0,
            ),
            dict(
                pos=1,
                neg=2,
                u=0.5,
                D=(0, 2),
                W_C=2,
                W_D_bar=2,
                W_C_bar=2,
                W_D=2,
                ratio=1.0,
            ),
        ]
        expected_D = (1, 2)

        assert expected_D == actual_D
        assert expected_history == actual_history

        actual_D = cftp.simulate_backwards(return_history=False)
        assert expected_D == actual_D

    def test_sample(self, input_kwargs):
        # np.random.seed(12345)
        # random.seed(12345)
        cftp = CFTP(**input_kwargs, max_iters=1)
        n_reps = 5000
        samples = cftp.sample_k_times(n_reps)
        assert len(samples) == n_reps

        cnt = Counter(samples)
        actual_proba = {}
        for k, v in cnt.items():
            actual_proba[k] = v / n_reps

        assert len(cnt) == 4
        for k, v in cnt.items():
            np.testing.assert_allclose(actual_proba[k], 1 / 4, rtol=1e-1)
