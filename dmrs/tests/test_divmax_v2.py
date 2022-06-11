import pytest
import numpy as np
import pickle as pkl

from scipy import sparse as sp

from dmrs.greedy.divmax_v2 import GreedyDivMaxV2, GreedyCFTPDivMaxV2
from dmrs.rule import Rule
from dmrs.samplers.dummy import DummyHeadSampler, DummyTailSampler
from dmrs.samplers.discrim import CFTPDiscriminativitySampler
from dmrs.samplers.uncovered_area import UncoveredAreaSampler
from dmrs.utils import convert_sets_to_matrix, csr_matrix_equal
from dmrs.exceptions import InsufficientCandidates


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


class TestGreedyDivMaxV2():
    greedy_cls = GreedyDivMaxV2

    @pytest.fixture
    def dummy_head_sampler(self):
        return DummyHeadSampler()

    @pytest.fixture
    def dummy_tail_sampler(self):
        return DummyTailSampler()


    @pytest.fixture
    def dummy_greedy_alg(self, dummy_head_sampler, dummy_tail_sampler):
        return GreedyDivMaxV2(
            dummy_head_sampler, dummy_tail_sampler,
            n_heads_per_tail=5, n_tails_per_iter=5,
            n_max_rules=10
        )

    def test_select_best_candidate(self, dummy_greedy_alg, dataset):
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        alg.tail_sampler.fit(Y)
        cands = alg.generate_candidate_rules()

        rule = alg.select_best_candidate(cands)
        assert isinstance(rule, Rule)

    def test_should_stop(self, dummy_greedy_alg, dataset):
        X, Y = dataset
        alg = dummy_greedy_alg
        dummy_greedy_alg.X, dummy_greedy_alg.Y = X, Y

        # when n_max_rules is set
        alg.selected_rules = [None] * 5
        alg.n_max_rules = 5
        assert alg.should_stop()

    def test_fit_until_fully_covered(self, dummy_greedy_alg, dataset):
        """we fit until all label occurence is covered
        """
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.n_max_rules = None
        
        learned_rules = alg.fit(X, Y)
        
        assert alg.num_rules_selected > 1

        assert learned_rules == alg.selected_rules
        for rule in alg.selected_rules:
            assert isinstance(rule, Rule)

        assert alg.coverage_ratio() == 1.0  # all label occurence is covered

    def test_fit_one_rule_only(self, dummy_greedy_alg, dataset):
        """just learn on rule
        """
        X, Y = dataset
        n_max_rules = 1
        alg = dummy_greedy_alg
        alg.n_max_rules = n_max_rules
        
        alg.fit(X, Y)

        assert alg.num_rules_selected == n_max_rules

        for rule in alg.selected_rules:
            assert isinstance(rule, Rule)

        assert 0 < alg.coverage_ratio() < 1.0  # shouldn't cover fully

    def test_fit_until_fully_covered(self, dummy_greedy_alg, dataset):
        """we fit until all label occurence is covered
        """
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.n_max_rules = 10
        
        alg.fit(X, Y)
        
        assert alg.num_rules_selected > 1
        
        for rule in alg.selected_rules:
            assert isinstance(rule, Rule)

        assert alg.coverage_ratio() == 1.0  # all label occurence is covered
        
    @pytest.mark.parametrize('min_edge_proba',  # this parameter controls how large the sample space is
                             [0.5, 0.8]
                             )
    def test_runnable_for_nontrivial_samplers(self, dataset, min_edge_proba):
        X, Y = dataset
        head_sampler = CFTPDiscriminativitySampler(min_proba=min_edge_proba)
        tail_sampler = UncoveredAreaSampler(min_proba=min_edge_proba)
        alg = self.__class__.greedy_cls(head_sampler, tail_sampler, n_tails_per_iter=3)
        alg.n_max_rules = 10
        alg.fit(X, Y)
        assert alg.coverage_ratio() == 1.0  # all label occ should be covered    
        assert alg.num_rules_selected > 1


class TestGreedyCFTPDivMaxV2():
    def test_get_params(self):
        alg = GreedyCFTPDivMaxV2(min_feature_proba=0.7, min_label_proba=0.5,
                                 lambd=0.7,
                                 n_tails_per_iter=1000,
                                 n_heads_per_tail=99,
                                 n_max_rules=100)
        actual = alg.get_params()
        expected = {}
        expected['lambd'] = 0.7
        expected['n_tails_per_iter'] = 1000
        expected['n_heads_per_tail'] = 99
        expected['n_max_rules'] = 100
        expected['min_feature_proba'] = 0.7
        expected['min_label_proba'] = 0.5
        assert actual == expected

    def test_runnable(self, dataset):
        X, Y = dataset
        alg = GreedyCFTPDivMaxV2(n_max_rules=3)
        alg.fit(X, Y)
        assert alg.coverage_ratio() == 1.0  # all label occ should be covered    
        assert alg.num_rules_selected > 1        
        

    def test_pickle(self, dataset):
        X, Y = dataset
        rules = [Rule((0, ), (0, ))]
        for r in rules:
            r.bind_dataset(X.tocsc(), Y.tocsc())
            
        # init a classifier
        alg = GreedyCFTPDivMaxV2(n_max_rules=3)
        alg.selected_rules = rules
        alg.fit(X, Y)
        expected_preds = alg.predict(X)

        alg_data = pkl.dumps(alg)
        
        # save it
        alg_copy = pkl.loads(alg_data)
        assert alg.selected_rules == alg_copy.selected_rules
        actual_preds = alg_copy.predict(X)

        np.testing.assert_allclose(expected_preds.todense(), actual_preds.todense())
        assert alg.get_params() == alg_copy.get_params()        

        absent_attrs = ['X', 'Y', 'X_csc', 'Y_csc']
        for a in absent_attrs:
            assert not hasattr(alg_copy, a)
