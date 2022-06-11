import pytest
import numpy as np

from scipy import sparse as sp

from dmrs.greedy.submod import GreedySubmod
from dmrs.rule import Rule
from dmrs.samplers.dummy import DummyHeadSampler, DummyTailSampler
from dmrs.samplers.discrim import CFTPDiscriminativitySampler
from dmrs.samplers.uncovered_area import UncoveredAreaSampler
from dmrs.utils import convert_sets_to_matrix, csr_matrix_equal
from dmrs.exceptions import InsufficientCandidates


class TestGreedySubmod():
    @pytest.fixture
    def dataset(self):
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

    @pytest.fixture
    def dummy_head_sampler(self):
        return DummyHeadSampler()

    @pytest.fixture
    def dummy_tail_sampler(self):
        return DummyTailSampler()


    @pytest.fixture
    def dummy_greedy_alg(self, dummy_head_sampler, dummy_tail_sampler):
        return GreedySubmod(
            dummy_head_sampler, dummy_tail_sampler,
            n_heads_per_tail=5, n_tails_per_iter=5
        )
    # @pytest.mark.parametrize('rules, coverage',
    #                          [
    #                              ([Rule((0, ), (0, )), Rule((1, ), (1, ))], 4),
    #                              ([Rule((0, ), (0, ))], 2)
    #                          ])
    # def test_objective(self, dataset, dummy_greedy_alg, rules, coverage):
    #     X, Y = dataset
    #     alg = dummy_greedy_alg
    #     alg.lambd = 100.

    #     alg.set_X_Y(X, Y)

    #     for r in rules:
    #         r.bind_dataset(X, Y)

    #     alg.selected_rules = rules
    #     actual = alg.objective(rules)
    #     expected = (
    #         sum([r.KL() for r in rules])
    #         + alg.lambd * coverage
    #     )

        np.testing.assert_allclose(actual, expected)
        
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

    def test_fit_one_rule_only(self, dummy_greedy_alg, dataset):
        """just learn on rule
        """
        X, Y = dataset
        n_max_rules = 1
        alg = dummy_greedy_alg
        alg.n_max_rules = n_max_rules
        
        learned_rules = alg.fit_1st_round(X, Y)

        assert learned_rules == alg.selected_rules
        assert alg.num_rules_selected == n_max_rules

        for rule in alg.selected_rules:
            assert isinstance(rule, Rule)

        assert 0 < alg.coverage_ratio() < 1.0  # shouldn't cover fully

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
        alg = GreedySubmod(head_sampler, tail_sampler, n_tails_per_iter=3)
        alg.n_max_rules = 10
        alg.fit(X, Y)
        assert alg.coverage_ratio() == 1.0  # all label occ should be covered    
        assert alg.num_rules_selected > 1

