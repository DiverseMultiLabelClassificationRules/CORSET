import pytest
import numpy as np

from scipy import sparse as sp

from corset.greedy.diversity_maximization import GreedyDivMax, GreedyCFTPDivMax
from corset.rule import Rule
from corset.samplers.dummy import DummyHeadSampler, DummyTailSampler
from corset.samplers.discrim import CFTPDiscriminativitySampler
from corset.samplers.uncovered_area import UncoveredAreaSampler
from corset.utils import convert_sets_to_matrix, csr_matrix_equal
from corset.exceptions import InsufficientCandidates


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
    

class TestGreedyDivMax():
    @pytest.fixture
    def dummy_head_sampler(self):
        return DummyHeadSampler()

    @pytest.fixture
    def dummy_tail_sampler(self):
        return DummyTailSampler()


    @pytest.fixture
    def dummy_greedy_alg(self, dummy_head_sampler, dummy_tail_sampler):
        return GreedyDivMax(
            dummy_head_sampler, dummy_tail_sampler,
            n_heads_per_tail=5, n_tails_per_iter=5
        )

    @pytest.mark.parametrize('rules, expected', [
        [[], 0.0],
        [[Rule((0, ), (0, ))], 0.5],
        [[Rule((0, ), (0, )), Rule((1, ), (1, ))], 1.0],
        [[Rule((0, ), (0, )), Rule((1, ), (1, )), Rule((2, ), (1, ))], 1.0]
    ])
    def test_coverage_ratio(self, dummy_greedy_alg, dataset, rules, expected):
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)

        for r in rules:
            r.bind_dataset(X.tocsc(), Y.tocsc())

        alg.selected_rules = rules
        np.testing.assert_allclose(alg.coverage_ratio(), expected)

    @pytest.mark.parametrize('head, tail, expected', [
        ((0, ), (0, ), 0.5),
        ((0, 1), (0, ), 0.25)
    ])
    def test_marginal_coverage_ratio_1(self, dummy_greedy_alg, dataset, head, tail, expected):
        """no rules are added yet
        """
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        
        rule = Rule(head, tail)
        rule.bind_dataset(X.tocsc(), Y.tocsc())

        actual = alg.marginal_coverage_ratio(rule)
        np.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize('head, tail, expected', [
        ((0, ), (0, ), 0.),
        ((0, 1), (0, ), 0),
        ((0, 1), (1, ), 0.25),
    ])
    def test_marginal_coverage_ratio_2(self, dummy_greedy_alg, dataset, head, tail, expected):
        """
        a rule (0, ) -> (0, ) is added
        """
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        old_rule = Rule((0, ), (0, ))
        old_rule.bind_dataset(X.tocsc(), Y.tocsc())
        alg.selected_rules = [old_rule]
        
        rule = Rule(head, tail)
        rule.bind_dataset(X.tocsc(), Y.tocsc())

        actual = alg.marginal_coverage_ratio(rule)
        np.testing.assert_allclose(actual, expected)


    def test_generate_candidate_rules(self, dummy_greedy_alg, dataset):
        # number of candidates should match
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        # remember to fit the tail sampler first
        alg.tail_sampler.fit(Y)

        alg.n_heads_per_tail = 5
        alg.n_tails_per_iter = 5
        
        rules = alg.generate_candidate_rules()
        assert len(rules) <= alg.n_heads_per_tail * alg.n_tails_per_iter
        for r in rules:
            assert isinstance(r, Rule)

    def test_add_rule(self, dataset, dummy_greedy_alg):
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        # remember to fit the tail sampler first
        alg.tail_sampler.fit(Y)
        alg.n_heads_per_tail = 1
        alg.n_tails_per_iter = 1

        assert alg.num_rules_selected == 0

        rule = list(alg.generate_candidate_rules())[0]
        alg.add_rule(rule)

        assert alg.num_rules_selected == 1

    def test_objective(self, dataset, dummy_greedy_alg):
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.lambd = 100.

        alg.set_X_Y(X, Y)

        rules = [Rule((0, ), (0, )), Rule((1, ), (1, )), Rule((2, ), (1, ))]
        r0, r1, r2 = rules
        for r in rules:
            r.bind_dataset(X.tocsc(), Y.tocsc())

        alg.selected_rules = rules
        actual = alg.objective(rules)
        expected = (
            sum([alg.quality_of_rule(r) for r in rules])
            + alg.lambd * (r0.distance(r1) + r1.distance(r2) + r0.distance(r2))
        )

        np.testing.assert_allclose(actual, expected)
        
    def test_diversity_of_rule(self, dummy_greedy_alg, dataset):
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        alg.tail_sampler.fit(Y)
        rules = list(alg.generate_candidate_rules())
        rules_to_add = rules[:2]

        another_rule = rules[2]

        for r in rules_to_add:
            alg.add_rule(r)

        actual = alg.diversity_of_rule(another_rule)
        expected = sum(another_rule.distance(r) for r in rules_to_add)
        np.testing.assert_allclose(expected, actual)

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

        alg.tolerance = 0.0
  
        # when n_max_rules is set
        alg.selected_rules = [None] * 5
        alg.n_max_rules = 5
        assert alg.should_stop()

        alg.selected_rules = [None] * 4
        alg.n_max_rules = 5
        assert not alg.should_stop()

        # when n_max_rules is not set
        # first reset some attrs
        alg.selected_rules = []
        alg.n_max_rules = None

        r1 = Rule((0, ), (0, ))  # coverage is 0.5
        r1.bind_dataset(X.tocsc(), Y.tocsc())
        alg.tolerance = 0.49
        assert not alg.should_stop(r1)

        alg.tolerance = 0.51
        assert alg.should_stop(r1)

        
    def test_fit_1st_round_one_rule_only(self, dummy_greedy_alg, dataset):
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

    def test_fit_1st_round_until_fully_covered(self, dummy_greedy_alg, dataset):
        """we fit until all label occurence is covered
        """
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.n_max_rules = None
        
        learned_rules = alg.fit_1st_round(X, Y)
        
        assert alg.num_rules_selected > 1

        assert learned_rules == alg.selected_rules
        for rule in alg.selected_rules:
            assert isinstance(rule, Rule)

        assert alg.coverage_ratio() == 1.0  # all label occurence is covered

    def test_fit_2nd_round(self, dummy_greedy_alg, dataset):
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        r0, r1 = Rule((0, ), (0, )), Rule((1, ), (1, ))
        for r in (r0, r1):
            r.bind_dataset(X.tocsc(), Y.tocsc())

        with pytest.raises(RuntimeError):
            alg.fit_2nd_round()
            
        # candidate_rules are not populated yet
        alg.candidate_rules = []
        with pytest.raises(InsufficientCandidates):
            alg.fit_2nd_round()

        alg.candidate_rules = [r0]
        alg.selected_rules = [r1]

        learned_rules = alg.fit_2nd_round()
        assert alg.selected_rules_round_1 == [r1]
        assert alg.selected_rules == [r0] == learned_rules

        # now we select 2 duplicate rules
        alg.candidate_rules = [r0, r0]
        alg.selected_rules = [r1, r1]
        learned_rules = alg.fit_2nd_round()
        assert alg.selected_rules_round_1 == [r1, r1]
        assert alg.selected_rules == [r0, r0] == learned_rules

    def test_fit_one_rule_only(self, dummy_greedy_alg, dataset):
        """just learn on rule
        """
        X, Y = dataset
        n_max_rules = 1
        alg = dummy_greedy_alg
        alg.n_max_rules = n_max_rules
        
        alg.fit(X, Y)

        assert hasattr(alg, 'obj_round_1')
        assert hasattr(alg, 'obj_round_2')
        assert hasattr(alg, 'obj')
        assert alg.obj == min(alg.obj_round_1, alg.obj_round_2)
        
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
        
        alg.fit(X, Y)
        
        assert alg.num_rules_selected > 1

        assert hasattr(alg, 'obj_round_1')
        assert hasattr(alg, 'obj_round_2')
        assert hasattr(alg, 'obj')
        assert alg.obj == min(alg.obj_round_1, alg.obj_round_2)
        
        for rule in alg.selected_rules:
            assert isinstance(rule, Rule)

        assert alg.coverage_ratio() == 1.0  # all label occurence is covered
        
    @pytest.mark.parametrize('rules, predicted_sets', [
        [[], [set(), set(), set()]],  # no rules at all
        [[Rule((0, ), (0, ))], [{0}, set(), {0}]],  # one rule
        [[Rule((0, ), (0, )), Rule((1, ), (1, ))], [{0, 1}, {1}, {0}]],  # optimal rules 
    ])
    def test_predict_no_rules_at_all(self, dummy_greedy_alg, dataset, rules, predicted_sets):
        X, Y = dataset
        alg = dummy_greedy_alg
        alg.set_X_Y(X, Y)
        alg.selected_rules = rules
        for r in alg.selected_rules:
            r.bind_dataset(X.tocsc(), Y.tocsc())
            
        actual = alg.predict(X)

        pred_mat = convert_sets_to_matrix(predicted_sets, Y.shape[1])
        assert csr_matrix_equal(pred_mat, actual)

    @pytest.mark.parametrize('min_edge_proba',  # this parameter controls how large the sample space is
                             [0.5, 0.8]
                             )
    def test_runnable_for_nontrivial_samplers(self, dataset, min_edge_proba):
        X, Y = dataset
        head_sampler = CFTPDiscriminativitySampler(min_proba=min_edge_proba)
        tail_sampler = UncoveredAreaSampler(min_proba=min_edge_proba)
        alg = GreedyDivMax(head_sampler, tail_sampler, n_tails_per_iter=3)
        alg.n_max_rules = None
        alg.fit(X, Y)
        assert alg.coverage_ratio() == 1.0  # all label occ should be covered    
        assert alg.num_rules_selected > 1


class TestGreedyCFTPDivMax():        
    def test_get_params(self):
        alg = GreedyCFTPDivMax(min_feature_proba=0.7, min_label_proba=0.5,
                               lambd=0.7,
                               n_tails_per_iter=1000,
                               n_heads_per_tail=99,
                               n_max_rules=100,
                               tolerance=1e-5)
        actual = alg.get_params()
        expected = {}
        expected['lambd'] = 0.7
        expected['n_tails_per_iter'] = 1000
        expected['n_heads_per_tail'] = 99
        expected['n_max_rules'] = 100
        expected['tolerance'] = 1e-5
        expected['min_feature_proba'] = 0.7
        expected['min_label_proba'] = 0.5
        assert actual == expected

    def test_runnable(self, dataset):
        X, Y = dataset
        alg = GreedyCFTPDivMax(n_max_rules=3)
        alg.fit(X, Y)
        assert alg.coverage_ratio() == 1.0  # all label occ should be covered    
        assert alg.num_rules_selected > 1        
