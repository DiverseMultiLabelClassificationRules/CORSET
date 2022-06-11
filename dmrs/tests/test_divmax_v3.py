import pytest
import numpy as np
from .test_divmax_v2 import TestGreedyDivMaxV2, dataset
from dmrs.greedy.divmax_v3 import GreedyCFTPDivMaxV3, GreedyDivMaxV3
from dmrs.rule import Rule

class TestGreedyDivMaxV3(TestGreedyDivMaxV2):
    """identical test cases for GreedyDivMaxV2,
    so we inheri TestGreedyDivMaxV2 and change the algorithm constructor here"""
    greedy_cls = GreedyDivMaxV3
    
    @pytest.fixture
    def dummy_greedy_alg(self, dummy_head_sampler, dummy_tail_sampler):
        return GreedyDivMaxV3(
            dummy_head_sampler, dummy_tail_sampler,
            n_heads_per_tail=5, n_tails_per_iter=5,
            n_max_rules=10
        )


    @pytest.mark.parametrize('rule', [
        Rule((0, ), (0, )),
        Rule((1, ), (1, )),
        Rule((2, ), (1, ))])
    def test_rule_quality(self, dataset, dummy_greedy_alg, rule):
        X, Y = dataset
        rule.bind_dataset(X.tocsc(), Y.tocsc())

        alg = dummy_greedy_alg
        actual = alg.quality_of_rule(rule)
        expected = np.sqrt(rule.marginal_coverage([])) * rule.KL()

        assert actual == expected
