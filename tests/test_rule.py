import pytest
import pickle as pkl
import numpy as np
from scipy import sparse as sp
from corset.rule import Rule, jaccard_distance
from scipy.stats import entropy


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


def get_rule(dataset, head, tail):
    X, Y = dataset
    r = Rule(head=head, tail=tail)

    # rule.bind_dataset requires X and Y to be csc_matrix
    if not isinstance(X, sp.csc_matrix):
        X = X.tocsc()

    if not isinstance(Y, sp.csc_matrix):
        Y = Y.tocsc()

    r.bind_dataset(X, Y)
    return r

RULE1 = ((0, ), (0, ))
RULE2 = ((2, ), (1, ))
RULE3 = ((1, ), (0,))

class TestJaccard:
    @pytest.mark.parametrize("r1, r2, d", [
        (RULE1, RULE2, 1.),
        (RULE2, RULE1, 1.),
        (RULE2, RULE3, 1.),
        (RULE1, RULE3, 0.5),
        (RULE1, RULE1, 0.)
    ])
    def test_jaccard_distance_1(self, dataset, r1, r2, d):
        rule1, rule2 = get_rule(dataset, *r1), get_rule(dataset, *r2)
        actual = jaccard_distance(rule1, rule2)
        expected = d
        np.testing.assert_allclose(actual, expected)

        actual1 = rule1.distance(rule2)
        np.testing.assert_allclose(actual1, expected)


    def test_jaccard_distance_2(self):
        X = sp.csr_matrix(np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
        ]))
        Y = sp.csr_matrix(np.ones((4, 4)))
        r1 = Rule(head=(0, 1, 2), tail=(0, 1, 2))
        r2 = Rule(head=(1, 2, 3), tail=(1, 2, 3))
        X, Y = X.tocsc(), Y.tocsc()
        r1.bind_dataset(X, Y)
        r2.bind_dataset(X, Y)
        actual = jaccard_distance(r1, r2)
        expected = 1 - 4 / 14
        np.testing.assert_allclose(actual, expected)
        
class TestRule:
    @pytest.mark.parametrize(
        "tail, label_vector, support",
        [
            [(0,), [1, 0, 1], {0, 2}],
            [(0, 1), [1, 0, 0], {0}],
            [(1,), [1, 1, 0], {0, 1}],
        ],
    )
    def test_tail_support(self, dataset, tail, label_vector, support):
        r = get_rule(dataset, (0, ), tail)

        np.testing.assert_allclose(r.tail_label_vector, label_vector)
        assert r.tail_support == support
        assert r.tail_support_size == len(support)

    @pytest.mark.parametrize(
        "head, label_vector, support",
        [
            [(0,), [1, 0, 1], {0, 2}],
            [(1,), [1, 1, 0], {0, 1}],
            [(1, 2), [0, 1, 0], {1}],
            [(0, 2), [0, 0, 1], {2}],
        ],
    )
    def test_head_support(self, dataset, head, label_vector, support):
        r = get_rule(dataset, head, (0, ))

        assert r.head_support == support
        assert r.head_support_size == len(support)
        np.testing.assert_allclose(r.head_label_vector, label_vector)

    @pytest.mark.parametrize("head", [(0, 1, 2), (-1, 0), (0, 5)])
    def test_invalid_head(self, dataset, head):
        X, Y = dataset
        r = Rule(head=head, tail=(0,))
        with pytest.raises(ValueError):
            r.bind_dataset(X.tocsc(), Y.tocsc())

    @pytest.mark.parametrize(
        "head, tail, support",
        [
            ((0,), (0,), {0, 2}),
            ((2,), (1,), {1}),
            ((1,), (0, ), {0})
        ],
    )
    def test_support(self, dataset, head, tail, support):
        r = get_rule(dataset, head, tail)
        
        assert r.support == support
        assert r.support_size == len(support)

    @pytest.mark.parametrize(
        "head, tail, tpr, br",
        [
            ((0,), (0,), 1.0, 2 / 3),
            ((2,), (1,), 1 / 2, 2 / 3),
        ],
    )
    def test_true_positive_rate_and_base_rate(self, dataset, head, tail, tpr, br):
        r = get_rule(dataset, head, tail)

        actual = r.true_positive_rate()
        assert actual == tpr

        actual = r.base_rate()
        assert actual == br

    @pytest.mark.parametrize(
        "head, tail, tpr, br, kl",
        [
            ((0,), (0,), 1.0, 2 / 3, entropy([1, 0], [2 / 3, 1 / 3])),
            ((2,), (1,), 1 / 2, 2 / 3, 0),  # in this case, kl is zero
        ],
    )
    def test_kl(self, dataset, head, tail, tpr, br, kl):
        r = get_rule(dataset, head, tail)
        
        np.testing.assert_allclose(kl, r.KL())

    @pytest.mark.parametrize(
        "head, tail, area",
        [
            ((0,), (0,), 2),
            ((2,), (1,), 1),
            ((0, 1), (0, 1), 2)
        ],
    )
    def test_label_area(self, dataset, head, tail, area):
        r = get_rule(dataset, head, tail)
        assert r.label_area() == area
            
    def test_summary(self, dataset):
        r = get_rule(dataset, (0, ), (0, ))
        assert isinstance(r.summary(), str)

    @pytest.mark.parametrize('head, tail, expected', [
        ((0, ), (0, ), 2),
        ((0, 1), (0, ), 1)
    ])        
    def test_marginal_coverage_no_ref(self, dataset, head, tail, expected):
        rule = get_rule(dataset, head, tail)
        actual = rule.marginal_coverage([])
        assert actual == expected


    @pytest.mark.parametrize('head, tail, expected', [
        ((0, ), (0, ), 0.),
        ((0, 1), (0, ), 0),
        ((0, 1), (1, ), 1),
    ])        
    def test_marginal_coverage_with_ref(self, dataset, head, tail, expected):
        rule = get_rule(dataset, head, tail)
        ref_rule = get_rule(dataset, (0, ), (0, ))
        actual = rule.marginal_coverage([ref_rule])
        assert actual == expected


    @pytest.mark.parametrize('head, tail, expected', [
        ((0, ), (0, ), 0.),
        ((0, 1), (0, ), 0),
        ((0, 1), (1, ), 1),
    ])        
    def test_marginal_coverage_with_repeated_refs(self, dataset, head, tail, expected):
        rule = get_rule(dataset, head, tail)
        ref_rule = get_rule(dataset, (0, ), (0, ))
        actual = rule.marginal_coverage([ref_rule] * 100)  # repeat the ref many times
        assert actual == expected
        
    @pytest.mark.parametrize('r1, r2, eq', [
        (Rule((0, ), (0, )), Rule((0, ), (0, )), True),
        (Rule((0, ), (0, )), Rule((0, ), (1, )), False),
        (Rule((1, ), (0, )), Rule((0, ), (0, )), False),
    ])
    def test_equality(self, r1, r2, eq):
        if eq:
            assert r1 == r2
        else:
            assert r1 != r2


    def test_rule_set(self):
        rule_set = {Rule((0, ), (0, )), Rule((0, ), (0, ))}
        assert len(rule_set) == 1

    @pytest.mark.parametrize("rule", [RULE1, RULE2, RULE3])
    def test_pickle(self, dataset, rule):
        rule = get_rule(dataset, *rule)
        rule_string = pkl.dumps(rule)

        rule_cp = pkl.loads(rule_string)

        assert rule == rule_cp

        present_attrs = ['head', 'tail']
        for attr in present_attrs:
            assert hasattr(rule_cp, attr)
            
        absent_attrs = ['N', 'D', 'L', 'X', 'Y']
        for attr in absent_attrs:
            assert not hasattr(rule_cp, attr)
        
