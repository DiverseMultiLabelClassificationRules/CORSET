from scipy import sparse as sp
from sklearn.base import BaseEstimator
from ..samplers.base import Sampler
from ..rule import Rule
from ..utils import conjunctive_collapse_v2, convert_matrix_to_sets_v2

from itertools import product
from mlflow.pyfunc import PythonModel

class RuleSampler:
    def __init__(self, head_sampler: Sampler, tail_sampler: Sampler):
        """
        a sampler which generates rule candidates
        """
        kwargs = locals()
        del kwargs['self']
        self.__dict__.update(**kwargs)

    # @profile
    def generate_candidate_rules(self):
        """generate a list of candidate rules by calling the head_sampler and tail_sampler"""
        rules = set()
        for _ in range(self.n_tails_per_iter):
            tail = self.tail_sampler.sample_once()
            y = conjunctive_collapse_v2(self.Y_csc, tuple(tail))

            self.head_sampler.fit(self.X, y)

            heads = self.head_sampler.sample(self.n_heads_per_tail)

            for head in heads:
                rule = Rule(head, tail)
                if rule not in rules:
                    rule.bind_dataset(self.X_csc, self.Y_csc)

                    rules.add(rule)
        return rules


class GreedyBase(BaseEstimator, PythonModel):
    def __init__(self, random_state=None):
        """
        base class for a set of greedy algorithms

        it implements a few commonly used functions
        """
        kwargs = locals()
        del kwargs['self']
        self.__dict__.update(**kwargs)

        self.selected_rules = []
        self._num_labels = None
        self._num_features = None

    @property
    def total_num_label_occurences(self):
        return self.Y.nnz

    @property
    def num_rules_selected(self):
        return len(self.selected_rules)

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def num_features(self):
        return self._num_features        

    def coverage_ratio(self):
        """return the fraction of covered label occurence by the current set of rules

        TODO: can we make it faster?
        """
        Y_copy = self.Y.copy()
        for rule in self.selected_rules:
            row_ids = list(rule.support)
            col_ids = list(rule.tail)
            for rid, cid in product(row_ids, col_ids):
                Y_copy[rid, cid] = 0  # mark those are covered as zero

        cov = 1 - Y_copy.sum() / self.total_num_label_occurences
        return cov
        
    def marginal_coverage_ratio(self, rule):
        """the fraction of newly covered label occurence by the given rule, 
        with respect to the set of selected rules"""
        # get the total intersected label areas with selected rules
        uncovered_label_area = rule.marginal_coverage(self.selected_rules)

        if uncovered_label_area < 0:
            raise Exception('something terrible goes wrong: `uncovered_label_area` cannot be negative')

        return uncovered_label_area / self.total_num_label_occurences

    def objective_of_rule(self, rule):
        raise NotImplementedError('you should implement the selection objective')

    def select_best_candidate(self, candidate_rules):
        return max(candidate_rules, key=lambda r: self.objective_of_rule(r))

    def set_X_Y(self, X: sp.csr_matrix, Y: sp.csr_matrix):
        """set the training dataset
        """
        self.X, self.Y = X, Y

        # cache csc_matrix for better performance when calling rule.bind_dataset
        self.X_csc, self.Y_csc = X.tocsc(), Y.tocsc()

        # copy data shapes
        self._num_labels = Y.shape[1]
        self._num_features = X.shape[1]

    def predict(self, X: sp.csr_matrix):
        """
        make preditions given the input matrix

        output: a 2D sparse matrix
        """
        ret = sp.lil_matrix((X.shape[0], self.num_labels), dtype=bool)
        list_of_sets = convert_matrix_to_sets_v2(X)
        for i, features in enumerate(list_of_sets):
            # scan through each rule
            # TODO: faster version by reframing as a subset query containment problem
            for rule in self.selected_rules:
                if set(rule.head).issubset(features):  # the head is contained in the input features
                    for pred_label in rule.tail:
                        ret[i, pred_label] = 1
        return ret.tocsr()

    def fit(self, X, Y):
        self.set_X_Y(X, Y)


    def __setstate__(self, data):
        params = data['params']
        rules = data['rules']
        num_labels = data['num_labels']
        num_features = data['num_features']
        print("params: ", params)
        self.__dict__.update(params)
        # self.set_params(**params)  # does not work
        self.selected_rules = rules
        self._num_labels = num_labels
        self._num_features = num_features

    def __getstate__(self):
        ret = {}
        ret['params'] = self.get_params()
        ret['rules'] = self.selected_rules
        ret['num_labels'] = self._num_labels
        ret['num_features'] = self._num_features
        return ret
