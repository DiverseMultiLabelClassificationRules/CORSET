import numpy as np
from scipy import sparse as sp
from .utils import flatten, conjunctive_collapse_v2, binary_vector_to_set
from scipy.stats import entropy
from itertools import product


def jaccard_distance(r1, r2):
    intersect_size = len(r1.support.intersection(r2.support)) * len(
        set(r1.tail).intersection(set(r2.tail))
    )
    # by inclusion-exclusion principle
    union_size = (
        r1.support_size * len(r1.tail)
        + r2.support_size * len(r2.tail)
        - intersect_size
    )
    return 1 - intersect_size / union_size


class Rule:
    def __init__(self, head, tail):
        """ """
        head = tuple(sorted(list(head)))
        tail = tuple(sorted(list(tail)))
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

    # @profile
    def bind_dataset(self, X: sp.csc_matrix, Y: sp.csc_matrix):
        """bind a dataset to this rule
        X and Y are csc_matrix for better performance 
        """
        self.X = X
        self.Y = Y
        self.N, self.D = self.X.shape
        self.L = self.Y.shape[1]

        self._check_head()
        self.set_label_vector()
        self.set_head_support()
        self.set_tail_support()
        self.set_support()

    def _check_head(self):
        for el in self.head:
            if el < 0 or el >= self.D:
                print(el)
                raise ValueError("element {} in head is in invalid range".format(el))

    def set_label_vector(self):
        """label_vector is a 
        binary vector with length self.N"""
        
        self.tail_label_vector = conjunctive_collapse_v2(self.Y, self.tail)
        self.head_label_vector = conjunctive_collapse_v2(self.X, self.head)

    def set_head_support(self):
        """support of head only"""
        self.head_support = binary_vector_to_set(self.head_label_vector)
        self.head_support_size = len(self.head_support)
        if self.head_support_size == 0:
            raise ValueError("head {} has zero support".format(self.head))

    def set_tail_support(self):
        """support of tail only"""
        self.tail_support = binary_vector_to_set(self.tail_label_vector)
        self.tail_support_size = len(self.tail_support)

    def set_support(self):
        """support of the rule"""
        mask = np.logical_and(self.head_label_vector, self.tail_label_vector)
        self.support = binary_vector_to_set(mask)
        self.support_size = len(self.support)

    def true_positive_rate(self):
        """
        true positive rate by the rule
        i.e., |D[R]| / |D[H]|
        """
        return self.support_size / self.head_support_size

    def precision(self):
        """
        TP / (TP + FP), equivalently, |D[R]| / |D[H]|
        """
        return self.support_size / self.head_support_size

    def base_rate(self):
        """
        base rate within the dataset
        i.e., |D[T]| / |D|
        """
        return self.tail_support_size / self.N

    def KL(self):
        """
        KL divergence betwen TPR and base rate
        """
        p, q = self.true_positive_rate(), self.base_rate()
        # print("p, q: ", p, q)
        if p < q:
            return 0
        return entropy([p, 1 - p], [q, 1 - q])
    
    def distance(self, other):
        return jaccard_distance(self, other)

    def label_area(self):
        return self.support_size * len(self.tail)

    def marginal_coverage(self, ref_rules):
        """can be expensive to compute! 

        can it be faster?
        """
        def extract_label_occs(rule):
            return set(map(tuple, product(rule.support, rule.tail)))
        current_label_occurences = extract_label_occs(self)
        
        for ref_rule in ref_rules:
            current_label_occurences -= extract_label_occs(ref_rule)

        uncovered_label_area = len(current_label_occurences)

        if uncovered_label_area < 0:
            raise Exception('something terrible goes wrong: `uncovered_label_area={}` cannot be negative'.format(
                uncovered_label_area
            ))
        
        return uncovered_label_area
    
    def summary(self):
        return "Rule(|H|={}, |L|={}, |supp|={}, area={}, p={:.2f}, q={:.2f})".format(
            len(self.head), len(self.tail), self.support_size, self.label_area(), self.true_positive_rate(), self.base_rate()
        )

    def __eq__(self, other):
        return (self.head == other.head) and (self.tail == other.tail)

    def __hash__(self):
        return hash((self.head, self.tail))

    def __setstate__(self, data):
        self.head = data['head']
        self.tail = data['tail']

    def __getstate__(self):
        return {'head': self.head, 'tail': self.tail}
