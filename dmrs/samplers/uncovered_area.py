import numpy as np
from scipy import sparse as sp
from collections import Counter
from .assignment import TrieSampleAssignmentMixin, PRETTISampleAssignmentMixin
from .pc import PCSampleSpaceConstructor
from .surs import ReducedSpaceSampler
from ..exceptions import NoMoreSamples

class UncoveredAreaMixin:
    def __init__(self):
        self.new_rows = []
        self._sample_space = []

        self.update_history = []

    def _assign_row_weights(self):
        """assign weights to each row"""
        self.row_weights = np.zeros(len(self.new_rows), dtype=float)
        for i, row in enumerate(self.new_rows):
            self.row_weights[i] = sum(
                [len(self._sample_space[sample_id]) for sample_id in row]
            )
        self._update_row_probas()
        self._build_element_counter()

    def _build_element_counter(self):
        """
        for each data record, build a counter, mapping
        each element in the DR
        to the number of times the element occurs in the associated samples
        """
        self.element_counter_per_row = []
        for i, row in enumerate(self.new_rows):
            bag_of_elems = [
                el for sample_id in row for el in self._sample_space[sample_id]
            ]
            cnt = Counter(bag_of_elems)
            self.element_counter_per_row.append(cnt)

    def update_row_weights_naive(self, affected_row_ids, new_set):
        """naive way of updating row_weights

        affected_row_ids: list of row indices whose weights are to do updated e.g., the support of a rule
        new_set: a set of newly added elements (e.g., the tail)
        """
        print("please use `update_row_weights`, which is faster")
        for row_id in affected_row_ids:
            for sample_id in self.new_rows[row_id]:
                for el in new_set:
                    if el in self._sample_space[sample_id]:
                        self.row_weights[row_id] -= 1
        self._update_row_probas()

    def update_row_weights(self, affected_row_ids, new_set):
        """faster version of update_row_weights_naive"""
        if not hasattr(self, "element_counter_per_row"):
            raise AttributeError(
                "`element_counter_per_row` is not set, have you called `_build_element_counter`?"
            )
        for row_id in affected_row_ids:
            for el in new_set:
                if el in self.element_counter_per_row[row_id]:  # if not covered before
                    self.row_weights[row_id] -= self.element_counter_per_row[row_id][el]
                    del self.element_counter_per_row[row_id][el]
        self._update_row_probas()
        self.update_history.append((affected_row_ids, new_set))

    def _update_row_probas(self):
        """normalize the row weights as a probability distribution"""
        if self.row_weights.sum() == 0:
            self.row_probas = np.ones(self.row_weights.shape) / np.prod(
                self.row_weights.shape
            )
        else:
            self.row_probas = self.row_weights / self.row_weights.sum()

    def _sample_itemset_from_row(self, row_id):
        """given a row index, sample a pattern from that row, taking the covered elements into account"""
        sample_ids = self.new_rows[row_id]
        # weight for each sample of the current DR
        sample_weights = np.array(
            [len(self._sample_space[sample_id]) for sample_id in sample_ids],
            dtype=float,
        )

        # iterate each update to get the set of covered elements by the update
        covered_elems = set()
        for affected_row_ids, new_set in self.update_history:
            if row_id in affected_row_ids:  # if this row is affected by the update
                covered_elems |= new_set
                
        # decrease the weight of each sample by its intersection size with the covered set
        for idx, sample_id in enumerate(sample_ids):
            sample_weights[idx] -= len(
                self._sample_space[sample_id].intersection(covered_elems)
            )

        if not np.allclose(sample_weights.sum(), self.row_weights[row_id]):
            print("row_id: ", row_id)
            print('row samples', [self._sample_space[sample_id] for sample_id in sample_ids])
            print('self.update_history', self.update_history)
            raise Exception(
                "sample_weights sum does not equal the DR's weight: {} != {}".format(
                    sample_weights.sum(), self.row_weights[row_id]
                )
            )

        if sample_weights.sum() == 0:
            raise NoMoreSamples(
                f"row {row_id} has weight zero, sampling from it is ill-defined"
            )

        random_sample_id = np.random.choice(
            sample_ids, p=sample_weights / sample_weights.sum()
        )
        return self._sample_space[random_sample_id]


class UncoveredAreaSampler(
        # TrieSampleAssignmentMixin,
        PRETTISampleAssignmentMixin,
        PCSampleSpaceConstructor,
        UncoveredAreaMixin,
        ReducedSpaceSampler,
):
    """sampling according to uncovered area under SURS framework"""

    def __init__(
        self,
        random_state=12345,
        min_proba=0.05,
        do_prune_edges=True,
        dfs_backend="dfs_v2",
    ):

        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

        self.update_history = []

    def __repr__(self):
        return "dmrs.samplers.uncovered_area.UncoveredAreaSampler(min_proba={})".format(self.min_proba)
    
    def fit(self, Y: sp.csr_matrix):
        self._build_graph(Y)

        self._build_sample_space()

        self._generate_new_rows(Y)
        self._assign_row_weights()
