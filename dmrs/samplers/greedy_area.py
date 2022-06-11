import random
import numpy as np
from collections import defaultdict, Counter
from .base import Sampler


def compute_frequencies(X, Y):
    """X: feature matrix
    Y: label matrix"""
    d = defaultdict(lambda: defaultdict(int))
    d_label = defaultdict(lambda: defaultdict(int))
    d_attributes = defaultdict(lambda: defaultdict(int))

    all_labels = []
    for i in range(len(X)):
        example_attributes_bag = set(
            [j for j in range(X.shape[1]) if X[i][j] == 1])
        labels_bag = set([l for l in range(Y.shape[1]) if Y[i][l] == 1])
        list_label_bag = list(labels_bag)
        list_attributes_bag = list(example_attributes_bag)

        for i in range(len(list_label_bag)):
            for j in range(i + 1, len(list_label_bag)):

                lab_i = list_label_bag[i]
                lab_j = list_label_bag[j]
                d_label[lab_i][lab_j] += 1
                d_label[lab_j][lab_i] += 1

        for i in range(len(list_attributes_bag)):
            for j in range(i + 1, len(list_attributes_bag)):

                attr_i = list_attributes_bag[i]
                attr_j = list_attributes_bag[j]
                d_attributes[attr_i][attr_j] += 1
                d_attributes[attr_j][attr_i] += 1

        all_labels.extend(
            list_label_bag
        )  # we keep frequencies of labels to set label-dependent epsiolo threshold
        for label in labels_bag:
            for attr in example_attributes_bag:
                d[label][attr] += 1

    d_formatted = dict()
    # frequency are now computed. change format.
    for label in d.keys():
        key_list = list(d[label].keys())
        value_list = list([x for x in d[label].values()])
        list_key_values = sorted(list(zip(key_list, value_list)),
                                 key=lambda x: x[1])
        d_formatted[label] = list_key_values
    return d, d_formatted, Counter(all_labels), d_label, d_attributes


class GreedyAreaSampler(Sampler):
    """greedy tail sampler"""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, Y):
        d, d_formatted, label_frequencies, d_label, d_attributes = compute_frequencies(
            X, Y)
        self.__dict__.update(
            dict(
                X=X,
                Y=Y,
                d=d,
                d_formatted=d_formatted,
                label_frequencies=label_frequencies,
                d_label=d_label,
                d_attributes=d_attributes,
            ))
        self.compute_weights()

    def compute_weights(self):
        """
        compute weights for all data records

        returns
        W: array of weights for each data record
        """
        # array of weight
        self.W_tail = np.zeros(self.Y.shape[0])

        # compute weights
        for i in range(self.Y.shape[0]):
            bar_covDR_size = len([
                self.Y[i][h] for h in range(len(self.Y[i]))
                if self.Y[i][h] == 1
            ])
            self.W_tail[i] = bar_covDR_size * 2**(
                float(sum(self.Y[i]) - 1)
            )  # sum is equal to |D| with 0/1 representation

    def sample_once(self, seed=None):
        """
        Sample one tail according to area

        returns
        T: set of sampled labels and set of positive examples
        """
        random.seed(seed)

        sampled_index = random.choices([i for i in range(self.Y.shape[0])],
                                       weights=self.W_tail,
                                       k=1)[0]

        sampled_T = self.Y[sampled_index]
        # draw T1 - 0-indexed attributes
        bar_covDR = [h for h in range(len(sampled_T)) if sampled_T[h] == 1]
        if len(bar_covDR) == 0:
            return set(), set()
        sampled_size = random.choices(
            [i for i in range(1,
                              len(bar_covDR) + 1)],
            weights=[i for i in range(len(bar_covDR))],
            k=1,
        )[0]  # we do not want to sample singletons
        # for the moment let us use sampled size in this way but in the future it would be
        # more appropriate maybe to stop when the target drops more than a fixed threshold ?
        T = set()
        # take first at random
        first_element = random.choice(bar_covDR)
        T.add(first_element)

        cur_bar_covDR = set(bar_covDR)
        cur_bar_covDR.remove(first_element)

        while len(T) < sampled_size:
            # greedy choice
            min_co_occur = float("-inf")
            for element in cur_bar_covDR:
                this_co_occurences = []
                for element_solution in T:
                    this_co_occurences.append(
                        self.d_label[element][element_solution])
                this_min_co_occur = min(this_co_occurences)

                if this_min_co_occur > min_co_occur:
                    min_co_occur = this_min_co_occur
                    chosen_element = element
            cur_bar_covDR.remove(chosen_element)

            T.add(chosen_element)

        return T
