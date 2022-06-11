import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from collections import OrderedDict
from itertools import permutations

from .data import Dataset
from .utils import support_size

AVAILABLE_METRICS = ('hamming_accuracy', 'subset_accuracy', 'micro_precision',
                     'micro_recall', 'micro_f1', 'macro_precision',
                     'macro_recall', 'macro_f1')


def hamming_accuracy(predictions, ground_truth):
    return 1 - metrics.hamming_loss(ground_truth, predictions)


def subset_accuracy(predictions, ground_truth):
    return metrics.accuracy_score(ground_truth, predictions)


def micro_precision(predictions, ground_truth):
    return metrics.precision_score(ground_truth,
                                   predictions,
                                   average='micro',
                                   zero_division=1)


def micro_recall(predictions, ground_truth):
    return metrics.recall_score(ground_truth,
                                predictions,
                                average='micro',
                                zero_division=1)


def micro_f1(predictions, ground_truth):
    return metrics.f1_score(ground_truth,
                            predictions,
                            average='micro',
                            zero_division=1)


def macro_precision(predictions, ground_truth):
    return metrics.precision_score(ground_truth,
                                   predictions,
                                   average='macro',
                                   zero_division=1)


def macro_recall(predictions, ground_truth):
    return metrics.recall_score(ground_truth,
                                predictions,
                                average='macro',
                                zero_division=1)


def macro_f1(predictions, ground_truth):
    return metrics.f1_score(ground_truth,
                            predictions,
                            average='macro',
                            zero_division=1)


class Evaluator:
    """
    how to use it:
    
    > pred_Y, true_Y = {predictions, ground truth}
    > ev = Evaluator()
    > ev.report(pred_Y, true_Y)
    """

    def __init__(self, metrics=AVAILABLE_METRICS):
        self.metrics = metrics
        for m in self.metrics:
            assert m in AVAILABLE_METRICS

        self.name2func = {
            'hamming_accuracy': hamming_accuracy,
            'subset_accuracy': subset_accuracy,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }

    def report(self, predictions, ground_truth):
        """
        report evaluation scores across different metrics

        predictions: a scipy sparse matrix
        ground_truth: a scipy sparse matrix
        """
        res = OrderedDict()
        for m in self.metrics:
            res[m] = self.name2func[m](predictions, ground_truth)
        return res


class TailEvaluator:

    def __init__(self, ds: Dataset):
        self._ds = ds
        self.Y_csc = self._ds.trn_Y.tocsc()

    def support_size(self, Ss: list):
        return np.array([support_size(self.Y_csc, S) for S in Ss])

    def frequency(self, Ss: list):
        return self.support_size(Ss) / self._ds.ntrn

    def area(self, Ss: list):
        return self.support_size(Ss) * np.array(list(map(len, Ss)))

    def edge_density(self, Ss: list):
        """aka average degree"""

        def aux(S: set):
            if len(S) == 0:
                return np.NAN
            else:
                return np.sum([
                    self._ds._g[u][v]['proba']
                    for u, v in permutations(S, 2)
                ]) / len(S)

        return np.array([aux(S) for S in Ss])

    def edge_ratio(self, Ss: list):
        """aka sum of edge weights / |S| / (|S|-1)"""

        def aux(S: set):
            if len(S) <= 1:
                return np.NAN
            else:
                return np.sum([
                    self._ds._g[u][v]['proba']
                    for u, v in permutations(S, 2)
                ]) / len(S) / (len(S) - 1)

        return np.array([aux(S) for S in Ss])

    def length(self, Ss: list):
        return np.array([len(S) for S in Ss])

    def report(self, Ss):
        df = pd.DataFrame.from_dict({
            'support_size': self.support_size(Ss),
            'frequency': self.frequency(Ss),
            'edge_density': self.edge_density(Ss),
            'edge_ratio': self.edge_ratio(Ss),
            'length': self.length(Ss),
            'area': self.area(Ss)
        })
        return df.describe()
