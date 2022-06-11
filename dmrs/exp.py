import mlflow
import pandas as pd
from .data import Dataset
from .evaluation import Evaluator
from .greedy import GreedyDivMax, GreedyCFTPDivMaxV2, GreedyCFTPDivMaxV3
from .samplers.discrim import CFTPDiscriminativitySampler
from .samplers.uncovered_area import UncoveredAreaSampler
from .utils import get_experiment_id_by_name
from logzero import logger


class Default:
    """default values for hyperparameters"""

    min_feature_proba = 0.8
    min_label_proba = 0.8
    lambd = 10.0
    n_max_rules = None
    n_tails_per_iter = 100
    n_heads_per_tail = 10
    tolerance = 1e-3


class V2Default(Default):
    """default values for hyperparameters for greedy V2"""
    n_max_rules = 10


class V3Default(V2Default):
    """default values for hyperparameters for greedy V3"""


def train_greedy_and_eval(ds: Dataset, config: dict, greedy_version: str, validation_mode=False):
    """config: best configuration to initialize the greedy learner
    greedy_version: which version of greedy to use
    if validation_mode is True, use the validation set for evaluation
    """
    logger.info("validation mode is {}".format("ON" if validation_mode else "OFF"))

    ds.load()
    clf = create_greedy_clf_from_config_by_version(config, greedy_version)
    clf.fit(ds.trn_X, ds.trn_Y)

    if validation_mode:
        target_X, target_Y = ds.dev_X, ds.dev_Y
    else:
        target_X, target_Y = ds.tst_X, ds.tst_Y
    pred = clf.predict(target_X)
    evaluator = Evaluator()
    perf = evaluator.report(pred, target_Y)
    
    perf["coverage"] = clf.coverage_ratio()
    perf["n_rules"] = clf.num_rules_selected

    return clf, perf

def train_and_eval(ds: Dataset, clf):
    """clf: a classifier with sklearn-like interface,
    e.g., fit and predict
    """
    ds.load()
    clf.fit(ds.trn_X, ds.trn_Y)

    pred = clf.predict(ds.tst_X)
    evaluator = Evaluator()
    perf = evaluator.report(pred, ds.tst_Y)
    perf["coverage"] = clf.coverage_ratio()
    perf["n_rules"] = clf.num_rules_selected
    return perf


def create_greedy_from_config(config: dict):
    """config: a dictionary mapping hyperparameter name to it value

    if the name is not given, its default value is used

    the greedy uses:

    - sampling under the reduced space
    - CFTP with discriminativity for head sampling
    - uncovered area for tail sampling
    """
    logger.warn("GreedyDivMax is deprecated, use GreedyDivMaxV2 or GreedyCFTPDivMaxV2 instead!")
    min_feature_proba = config.get("min_feature_proba", Default.min_feature_proba)
    min_label_proba = config.get("min_label_proba", Default.min_label_proba)
    lambd = config.get("lambd", Default.lambd)
    n_tails_per_iter = config.get("n_tails_per_iter", Default.n_tails_per_iter)
    n_heads_per_tail = config.get("n_heads_per_tail", Default.n_heads_per_tail)
    tolerance = config.get("tolerance", Default.tolerance)
    n_max_rules = config.get('n_max_rules', Default.n_max_rules)

    head_sampler = CFTPDiscriminativitySampler(min_proba=min_feature_proba)
    tail_sampler = UncoveredAreaSampler(min_proba=min_label_proba)
    alg = GreedyDivMax(
        head_sampler,
        tail_sampler,
        n_tails_per_iter=n_tails_per_iter,
        n_heads_per_tail=n_heads_per_tail,
        lambd=lambd,
        tolerance=tolerance,
        n_max_rules=n_max_rules
    )

    return alg

def create_greedy_v2_from_config(config: dict):
    min_feature_proba = config.get("min_feature_proba", V2Default.min_feature_proba)
    min_label_proba = config.get("min_label_proba", V2Default.min_label_proba)
    lambd = config.get("lambd", V2Default.lambd)
    n_tails_per_iter = config.get("n_tails_per_iter", V2Default.n_tails_per_iter)
    n_heads_per_tail = config.get("n_heads_per_tail", V2Default.n_heads_per_tail)
    n_max_rules = config.get('n_max_rules', V2Default.n_max_rules)

    alg = GreedyCFTPDivMaxV2(
        min_feature_proba=min_feature_proba,
        min_label_proba=min_label_proba,
        n_tails_per_iter=n_tails_per_iter,
        n_heads_per_tail=n_heads_per_tail,
        lambd=lambd,
        n_max_rules=n_max_rules
    )

    return alg


def create_greedy_v3_from_config(config: dict):
    min_feature_proba = config.get("min_feature_proba", V3Default.min_feature_proba)
    min_label_proba = config.get("min_label_proba", V3Default.min_label_proba)
    lambd = config.get("lambd", V3Default.lambd)
    n_tails_per_iter = config.get("n_tails_per_iter", V3Default.n_tails_per_iter)
    n_heads_per_tail = config.get("n_heads_per_tail", V3Default.n_heads_per_tail)
    n_max_rules = config.get('n_max_rules', V3Default.n_max_rules)

    alg = GreedyCFTPDivMaxV3(
        min_feature_proba=min_feature_proba,
        min_label_proba=min_label_proba,
        n_tails_per_iter=n_tails_per_iter,
        n_heads_per_tail=n_heads_per_tail,
        lambd=lambd,
        n_max_rules=n_max_rules
    )

    return alg


def create_greedy_clf_from_config_by_version(config, greedy_version):
    assert greedy_version in ('v2', 'v3'), f'{greedy_version} is not supported'
    logger.info("greedy version: {}".format(greedy_version))
    if greedy_version == 'v2':
        clf = create_greedy_v2_from_config(config)
    elif greedy_version == 'v3':
        clf = create_greedy_v3_from_config(config)
    return clf
