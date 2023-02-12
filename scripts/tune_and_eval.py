#!/usr/bin/env python
# coding: utf-8

import sys
import logzero
import ray
import pathlib
import pandas as pd
from time import time
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import TimeoutStopper
from sklearn.model_selection import train_test_split
from logzero import logger


from corset.utils import pjoin, makedir, save_json
from corset.data import Dataset
from corset.evaluation import Evaluator
from corset.exp import create_greedy_v2_from_config, train_greedy_and_eval

ROOT_DIR = "/scratch/cs/mldb/han/interpretable-mlc-code"

N_REPEATS = 10


def one_run(config):
    logzero.loglevel(logzero.INFO)

    ds_name = config["dataset"]

    ds = Dataset(name=ds_name, datadir=pjoin(ROOT_DIR, "data"))
    ds.load()

    trn_X, dev_X, trn_Y, dev_Y = train_test_split(
        ds.trn_X, ds.trn_Y, train_size=0.7, random_state=1234
    )

    perf_list = []
    # repeat multiple and report and average
    for i in range(N_REPEATS):
        logger.info("within-trial round: {}".format(i))
        alg = create_greedy_v2_from_config(config)
        s = time()
        alg.fit(ds.trn_X, ds.trn_Y)
        elasped_time = time() - s
        pred = alg.predict(dev_X)
        evaluator = Evaluator(metrics=["micro_f1", "macro_f1", "hamming_accuracy"])
        perf = evaluator.report(pred, dev_Y)
        perf["n_rules"] = alg.num_rules_selected
        perf["coverage"] = alg.coverage_ratio()
        perf["fit_time"] = elasped_time

        perf_list.append(perf)

    df = pd.DataFrame.from_records(perf_list)
    mean_perf = df.mean().to_dict()
    tune.report(**mean_perf)


def main():
    dataset = sys.argv[1]

    num_cpus = 8
    num_samples = 32
    max_hours = 23.5
    metric_for_ranking = "micro_f1"
    mode = "max"
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    # configs per dataset
    if dataset == "toy":
        search_space = {
            "lambd": tune.loguniform(1e0, 1e2),
            "n_max_rules": tune.randint(2, 5),
        }
        num_samples = 2
    elif dataset == "emotions":
        num_samples = 64
        search_space = {
            "lambd": tune.loguniform(1e0, 1e3),
            "n_max_rules": tune.randint(5, 50),
        }
    elif dataset == "mediamill":
        num_samples = 16
        search_space = {
            "min_feature_proba": 0.8,
            'min_label_proba': 0.5,
            "lambd": tune.loguniform(1e0, 1e3),
            "n_max_rules": tune.randint(20, 100),
        }
    elif dataset == "birds":
        search_space = {
            "min_feature_proba": 0.95,
            'min_label_proba': 0.5,
            "lambd": tune.loguniform(1e0, 1e3),
            "n_max_rules": tune.randint(5, 50),
        }
    elif dataset == "cal500":
        raise NotImplementedError('the feature graph is too dense!')
        search_space = {
            "lambd": tune.loguniform(1e0, 1e3),
            "n_max_rules": tune.randint(5, 50),
        }                
    elif dataset == "enron":
        search_space = {
            "min_feature_proba": 1.0,
            "lambd": tune.loguniform(1e0, 1e2),
            "n_max_rules": tune.randint(10, 50),
        }
    elif dataset == "bibtex":
        search_space = {
            "min_feature_proba": 0.95,
            "min_label_proba": 0.5,
            "lambd": tune.loguniform(1e0, 1e3),
            "n_max_rules": tune.randint(50, 150),
        }
    elif dataset == "medical":
        search_space = {
            "lambd": tune.loguniform(1e0, 1e3),
            "n_max_rules": tune.randint(10, 50),
        }

    logger.info("dataset: {}".format(dataset))
    search_space["dataset"] = dataset

    search_alg = OptunaSearch(
        metric=metric_for_ranking,
        mode=mode,
    )
    scheduler = AsyncHyperBandScheduler(
        grace_period=5,
        max_t=100,
        metric=metric_for_ranking,
        mode=mode,
    )
    analysis = tune.run(
        one_run,
        config=search_space,
        num_samples=num_samples,
        raise_on_failed_trial=False,  # may some failures maybe acceptable
        search_alg=search_alg,
        scheduler=scheduler,
        stop=TimeoutStopper(max_hours * 3600),  # add a timeout
    )
    best_config = analysis.get_best_config(metric=metric_for_ranking, mode=mode)
    logger.info("best_config: {}".format(best_config))
    ray.shutdown()

    logger.info("now we train and eval the classifier using tuned hyperparameters")
    # now we report test scores using the classifier with tuned hyperparamers

    test_perf = train_greedy_and_eval(
        Dataset(name=dataset, datadir=pjoin(ROOT_DIR, "data")), best_config, N_REPEATS
    )
    logger.info("performance on test data:")
    logger.info(pd.DataFrame.from_records([test_perf]).to_markdown())

    output_path = pjoin(ROOT_DIR, "outputs", dataset, "test_scores.json")
    makedir(output_path, usedir=True)
    save_json(test_perf, output_path)
    logger.info("result saved to {}".format(output_path))


if __name__ == "__main__":
    main()
