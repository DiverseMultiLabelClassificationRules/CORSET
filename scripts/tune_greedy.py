#!/usr/bin/env python
# coding: utf-8

import sys
import logzero
import ray
import mlflow
import pandas as pd
from time import time
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import TimeoutStopper
from logzero import logger


from corset.utils import pjoin, makedir, save_json, create_experiment_if_needed, get_experiment_id_by_name
from corset.data import Dataset
from corset.evaluation import Evaluator
from corset.exp import create_greedy_v2_from_config, train_greedy_and_eval

# WARNING:
# change the following variables to match your setup
ROOT_DIR = "/Users/hanxiao1/code/corset"
MLFLOW_TRACK_URI = 'file:///Users/hanxiao1/code/corset/mlruns'

N_REPEATS = 10

def one_run(config):
    mlflow.set_tracking_uri(MLFLOW_TRACK_URI)
    logzero.loglevel(logzero.INFO)

    ds_name = config["dataset"]

    ds = Dataset(name=ds_name, datadir=pjoin(ROOT_DIR, "data"), split_train=True, train_ratio=0.7)
    perf_list = []
    
    for i in range(N_REPEATS):
        experiment_id = get_experiment_id_by_name(config['experiment_name'])

        tags = {'dataset': ds.name, "greedy_version": config['greedy_version']}

        with mlflow.start_run(experiment_id=experiment_id, tags=tags):
            clf, perf = train_greedy_and_eval(
                ds, config, greedy_version=config['greedy_version'], validation_mode=True
            )

            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(perf)
            # mlflow.sklearn.log_model(clf, 'model')

        perf_list.append(perf)

    df = pd.DataFrame.from_records(perf_list)
    mean_perf = df.mean().to_dict()
    tune.report(**mean_perf)

def main():
    dataset = sys.argv[1]
    greedy_version = sys.argv[2]

    num_cpus = 8
    num_samples = 32
    max_hours = 23.5
    metric_for_ranking = "micro_f1"
    mode = "max"
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    # configs per dataset
    search_space_base = {'greedy_version': greedy_version}
    search_space_base['experiment_name'] = 'tune_greedy_{}'.format(greedy_version)

    if dataset == "toy":
        search_space = {
            "lambd": tune.loguniform(1e0, 1e2),
            "n_max_rules": tune.randint(2, 5)
        }
        num_samples = 2
        search_space_base['experiment_name'] += '_debug'
    elif dataset == "emotions":
        num_samples = 64
        search_space = {
            "lambd": tune.loguniform(1e0, 1e3),
            "n_max_rules": tune.randint(5, 50),
        }
    elif dataset == "mediamill":
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

    search_space.update(search_space_base)
    create_experiment_if_needed(search_space['experiment_name'])

    logger.info("dataset: {}".format(dataset))
    search_space["dataset"] = dataset

    logger.info("search space: {}".format(search_space))

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

if __name__ == "__main__":
    main()
