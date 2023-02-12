import ray
import mlflow
import sys
import logzero
import pandas as pd
from corset.data import Dataset
from corset.utils import makedir, get_experiment_id_by_name
from corset.exp import train_greedy_and_eval

logzero.loglevel(logzero.WARNING)

EXPERIMENT_NAME = "effect_of_min_feature_proba"
GREEDY_VERSION = 'v2'

ray.init(num_cpus=8, ignore_reinit_error=True)

def one_run(ds_name, min_feature_proba, config):
    config["min_feature_proba"] = min_feature_proba
    ds = Dataset(name=ds_name)
    experiment_id = get_experiment_id_by_name(EXPERIMENT_NAME)

    tags = {'dataset': ds.name, "greedy_version": GREEDY_VERSION}

    with mlflow.start_run(experiment_id=experiment_id, tags=tags):
        clf, perf = train_greedy_and_eval(
            ds,
            config=config,
            greedy_version=GREEDY_VERSION,
            validation_mode=False
        )

        mlflow.log_params(clf.get_params())
        mlflow.log_metrics(perf)
        mlflow.sklearn.log_model(clf, 'model')

remote_func = ray.remote(one_run)

##############
# configuration
##############
ds = sys.argv[1]
n_repeats = 8
config = {"n_tails_per_iter": 1000, "n_heads_per_tail": 10}


if ds == "toy":
    EXPERIMENT_NAME = 'effect_of_min_feature_proba_debug'
    config['n_tails_per_iter'] = 10
    config['n_heads_per_tail'] = 1
    n_repeats = 4
    min_probas = [0.0, 0.5, 1.0]
elif ds == "bibtex":
    config["lambd"] = 2.14
    config["n_max_rules"] = 85
    config["min_label_proba"] = 0.5
    min_probas = [0.7, 0.8, 0.9, 1.0]
elif ds == "medical":
    config["lambd"] = 42.05799926225756
    config[
        "n_max_rules"
    ] = 40  # assumption: 40 is a good esimate of the number of rules that should be learned
    min_probas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
elif ds == "emotions":
    config["lambd"] = 1.0092288035165227
    config["n_max_rules"] = 25
    min_probas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

##############
# run the experiments
##############

res_ids = [
    remote_func.remote(ds, min_proba, config)
    for _ in range(n_repeats)
    for min_proba in min_probas
]
perfs = ray.get(res_ids)

ray.shutdown()
