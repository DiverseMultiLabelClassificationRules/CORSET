import ray
import mlflow
import sys
import logzero
import pandas as pd
from corset.data import Dataset
from corset.utils import makedir, get_experiment_id_by_name, create_experiment_if_needed
from corset.exp import train_greedy_and_eval

logzero.loglevel(logzero.WARNING)

GREEDY_VERSION = 'v3'
EXPERIMENT_NAME = "greedy"

ray.init(num_cpus=8, ignore_reinit_error=True)

def one_run(ds_name, config):
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

        mlflow.pyfunc.log_model('model', python_model=clf)


remote_func = ray.remote(one_run)

ds = sys.argv[1]
n_repeats = 8

###############
# configs
###############

config = {"n_tails_per_iter": 1000, "n_heads_per_tail": 10}

if ds == "toy":
    EXPERIMENT_NAME += '_debug'
    config['n_tails_per_iter'] = 10
    config['n_heads_per_tail'] = 1
    n_repeats = 1
elif ds == "bibtex":
    config["lambd"] = 1.20381
    config["n_max_rules"] = 84
    config["min_label_proba"] = 0.5
    config["min_feature_proba"] = 1.0
elif ds == "medical":
    config["lambd"] = 5.95374
    config["n_max_rules"] = 49
    config["min_feature_proba"] = 0.5
elif ds == "emotions":
    config["lambd"] = 1.21086
    config["n_max_rules"] = 19
    config["min_feature_proba"] = 0.8
elif ds == "birds":
    config["lambd"] =  1.1066
    config["n_max_rules"] = 48
    config["min_feature_proba"] = 0.95
elif ds == 'enron':
    config["lambd"] = 1.05774
    config["n_max_rules"] = 49
    config["min_feature_proba"] = 1.0
elif ds == 'mediamill':
    config["lambd"] = 1.33413
    config["n_max_rules"] = 44
    config["min_feature_proba"] = 0.8
    

##############
# run the experiments
##############

# first create experiment to avoid racing condition
create_experiment_if_needed(EXPERIMENT_NAME)

res_ids = [
    remote_func.remote(ds, config)
    for _ in range(n_repeats)
]
perfs = ray.get(res_ids)

ray.shutdown()
    
