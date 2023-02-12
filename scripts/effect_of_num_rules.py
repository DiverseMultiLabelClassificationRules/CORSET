import ray
import sys
import logzero
import pandas as pd
from corset.data import Dataset
from corset.utils import makedir
from corset.exp import train_greedy_and_eval

logzero.loglevel(logzero.WARNING)

ray.init(num_cpus=8, ignore_reinit_error=True)

def one_run(ds_name, n_rules, config):
    config['n_max_rules'] = n_rules
    ds = Dataset(name=ds_name)
    perf = train_greedy_and_eval(ds, config=config, n_repeats=1)
    return perf


remote_func = ray.remote(one_run)

##############
# configuration
##############
ds = sys.argv[1]
n_repeats = 10
config = {}

if ds == 'toy':
    n_rules_list = [1, 2]
    n_repeats = 4
elif ds == 'enron':
    config['lambd'] = 10
    config['min_feature_proba'] = 1.0
    n_rules_list = [10, 20, 30, 40, 50, 60]
    n_repeats = 8
elif ds == 'medical':
    config['lambd'] = 140
    n_rules_list = [30, 40, 50, 60, 70]
    n_repeats = 8
    
##############
# run the experiments
##############

res_ids = [remote_func.remote(ds, n_rules, config) for _ in range(n_repeats) for n_rules in n_rules_list]
perfs = ray.get(res_ids)

###########
# save results
###########
df = pd.DataFrame.from_records(perfs)
output_path = f'outputs/effect_of_rules/{ds}.csv'
makedir(output_path, usedir=True)
df.to_csv(output_path)
print(f'result saved to {output_path}')
print(df.groupby('n_rules').mean().to_markdown())

ray.shutdown()
