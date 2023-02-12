import ray
import pandas as pd
from time import time
from copy import copy
from corset.data import Dataset
from corset.exp import train_and_eval, create_greedy_v2_from_config
ray.init(num_cpus=10, ignore_reinit_error=True)


def one_run(ds_name, config, n_tails_per_iter, n_heads_per_tail):
    ds = Dataset(name=ds_name)
    config['n_tails_per_iter'] = n_tails_per_iter
    config['n_heads_per_tail'] = n_heads_per_tail
    clf = create_greedy_v2_from_config(config)

    s = time()
    perf = train_and_eval(ds, clf)
    elasped = time() - s
    perf['elapsed'] = elasped
        
    ret = copy(config)
    ret.update(perf)
    return ret
# config = {'lambd': 42.05799926225756, 'n_max_rules': 10, 'min_feature_proba': 1.0, 'min_label_proba': 1.0}
# one_run('medical', config, 10, 10)
one_run_r = ray.remote(one_run)
config = {'lambd': 42.05799926225756, 'n_max_rules': 44}
n_repeats = 10
res_ids = [one_run_r.remote('medical', config, n_tails, n_heads)
           for _ in range(n_repeats)
           for (n_tails, n_heads) in [(1000, 10), (10, 1000)]]
res_list = ray.get(res_ids)
df = pd.DataFrame.from_records(res_list)
summ = df.groupby(['n_tails_per_iter', 'n_heads_per_tail'])['micro_f1'].describe()
print(summ)

ray.shutdown()
