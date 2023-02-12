import pandas as pd
import sys
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
from flatten_dict import flatten
from corset.utils import get_experiment_id_by_name, makedir, pjoin, save_file
from logzero import logger

pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', 500)
TRACKING_URI = 'file:///scratch/cs/mldb/han/interpretable-mlc-code/mlruns'
EXPERIMENT_NAME = 'tune_greedy_v3'
METRIC_FOR_RANKING = 'micro_f1'

OUTPUT_DIR = f'./outputs/{EXPERIMENT_NAME}'

dataset = sys.argv[1]
logger.info('\n--- SETUP ---\n experiment={}\n dataset={}\n metric_for_ranking={}'.format(EXPERIMENT_NAME, dataset, METRIC_FOR_RANKING))

cli = MlflowClient(tracking_uri=TRACKING_URI)
experiment_id = get_experiment_id_by_name(EXPERIMENT_NAME)
runs = cli.search_runs(
    experiment_ids=experiment_id,
    filter_string="tag.dataset = '{}'".format(dataset),
    run_view_type=ViewType.ACTIVE_ONLY,
)
rows = []
for run in runs:
    one_row = {'M': run.data.metrics, 'P': run.data.params}
    rows.append(flatten(one_row, reducer='path'))

hyperparam_keys = [key for key in rows[0].keys() if key.startswith('P')]
df = pd.DataFrame.from_records(rows)
summ = df.groupby(hyperparam_keys)['M/{}'.format(METRIC_FOR_RANKING)].describe()[['mean', 'std']].sort_values(by='mean', ascending=False)

output_df = summ.reset_index()
output_df = output_df.rename({'mean': '{}/mean'.format(METRIC_FOR_RANKING), 'std': '{}/std'.format(METRIC_FOR_RANKING)}, axis=1)
output = output_df.to_markdown(index=None)
print(output)
output_path = pjoin(OUTPUT_DIR, dataset, 'results.md')
makedir(output_path, usedir=True)
with open(output_path, 'w') as f:
    f.write(output)
logger.info('\nresults written to {}'.format(output_path))
