from dmrs.evaluation import TailEvaluator

from .fixtures import toy_dataset as ds

def test_TailEvaluator(ds):
    e = TailEvaluator(ds)
    samples = [[0, 1, 2]]
    assert e.support_size(samples)[0] == 1
    assert e.frequency(samples)[0] == (1 / 4)

    assert e.area(samples)[0] == 3
    assert e.length(samples)[0] == 3

    assert e.edge_ratio(samples)[0] == (2 + 2 + 0.5) / 6
    assert e.edge_density(samples)[0] == (2 + 2 + 0.5) / 3

    df = e.report(samples)
    assert set(df.columns.tolist()) == \
        {'support_size', 'frequency', 'area', 'edge_density', 'edge_ratio', 'length'}