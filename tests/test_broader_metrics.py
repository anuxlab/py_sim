import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.broader_metrics import jains_fairness_index, percentile  # noqa: E402


def test_jains_index_is_one_when_all_equal():
    assert jains_fairness_index([5.0, 5.0, 5.0, 5.0]) == 1.0


def test_jains_index_approaches_one_over_n_when_maximally_unfair():
    # one entity gets everything, n-1 get nothing
    n = 4
    idx = jains_fairness_index([100.0, 0.0, 0.0, 0.0])
    assert abs(idx - 1.0 / n) < 1e-9


def test_jains_index_is_between_bounds_for_mixed_values():
    idx = jains_fairness_index([10.0, 20.0, 5.0, 15.0])
    assert 0.25 < idx < 1.0


def test_jains_index_handles_all_zero_as_vacuously_fair():
    assert jains_fairness_index([0.0, 0.0, 0.0]) == 1.0


def test_jains_index_empty_list():
    assert jains_fairness_index([]) == 1.0


def test_percentile_basic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 1.0) == 5.0  # clamped to last element, not out of range
    assert percentile(vals, 0.5) == 3.0


def test_percentile_empty():
    assert percentile([], 0.95) == 0.0
