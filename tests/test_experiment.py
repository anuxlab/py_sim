import os

import pytest

from k8s_sim.trace import DATA_DIR

DATA_PRESENT = os.path.isdir(DATA_DIR) and os.path.isfile(os.path.join(DATA_DIR, "openb_pod_list_default.csv"))
pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="data/csv/ trace files not present in this checkout")


def _small_nodes(n=20):
    from k8s_sim.trace import load_nodes_csv
    import random
    nodes = load_nodes_csv()
    return random.Random(0).sample(nodes, n)


def test_run_experiment_smoke():
    from k8s_sim.experiment import run_experiment, RunConfig

    cfg = RunConfig(policy="fgd", workload="openb_pod_list_gpushare100.csv", n_pods=20, seed=0)
    r = run_experiment(cfg, nodes=_small_nodes())
    assert r.scheduled + r.unscheduled == 20
    assert 0.0 <= r.alloc_ratio_cpu <= 1.0
    assert 0.0 <= r.frag_ratio <= 1.0
    assert set(r.frag_pct.keys()) == {
        "q1_lack_both", "q2_lack_gpu", "q3_satisfied", "q4_lack_cpu",
        "xl_satisfied", "xr_lack_cpu", "no_access",
    }


def test_run_benchmark_suite_writes_csv(tmp_path):
    from k8s_sim.experiment import run_benchmark_suite

    out_path = str(tmp_path / "bench.csv")
    results = run_benchmark_suite(
        policies=["random", "fgd"],
        workloads=["openb_pod_list_gpushare100.csv"],
        pod_counts=(20,),
        seeds=(0,),
        node_sample_size=20,
        out_path=out_path,
        verbose=False,
    )
    assert len(results) == 2
    assert os.path.isfile(out_path)

    import csv
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["policy"] for r in rows} == {"random", "fgd"}


def test_plotting_generates_files(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from k8s_sim.experiment import run_benchmark_suite
    from k8s_sim import plotting

    out_path = str(tmp_path / "bench.csv")
    run_benchmark_suite(
        policies=["random", "fgd", "best-fit"],
        workloads=["openb_pod_list_gpushare100.csv"],
        pod_counts=(20,),
        seeds=(0,),
        node_sample_size=20,
        out_path=out_path,
        verbose=False,
    )

    monkeypatch.setattr(plotting, "PLOTS_DIR", str(tmp_path))
    df = plotting.load_results(out_path)
    p1 = plotting.plot_frag_ratio_by_policy(df, out_name="t1.png")
    p2 = plotting.plot_frag_breakdown_stacked(df, out_name="t2.png")
    assert os.path.isfile(p1)
    assert os.path.isfile(p2)
