import os

import pytest

from k8s_sim.trace import load_nodes_csv, load_pods_csv, build_typical_pods, list_available_traces, DATA_DIR

DATA_PRESENT = os.path.isdir(DATA_DIR) and os.path.isfile(os.path.join(DATA_DIR, "openb_pod_list_default.csv"))

pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="data/csv/ trace files not present in this checkout")


def test_list_available_traces_nonempty():
    traces = list_available_traces()
    assert len(traces) > 0
    assert all(t.startswith("openb_pod_list_") for t in traces)


def test_load_nodes_csv():
    nodes = load_nodes_csv()
    assert len(nodes) > 0
    gpu_nodes = [n for n in nodes if n.gpu_number > 0]
    cpu_nodes = [n for n in nodes if n.gpu_number == 0]
    assert gpu_nodes and cpu_nodes  # heterogeneous cluster, both kinds present
    for n in nodes:
        assert n.milli_cpu_left == n.milli_cpu_capacity  # freshly loaded == fully idle
        assert len(n.milli_gpu_left_list) == n.gpu_number
        assert all(g == 1000 for g in n.milli_gpu_left_list)


def test_load_pods_csv_limit():
    pods = load_pods_csv(limit=50)
    assert len(pods) == 50
    for tp in pods:
        assert tp.pod.milli_cpu > 0 or tp.pod.gpu_number >= 0
        assert tp.pod.name.startswith("openb-pod-")


def test_load_pods_csv_sample_is_reproducible_with_seed():
    a = load_pods_csv(limit=30, sample=True, seed=42)
    b = load_pods_csv(limit=30, sample=True, seed=42)
    assert [tp.pod.name for tp in a] == [tp.pod.name for tp in b]


def test_build_typical_pods_from_trace():
    trace_pods = load_pods_csv(limit=500, sample=True, seed=1)
    typical = build_typical_pods(trace_pods)
    assert len(typical) > 0
    total_pct = sum(tp.percentage for tp in typical)
    assert abs(total_pct - 1.0) < 1e-6
