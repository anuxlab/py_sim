import pytest

from k8s_sim.metrics import MetricsCollector, _percentile, _jains_fairness_index, default_tenant_key_fn
from k8s_sim.resource import PodResource


def test_percentile_basic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 0.0) == 1.0
    assert _percentile(vals, 1.0) == 5.0
    assert _percentile(vals, 0.5) == 3.0


def test_percentile_empty():
    assert _percentile([], 0.5) == 0.0


def test_jains_fairness_perfect_equality():
    assert _jains_fairness_index([10.0, 10.0, 10.0, 10.0]) == pytest.approx(1.0)


def test_jains_fairness_max_unfairness():
    # one tenant gets everything, others get nothing -> J over the nonzero
    # entries alone is trivially 1.0 with a single nonzero entry, which is
    # why summary() also reports num_tenants alongside the index.
    assert _jains_fairness_index([100.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_jains_fairness_skewed_among_active_tenants():
    val = _jains_fairness_index([100.0, 10.0])
    assert 0.5 < val < 1.0  # less fair than equal split, more fair than all-or-nothing


def test_jains_fairness_empty_is_one():
    assert _jains_fairness_index([]) == 1.0


def test_default_tenant_key_fn_is_stable():
    p = PodResource(milli_cpu=1000, name="pod-a")
    assert default_tenant_key_fn(p) == default_tenant_key_fn(p)


def test_default_tenant_key_fn_distributes_across_buckets():
    tenants = {default_tenant_key_fn(PodResource(milli_cpu=1000, name=f"pod-{i}")) for i in range(200)}
    assert len(tenants) > 1  # not everything collapsing into one bucket


def test_time_weighted_mean_constant_series():
    m = MetricsCollector()
    samples = [(0.0, 0.5), (10.0, 0.5), (20.0, 0.5)]
    assert m._time_weighted_mean(samples) == pytest.approx(0.5)


def test_time_weighted_mean_weights_by_duration():
    m = MetricsCollector()
    samples = [(0.0, 0.0), (10.0, 1.0), (11.0, 1.0)]
    result = m._time_weighted_mean(samples)
    assert result < 0.2


def test_time_weighted_mean_single_sample():
    m = MetricsCollector()
    assert m._time_weighted_mean([(5.0, 0.7)]) == 0.7


def test_time_weighted_mean_empty():
    m = MetricsCollector()
    assert m._time_weighted_mean([]) == 0.0


def test_full_lifecycle_summary_sane():
    m = MetricsCollector(starvation_threshold_sec=100.0)
    pod = PodResource(milli_cpu=1000, milli_gpu=500, gpu_number=1, gpu_type="V100", name="p1")
    m.record_arrival(pod, arrival_time=0.0)
    m.record_scheduling_attempt(0.001)
    m.record_start(pod, arrival_time=0.0, start_time=5.0, wait_time=5.0, node_name="n0", gpu_ids=[0])
    m.record_utilization_sample(0.0, 0.1, 0.2)
    m.record_utilization_sample(5.0, 0.5, 0.6)
    m.record_departure(pod, departure_time=15.0, node_name="n0", gpu_ids=[0])

    s = m.summary()
    assert s["jobs_completed"] == 1
    assert s["jobs_never_scheduled"] == 0
    assert s["job_waiting_time_sec"]["mean"] == pytest.approx(5.0)
    assert s["starvation_count"] == 0
    assert s["scheduling_latency_ms"]["mean"] == pytest.approx(1.0)
    assert s["fairness_jains_index"] == pytest.approx(1.0)


def test_starvation_counts_both_slow_starts_and_never_scheduled():
    m = MetricsCollector(starvation_threshold_sec=10.0)
    slow = PodResource(milli_cpu=1000, name="slow")
    stuck = PodResource(milli_cpu=1000, name="stuck")
    m.record_arrival(slow, arrival_time=0.0)
    m.record_arrival(stuck, arrival_time=0.0)
    m.record_start(slow, arrival_time=0.0, start_time=50.0, wait_time=50.0)
    m.record_never_scheduled(stuck, last_time=200.0)

    s = m.summary()
    assert s["starvation_count"] == 2
    assert s["jobs_never_scheduled"] == 1


def test_interference_zero_without_gpu_sharing():
    m = MetricsCollector()
    pod = PodResource(milli_cpu=1000, milli_gpu=1000, gpu_number=1, name="exclusive")
    m.record_arrival(pod, arrival_time=0.0)
    m.record_start(pod, arrival_time=0.0, start_time=0.0, wait_time=0.0, node_name="n0", gpu_ids=[0])
    s = m.summary()
    assert s["interference_intensity"]["max"] == 0.0


def test_interference_nonzero_when_two_share_pods_co_resident():
    m = MetricsCollector()
    p1 = PodResource(milli_cpu=500, milli_gpu=300, gpu_number=1, name="share1")
    p2 = PodResource(milli_cpu=500, milli_gpu=300, gpu_number=1, name="share2")
    for p in (p1, p2):
        m.record_arrival(p, arrival_time=0.0)
        m.record_start(p, arrival_time=0.0, start_time=0.0, wait_time=0.0, node_name="n0", gpu_ids=[0])
    s = m.summary()
    assert s["interference_intensity"]["max"] >= 1.0
