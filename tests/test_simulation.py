import pytest

from k8s_sim import Cluster, NodeResource, PodResource
from k8s_sim.trace import TracePod
from k8s_sim.simulation import TimeDrivenSimulator, _job_duration, DEFAULT_DURATION_SEC


def _trace_pod(name, cpu=1000, gpu_milli=0, gpu_num=0, creation=0, deletion=100, scheduled=0):
    pod = PodResource(milli_cpu=cpu, milli_gpu=gpu_milli, gpu_number=gpu_num, name=name)
    return TracePod(pod=pod, qos="LS", pod_phase="Running",
                     creation_time=creation, deletion_time=deletion, scheduled_time=scheduled)


def _small_cluster(n_nodes=2, cpu=8000, gpu=2):
    nodes = [NodeResource(name=f"n{i}", milli_cpu_left=cpu, milli_cpu_capacity=cpu,
                          milli_gpu_left_list=[1000] * gpu, gpu_type="V100") for i in range(n_nodes)]
    return Cluster(nodes)


def test_job_duration_uses_real_trace_timestamps():
    tp = _trace_pod("p", creation=0, scheduled=10, deletion=110)
    assert _job_duration(tp, DEFAULT_DURATION_SEC) == 100.0


def test_job_duration_falls_back_when_missing():
    tp = _trace_pod("p", creation=0, scheduled=-1, deletion=110)
    assert _job_duration(tp, DEFAULT_DURATION_SEC) == DEFAULT_DURATION_SEC


def test_all_pods_eventually_depart_and_release_resources():
    """After every pod has arrived AND departed, the cluster should be back
    to its original free capacity -- this is the correctness property that
    motivated tracking exact gpu_ids for release."""
    cluster = _small_cluster(n_nodes=2, cpu=8000, gpu=2)
    trace_pods = [_trace_pod(f"p{i}", cpu=1000, gpu_milli=500, gpu_num=1,
                              creation=i * 10, scheduled=i * 10, deletion=i * 10 + 50)
                  for i in range(6)]
    sim = TimeDrivenSimulator(cluster, policy="best-fit")
    metrics = sim.run(trace_pods)

    s = metrics.summary()
    assert s["jobs_completed"] == 6
    assert s["jobs_never_scheduled"] == 0

    for node in cluster.node_list():
        assert node.milli_cpu_left == node.milli_cpu_capacity
        assert all(g == 1000 for g in node.milli_gpu_left_list)


def test_contention_produces_nonzero_waiting_time():
    """1 node, 1 GPU; 3 pods each needing the whole GPU, all arriving at
    once -- the 2nd and 3rd MUST wait for the 1st (and 2nd) to depart."""
    cluster = _small_cluster(n_nodes=1, cpu=8000, gpu=1)
    trace_pods = [_trace_pod(f"p{i}", cpu=1000, gpu_milli=1000, gpu_num=1,
                              creation=0, scheduled=0, deletion=100)
                  for i in range(3)]
    sim = TimeDrivenSimulator(cluster, policy="best-fit")
    metrics = sim.run(trace_pods)
    s = metrics.summary()
    assert s["jobs_completed"] == 3
    assert s["job_waiting_time_sec"]["max"] > 0


def test_permanently_oversized_demand_is_never_scheduled():
    cluster = _small_cluster(n_nodes=1, cpu=1000, gpu=1)
    trace_pods = [_trace_pod("too-big", cpu=999999, gpu_milli=0, gpu_num=0,
                              creation=0, scheduled=0, deletion=100)]
    sim = TimeDrivenSimulator(cluster, policy="best-fit")
    metrics = sim.run(trace_pods)
    s = metrics.summary()
    assert s["jobs_never_scheduled"] == 1
    assert s["jobs_completed"] == 0


def test_departure_frees_room_for_queued_pod():
    """1 GPU; pod A occupies it until t=50; pod B arrives at t=0 needing the
    whole GPU too -- B should start when A departs, not before."""
    cluster = _small_cluster(n_nodes=1, cpu=8000, gpu=1)
    trace_pods = [
        _trace_pod("A", cpu=1000, gpu_milli=1000, gpu_num=1, creation=0, scheduled=0, deletion=50),
        _trace_pod("B", cpu=1000, gpu_milli=1000, gpu_num=1, creation=0, scheduled=0, deletion=150),
    ]
    sim = TimeDrivenSimulator(cluster, policy="best-fit")
    metrics = sim.run(trace_pods)
    s = metrics.summary()
    assert s["jobs_completed"] == 2
    assert s["job_waiting_time_sec"]["max"] == pytest.approx(50.0, abs=1.0)


def test_scheduling_latency_is_measured():
    cluster = _small_cluster(n_nodes=3, cpu=8000, gpu=2)
    trace_pods = [_trace_pod(f"p{i}", cpu=1000, creation=i, scheduled=i, deletion=i + 50) for i in range(5)]
    sim = TimeDrivenSimulator(cluster, policy="fgd", typical_pods=[])
    metrics = sim.run(trace_pods)
    s = metrics.summary()
    assert s["scheduling_latency_ms"]["mean"] >= 0.0
    assert s["scheduling_latency_ms"]["max"] >= s["scheduling_latency_ms"]["mean"]
