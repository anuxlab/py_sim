import pytest

from k8s_sim.resource import NodeResource, PodResource
from k8s_sim.fragmentation import build_typical_pods, unfit_fraction, cluster_fragmentation_score
from k8s_sim.policies import list_policies, get_policy, POLICIES
from k8s_sim.cluster import Cluster
from k8s_sim.trace import reset_cluster
from k8s_sim.gputrace_bridge import TimedPodEvent, load_gputrace_export
from k8s_sim.event_runtime import EventDrivenRunner, RunConfig


def make_nodes(n=4, milli_cpu=8000, gpu_count=4):
    return {
        f"node_{i}": NodeResource(node_id=f"node_{i}", milli_cpu_capacity=milli_cpu,
                                   gpu_count=gpu_count, gpu_type="A100")
        for i in range(n)
    }


def test_node_add_remove_roundtrip():
    n = NodeResource(node_id="n0", milli_cpu_capacity=4000, gpu_count=2)
    pod = PodResource(pod_id="p0", milli_cpu=1000, milli_gpu=500, gpu_number=1)
    assert n.add(pod) is True
    assert n.remaining_milli_cpu == 3000
    assert n.remaining_gpu_count == 1
    removed = n.remove("p0")
    assert removed is pod
    assert n.remaining_milli_cpu == 4000
    assert n.remaining_gpu_count == 2


def test_node_add_rejects_oversized_pod():
    n = NodeResource(node_id="n0", milli_cpu_capacity=1000, gpu_count=1)
    pod = PodResource(pod_id="p0", milli_cpu=2000)
    assert n.add(pod) is False
    assert n.remaining_milli_cpu == 1000  # unchanged on rejection


def test_gpu_type_mismatch_rejected():
    n = NodeResource(node_id="n0", milli_cpu_capacity=4000, gpu_count=2, gpu_type="A100")
    pod = PodResource(pod_id="p0", milli_cpu=1000, gpu_number=1, gpu_type="V100")
    assert pod.fits_in(n) is False


@pytest.mark.parametrize("policy_name", list_policies())
def test_every_policy_places_when_capacity_available(policy_name):
    nodes = make_nodes()
    cluster = Cluster(nodes)
    pod = PodResource(pod_id="p0", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100")
    node_id = cluster.schedule_pod(pod, policy=policy_name)
    assert node_id is not None


@pytest.mark.parametrize("policy_name", list_policies())
def test_every_policy_returns_none_when_nothing_fits(policy_name):
    nodes = {"n0": NodeResource(node_id="n0", milli_cpu_capacity=500, gpu_count=0)}
    cluster = Cluster(nodes)
    pod = PodResource(pod_id="p0", milli_cpu=1000)
    assert cluster.schedule_pod(pod, policy=policy_name) is None


def test_best_fit_minimizes_leftover():
    nodes = {
        "small_leftover": NodeResource(node_id="small_leftover", milli_cpu_capacity=1100, gpu_count=0),
        "big_leftover": NodeResource(node_id="big_leftover", milli_cpu_capacity=5000, gpu_count=0),
    }
    cluster = Cluster(nodes)
    pod = PodResource(pod_id="p0", milli_cpu=1000)
    chosen = cluster.schedule_pod(pod, policy="best_fit")
    assert chosen == "small_leftover"


def test_worst_fit_maximizes_leftover():
    nodes = {
        "small_leftover": NodeResource(node_id="small_leftover", milli_cpu_capacity=1100, gpu_count=0),
        "big_leftover": NodeResource(node_id="big_leftover", milli_cpu_capacity=5000, gpu_count=0),
    }
    cluster = Cluster(nodes)
    pod = PodResource(pod_id="p0", milli_cpu=1000)
    chosen = cluster.schedule_pod(pod, policy="worst_fit")
    assert chosen == "big_leftover"


def test_fragmentation_score_zero_on_empty_cluster():
    nodes = make_nodes()
    pods = [PodResource(pod_id=f"p{i}", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100")
            for i in range(3)]
    typical = build_typical_pods(pods)
    score = cluster_fragmentation_score(list(nodes.values()), typical)
    assert score == 0.0


def test_fragmentation_score_increases_as_cluster_fills():
    nodes = {"n0": NodeResource(node_id="n0", milli_cpu_capacity=4000, gpu_count=4, gpu_type="A100")}
    typical = [PodResource(pod_id="t0", milli_cpu=1000, milli_gpu=1000, gpu_number=1)]
    before = unfit_fraction(nodes["n0"], typical)
    nodes["n0"].add(PodResource(pod_id="p0", milli_cpu=4000, milli_gpu=4000, gpu_number=4))
    after = unfit_fraction(nodes["n0"], typical)
    assert after > before
    assert after == 1.0  # fully packed node can't fit anything


def test_fgd_prefers_less_fragmenting_placement():
    """Construct two nodes where best_fit and fgd should disagree: best_fit
    picks the tightest-leftover node regardless of whether that leftover is
    still useful; fgd should avoid leaving a node in a state where a
    typical pod no longer fits, when an alternative avoids that."""
    nodes = {
        # placing a 1-GPU pod here leaves exactly one more 1-GPU slot —
        # still useful
        "keeps_usable": NodeResource(node_id="keeps_usable", milli_cpu_capacity=4000, gpu_count=2, gpu_type="A100"),
        # smaller node where placing the pod exactly fills it — tightest
        # leftover (what best_fit wants) but leaves nothing usable
        "fills_exactly": NodeResource(node_id="fills_exactly", milli_cpu_capacity=1000, gpu_count=1, gpu_type="A100"),
    }
    typical = [PodResource(pod_id="t0", milli_cpu=1000, milli_gpu=1000, gpu_number=1)]
    cluster = Cluster(nodes)
    pod = PodResource(pod_id="p0", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100")

    chosen_bf = cluster.schedule_pod(pod, policy="best_fit")
    assert chosen_bf == "fills_exactly"

    nodes2 = reset_cluster(nodes)
    cluster2 = Cluster(nodes2)
    chosen_fgd = cluster2.schedule_pod(pod, policy="fgd", typical_pods=typical)
    assert chosen_fgd == "keeps_usable"


def test_reset_cluster_restores_full_capacity():
    nodes = make_nodes()
    nodes["node_0"].add(PodResource(pod_id="p0", milli_cpu=1000, gpu_number=1, milli_gpu=1000))
    fresh = reset_cluster(nodes)
    assert fresh["node_0"].remaining_milli_cpu == nodes["node_0"].milli_cpu_capacity
    assert fresh["node_0"].allocated == {}


# -- event-driven runtime -----------------------------------------------

def _events(n=20, gap=10.0, duration=50.0, milli_cpu=1000, gpu_number=1, milli_gpu=1000):
    return [
        TimedPodEvent(
            submit_time=i * gap, duration=duration,
            pod=PodResource(pod_id=f"p{i}", milli_cpu=milli_cpu, milli_gpu=milli_gpu,
                             gpu_number=gpu_number, gpu_type="A100"),
        )
        for i in range(n)
    ]


def test_event_runner_admits_all_when_capacity_sufficient():
    nodes = make_nodes(n=10)
    cluster = Cluster(nodes)
    events = _events(n=20, gap=10.0, duration=5.0)  # plenty of turnover
    runner = EventDrivenRunner(cluster, RunConfig(policy="best_fit"))
    result = runner.run(events)
    assert result.n_admitted == 20
    assert result.rejection_rate == 0.0


def test_event_runner_queues_under_contention_then_admits():
    """One node, capacity for one pod at a time. Ten pods submitted
    instantly (gap=0) with short duration should all eventually get
    admitted (finite wait), none should be permanently rejected, since
    unlimited queueing (max_queue_time=None) is the default."""
    nodes = {"n0": NodeResource(node_id="n0", milli_cpu_capacity=1000, gpu_count=1, gpu_type="A100")}
    cluster = Cluster(nodes)
    events = [
        TimedPodEvent(submit_time=0.0, duration=1.0,
                       pod=PodResource(pod_id=f"p{i}", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100"))
        for i in range(10)
    ]
    runner = EventDrivenRunner(cluster, RunConfig(policy="first_fit"))
    result = runner.run(events)
    assert result.n_admitted == 10
    assert result.rejection_rate == 0.0
    # only the first pod should have zero wait; the rest must queue
    waits = sorted(o.wait_time for o in result.outcomes)
    assert waits[0] == 0.0
    assert waits[-1] > 0.0


def test_event_runner_max_queue_time_causes_rejection():
    nodes = {"n0": NodeResource(node_id="n0", milli_cpu_capacity=1000, gpu_count=1, gpu_type="A100")}
    cluster = Cluster(nodes)
    events = [
        TimedPodEvent(submit_time=0.0, duration=100.0,
                       pod=PodResource(pod_id="blocker", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100")),
        TimedPodEvent(submit_time=1.0, duration=5.0,
                       pod=PodResource(pod_id="impatient", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100")),
    ]
    runner = EventDrivenRunner(cluster, RunConfig(policy="first_fit", max_queue_time=2.0))
    result = runner.run(events)
    impatient = [o for o in result.outcomes if o.pod_id == "impatient"][0]
    assert impatient.admitted is False


def test_event_runner_rejects_pod_larger_than_any_node():
    nodes = {"n0": NodeResource(node_id="n0", milli_cpu_capacity=1000, gpu_count=1, gpu_type="A100")}
    cluster = Cluster(nodes)
    events = [TimedPodEvent(submit_time=0.0, duration=10.0,
                             pod=PodResource(pod_id="whale", milli_cpu=1000, milli_gpu=1000, gpu_number=5, gpu_type="A100"))]
    runner = EventDrivenRunner(cluster, RunConfig(policy="first_fit"))
    result = runner.run(events)
    assert result.n_admitted == 0
    assert result.outcomes[0].wait_time is None


def test_event_runner_seed_reproducibility():
    nodes1 = make_nodes(n=3)
    nodes2 = make_nodes(n=3)
    events = _events(n=30, gap=1.0, duration=5.0)
    r1 = EventDrivenRunner(Cluster(nodes1), RunConfig(policy="random", seed=7)).run(events)
    r2 = EventDrivenRunner(Cluster(nodes2), RunConfig(policy="random", seed=7)).run(events)
    assert [o.node_id for o in r1.outcomes] == [o.node_id for o in r2.outcomes]


def test_bursty_load_rejects_more_than_smooth_load_same_total_count():
    """The core claim this whole bridge exists to demonstrate: identical
    total job count and identical average rate, but clustered arrivals
    produce worse outcomes on a capacity-constrained cluster than smooth
    arrivals — a difference static/snapshot scheduling cannot show at all.
    """
    n_jobs, span, duration = 40, 40.0, 8.0
    smooth = [
        TimedPodEvent(submit_time=float(i), duration=duration,
                       pod=PodResource(pod_id=f"s{i}", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100"))
        for i in range(n_jobs)
    ]
    # same 40 jobs, same span, but clustered into two bursts of 20 at t=0 and t=20
    bursty = [
        TimedPodEvent(submit_time=0.0 if i < 20 else 20.0, duration=duration,
                       pod=PodResource(pod_id=f"b{i}", milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="A100"))
        for i in range(n_jobs)
    ]
    nodes_smooth = {"n0": NodeResource(node_id="n0", milli_cpu_capacity=3000, gpu_count=3, gpu_type="A100")}
    nodes_bursty = {"n0": NodeResource(node_id="n0", milli_cpu_capacity=3000, gpu_count=3, gpu_type="A100")}

    r_smooth = EventDrivenRunner(Cluster(nodes_smooth), RunConfig(policy="first_fit", max_queue_time=5.0)).run(smooth)
    r_bursty = EventDrivenRunner(Cluster(nodes_bursty), RunConfig(policy="first_fit", max_queue_time=5.0)).run(bursty)

    assert r_bursty.rejection_rate > r_smooth.rejection_rate
