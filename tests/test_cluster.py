"""
End-to-end scheduling integration tests: for every registered policy, run a
batch of pods through Cluster.schedule_pods and verify resource conservation
and consistent bookkeeping. This is what actually proves a new policy is
"compatible and integrated," beyond the per-call contract in
test_policy_contract.py.
"""

import pytest

from k8s_sim import Cluster, NodeResource, PodResource, policies as policy_mod

ALL_POLICIES = sorted(policy_mod.POLICIES.keys())


def make_test_cluster():
    return [
        NodeResource(name=f"node-{i}", milli_cpu_left=16000, milli_cpu_capacity=16000,
                     milli_gpu_left_list=[1000, 1000], gpu_type="V100")
        for i in range(6)
    ]


def make_workload():
    pods = []
    for i in range(20):
        pods.append(PodResource(milli_cpu=1000, milli_gpu=500, gpu_number=1,
                                 gpu_type="V100", name=f"pod-{i}"))
    return pods


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_schedule_pods_runs_without_error(policy, typical_pods):
    cluster = Cluster(make_test_cluster())
    result = cluster.schedule_pods(make_workload(), policy, typical_pods=typical_pods)
    assert len(result.scheduled) + len(result.unscheduled) == 20


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_schedule_pods_conserves_resources(policy, typical_pods):
    """No node should ever end up with negative free capacity or free
    capacity exceeding its own declared capacity."""
    cluster = Cluster(make_test_cluster())
    cluster.schedule_pods(make_workload(), policy, typical_pods=typical_pods)
    for node in cluster.node_list():
        assert node.milli_cpu_left >= 0, f"[{policy}] node {node.name} has negative CPU left"
        assert node.milli_cpu_left <= node.milli_cpu_capacity, \
            f"[{policy}] node {node.name} exceeds its own CPU capacity"
        for g in node.milli_gpu_left_list:
            assert 0 <= g <= 1000, f"[{policy}] node {node.name} has an out-of-range GPU device ({g}m)"


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_schedule_pods_placement_matches_result(policy, typical_pods):
    cluster = Cluster(make_test_cluster())
    result = cluster.schedule_pods(make_workload(), policy, typical_pods=typical_pods)
    assert set(result.placement.keys()) == set(result.scheduled)
    node_names = {n.name for n in cluster.node_list()}
    for pod_name, node_name in result.placement.items():
        assert node_name in node_names, f"[{policy}] placed {pod_name} on unknown node {node_name}"


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_overloaded_workload_leaves_some_pods_unscheduled(policy, typical_pods):
    """Sanity check: a workload that can't possibly fit should report
    unscheduled pods rather than silently dropping/over-allocating them."""
    cluster = Cluster(make_test_cluster())  # 6 nodes x 2 GPUs = 12 GPUs total
    huge_workload = [
        PodResource(milli_cpu=1000, milli_gpu=1000, gpu_number=1, gpu_type="V100", name=f"pod-{i}")
        for i in range(50)  # way more than 12 whole GPUs
    ]
    result = cluster.schedule_pods(huge_workload, policy, typical_pods=typical_pods)
    assert len(result.scheduled) <= 12
    assert len(result.unscheduled) >= 38


def test_gpu_type_mismatch_is_never_scheduled():
    """A pod requiring a GPU type the cluster doesn't have must be rejected
    by every policy, not silently placed."""
    cluster_nodes = make_test_cluster()  # all V100
    pod = PodResource(milli_cpu=1000, milli_gpu=500, gpu_number=1, gpu_type="A100", name="mismatched")
    for policy in ALL_POLICIES:
        cluster = Cluster(cluster_nodes)
        result = cluster.schedule_pods([pod], policy)
        assert pod.name in result.unscheduled, \
            f"[{policy}] scheduled a GPU-type-mismatched pod -- filter step is broken"
