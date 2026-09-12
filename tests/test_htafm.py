"""
Unit tests for k8s_sim.topology / k8s_sim.htafm.

These didn't exist before this fix (H-TAFM previously only had demo-script-level
smoke coverage via htafm_demo.py, no dedicated pytest tests) -- see README.md's
"Recent CI fixes" for how topology.py/htafm.py were restored after being
accidentally dropped in an earlier refactor.
"""
from dataclasses import dataclass
from typing import List

import math
import pytest

from k8s_sim.htafm import HTAFMConfig, HTAFMScheduler, VARIANTS, compute_tafi
from k8s_sim.topology import DEFAULT_LEVEL_WEIGHTS, MILLI, Topology, TopoDemand, load_topology_from_csv


@dataclass
class _FakeNode:
    """Minimal stand-in matching what Topology.from_nodes actually needs
    (.name, .milli_cpu_capacity, .milli_gpu_left_list, .gpu_type) --
    deliberately NOT k8s_sim.resource.NodeResource, to pin down that
    from_nodes only depends on this small structural interface."""
    name: str
    milli_cpu_capacity: int
    milli_gpu_left_list: List[int]
    gpu_type: str = "V100"


def _small_cluster(n_nodes: int = 4, gpus_per_node: int = 4, milli_cpu: int = 32000) -> Topology:
    nodes = [_FakeNode(name=f"node-{i}", milli_cpu_capacity=milli_cpu,
                        milli_gpu_left_list=[MILLI] * gpus_per_node)
             for i in range(n_nodes)]
    return Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=2, nodes_per_rack=2)


# ---------------------------------------------------------------------------
# Topology construction
# ---------------------------------------------------------------------------

def test_from_nodes_vertex_count_matches_socket_fanout():
    topo = _small_cluster(n_nodes=4, gpus_per_node=4)
    # sockets_per_node=2, numa_per_socket=1 -> 2 vertices per physical node
    assert len(topo.vertices) == 4 * 2


def test_from_nodes_conserves_total_cpu_capacity():
    topo = _small_cluster(n_nodes=3, milli_cpu=32000)
    total = sum(v.milli_cpu_capacity for v in topo.vertices.values())
    assert total == 3 * 32000


def test_from_nodes_conserves_total_gpu_count():
    topo = _small_cluster(n_nodes=3, gpus_per_node=4)
    total_gpus = sum(len(v.milli_gpu_left_list) for v in topo.vertices.values())
    assert total_gpus == 3 * 4


def test_from_nodes_falls_back_to_memory_heuristic_when_absent():
    # _FakeNode has no memory_mib_capacity attribute at all
    topo = _small_cluster(n_nodes=1, milli_cpu=32000)
    mem_total = sum(v.memory_mib_capacity for v in topo.vertices.values())
    assert mem_total == 32000 * 8  # documented 8 MiB-per-milli-cpu fallback ratio


def test_load_topology_from_csv_reads_real_bundled_data():
    topo = load_topology_from_csv(nodes_per_rack=40)
    # 1523 real nodes (data/ORIGINAL_DATA_README.md), sockets_per_node=2 default
    assert len(topo.vertices) == 1523 * 2
    assert len(topo.edges) > 0


# ---------------------------------------------------------------------------
# HTAFMConfig
# ---------------------------------------------------------------------------

def test_htafm_config_accepts_every_documented_variant():
    for variant in VARIANTS:
        HTAFMConfig(variant=variant)  # must not raise


def test_htafm_config_rejects_unknown_variant():
    with pytest.raises(ValueError):
        HTAFMConfig(variant="not_a_real_variant")


# ---------------------------------------------------------------------------
# HTAFMScheduler.schedule()
# ---------------------------------------------------------------------------

def _demand(name, milli_cpu=1000, memory_mib=2000, milli_gpu=1000, gpu_number=1):
    return TopoDemand(milli_cpu=milli_cpu, memory_mib=memory_mib,
                       milli_gpu=milli_gpu, gpu_number=gpu_number, name=name)


@pytest.mark.parametrize("variant", VARIANTS)
def test_schedule_admits_all_when_capacity_sufficient(variant):
    topo = _small_cluster(n_nodes=4, gpus_per_node=4)
    demands = [_demand(f"d{i}") for i in range(5)]  # well within 4x4=16 GPUs of capacity
    sched = HTAFMScheduler(topo, HTAFMConfig(variant=variant, pending_demands=demands[:2]))
    result = sched.schedule(demands)
    assert result.unscheduled == []
    assert len(result.scheduled) == 5
    assert set(result.placement) == set(result.scheduled)


@pytest.mark.parametrize("variant", VARIANTS)
def test_schedule_rejects_demand_larger_than_any_vertex(variant):
    topo = _small_cluster(n_nodes=2, gpus_per_node=2, milli_cpu=8000)
    huge = _demand("too-big", milli_cpu=999_000, gpu_number=1)
    sched = HTAFMScheduler(topo, HTAFMConfig(variant=variant))
    result = sched.schedule([huge])
    assert result.unscheduled == ["too-big"]
    assert result.scheduled == []


@pytest.mark.parametrize("variant", VARIANTS)
def test_unallocated_gpus_decreases_after_scheduling(variant):
    topo = _small_cluster(n_nodes=4, gpus_per_node=4)  # 16 GPUs total
    sched = HTAFMScheduler(topo, HTAFMConfig(variant=variant))
    before = sched.unallocated_gpus()
    assert before == 16.0
    sched.schedule([_demand(f"d{i}", gpu_number=1) for i in range(6)])
    after = sched.unallocated_gpus()
    assert after == before - 6


def test_schedule_stops_gracefully_once_cluster_is_full():
    # exactly as many single-GPU demands as GPUs exist, plus one more
    topo = _small_cluster(n_nodes=2, gpus_per_node=2)  # 4 GPUs
    demands = [_demand(f"d{i}", gpu_number=1) for i in range(5)]
    sched = HTAFMScheduler(topo, HTAFMConfig(variant="entropy"))
    result = sched.schedule(demands)
    assert len(result.scheduled) == 4
    assert len(result.unscheduled) == 1
    assert sched.unallocated_gpus() == 0.0


def test_compute_tafi_cut_is_zero_with_no_pending_demands():
    # phi_cut counts pending demands that don't fit in an edge's aggregate
    # free capacity -- with an empty pending_demands list there's nothing
    # to fail to fit, so TAFI-Cut is trivially 0 regardless of placement
    # state. (TAFI-Entropy is NOT zero on an empty cluster -- it's the
    # Shannon entropy of the free-capacity distribution, which is *maximal*
    # on a uniform/empty cluster, not zero; only "cut" has this
    # empty-pending-demands-implies-zero property, so this test is
    # deliberately scoped to that one variant rather than asserting a
    # shared baseline across all three that doesn't actually hold.)
    topo = _small_cluster(n_nodes=3, gpus_per_node=4)
    score = compute_tafi(topo, HTAFMConfig(variant="cut", pending_demands=[]))
    assert score == 0.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_compute_tafi_is_always_finite(variant):
    topo = _small_cluster(n_nodes=3, gpus_per_node=4)
    cfg = HTAFMConfig(variant=variant, pending_demands=[_demand("probe", gpu_number=1)])
    sched = HTAFMScheduler(topo, cfg)
    sched.schedule([_demand("d0", gpu_number=2), _demand("d1", gpu_number=1)])
    score = compute_tafi(topo, cfg)
    assert math.isfinite(score)


def test_compute_tafi_cut_detects_true_fragmentation():
    # TAFI-Cut's definition (phi_cut): 1 if the *aggregate* free capacity
    # across an edge's vertices could satisfy a pending demand, but *no
    # single vertex* can -- i.e. the capacity exists but is split awkwardly
    # across machines, the textbook fragmentation case FGD/H-TAFM target.
    # Two 1-vertex nodes (sockets_per_node=1, numa_per_socket=1) with 2 GPUs
    # each, in the same rack (nodes_per_rack=2) so they share one rack-level
    # hyperedge: aggregate free = 4 GPUs, but neither vertex alone has more
    # than 2 -- a demand for 3 GPUs is exactly the "exists in aggregate,
    # unusable per-vertex" case this variant exists to catch.
    topo = Topology.from_nodes(
        [_FakeNode(name=f"node-{i}", milli_cpu_capacity=32000, milli_gpu_left_list=[MILLI] * 2)
         for i in range(2)],
        numa_per_socket=1, sockets_per_node=1, nodes_per_rack=2,
    )
    assert len(topo.vertices) == 2
    assert len(topo.edges) == 1  # one rack edge spanning both vertices

    fragmenting_demand = _demand("probe", gpu_number=3)   # fits in aggregate (4), not per-vertex (2 each)
    fitting_demand = _demand("easy", gpu_number=2)         # fits a single vertex outright

    assert compute_tafi(topo, HTAFMConfig(variant="cut", pending_demands=[fragmenting_demand])) == 1.0
    assert compute_tafi(topo, HTAFMConfig(variant="cut", pending_demands=[fitting_demand])) == 0.0
