import math

import pytest

from k8s_sim.topology import Topology, TopoVertex, HyperEdge, TopoDemand
from k8s_sim.htafm import (
    phi_cut, phi_entropy, phi_hier, compute_tafi, delta_tafi,
    HTAFMConfig, HTAFMScheduler,
)


def _two_vertex_server(v1_free_cpu, v2_free_cpu, cap=10000):
    v1 = TopoVertex(id="v1", node_name="n0", socket_id="s0", rack_id="r0",
                     milli_cpu_capacity=cap, milli_cpu_left=v1_free_cpu,
                     memory_mib_capacity=cap * 10, memory_mib_left=v1_free_cpu * 10,
                     milli_gpu_left_list=[], gpu_type="")
    v2 = TopoVertex(id="v2", node_name="n0", socket_id="s0", rack_id="r0",
                     milli_cpu_capacity=cap, milli_cpu_left=v2_free_cpu,
                     memory_mib_capacity=cap * 10, memory_mib_left=v2_free_cpu * 10,
                     milli_gpu_left_list=[], gpu_type="")
    edge = HyperEdge(id="n0", level="server", weight=4.0, vertex_ids=["v1", "v2"])
    return Topology({"v1": v1, "v2": v2}, {"n0": edge}), edge


# ---------------------------------------------------------------------------
# TAFI-Cut
# ---------------------------------------------------------------------------

def test_cut_zero_when_single_vertex_can_host():
    topo, edge = _two_vertex_server(9000, 1000)
    d = TopoDemand(milli_cpu=1500, memory_mib=15000)
    assert phi_cut(topo, edge, [d]) == 0  # v1 alone (9000 free) can host it


def test_cut_one_when_only_aggregate_can_host():
    topo, edge = _two_vertex_server(9000, 1000)
    d = TopoDemand(milli_cpu=9500, memory_mib=95000)  # <= 10000 aggregate, > either vertex alone
    assert phi_cut(topo, edge, [d]) == 1


def test_cut_zero_when_not_even_aggregate_can_host():
    topo, edge = _two_vertex_server(9000, 1000)
    d = TopoDemand(milli_cpu=99999, memory_mib=999999)
    assert phi_cut(topo, edge, [d]) == 0  # Condition A fails


# ---------------------------------------------------------------------------
# TAFI-Entropy
# ---------------------------------------------------------------------------

def test_entropy_zero_when_all_free_on_one_vertex():
    topo, edge = _two_vertex_server(10000, 0)
    assert phi_entropy(topo, edge) == pytest.approx(0.0, abs=1e-9)


def test_entropy_maximal_when_balanced():
    topo, edge = _two_vertex_server(5000, 5000)
    assert phi_entropy(topo, edge) == pytest.approx(math.log(2), abs=1e-9)


def test_entropy_between_extremes_is_between_bounds():
    topo, edge = _two_vertex_server(7000, 3000)
    val = phi_entropy(topo, edge)
    assert 0.0 < val < math.log(2)


# ---------------------------------------------------------------------------
# TAFI-Hier
# ---------------------------------------------------------------------------

def test_hier_counts_vertices_below_threshold():
    # v1: 90% free (not fragmented @ 50% threshold), v2: 10% free (fragmented)
    topo, edge = _two_vertex_server(9000, 1000)
    val = phi_hier(topo, edge, threshold_frac=0.5, level_lambda={"server": 1.0})
    assert val == pytest.approx(0.5, abs=1e-9)  # 1 of 2 vertices fragmented


def test_hier_zero_when_all_above_threshold():
    topo, edge = _two_vertex_server(9000, 8000)
    val = phi_hier(topo, edge, threshold_frac=0.5)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_hier_max_when_all_below_threshold():
    topo, edge = _two_vertex_server(1000, 2000)
    val = phi_hier(topo, edge, threshold_frac=0.5, level_lambda={"server": 1.0})
    assert val == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# delta_tafi correctness: full recompute vs. incremental should agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["cut", "entropy", "hier"])
def test_delta_tafi_matches_full_recompute(variant):
    topo, edge = _two_vertex_server(9000, 1000)
    cfg = HTAFMConfig(variant=variant, pending_demands=[TopoDemand(milli_cpu=500, memory_mib=5000)])
    d = TopoDemand(milli_cpu=1000, memory_mib=10000)

    before = compute_tafi(topo, cfg)
    incremental = delta_tafi(topo, "v1", d, cfg)

    topo_after = topo.copy()
    topo_after.vertices["v1"].place(d)
    after = compute_tafi(topo_after, cfg)

    assert incremental == pytest.approx(after - before, abs=1e-9)


# ---------------------------------------------------------------------------
# Monotonicity (Section 8's claimed theoretical property)
# ---------------------------------------------------------------------------

def _run_and_check_monotonic(variant, n_placements=20):
    from k8s_sim.topology import _TraceNode, MILLI
    nodes = [_TraceNode(name=f"n{i}", milli_cpu_capacity=32000, memory_mib_capacity=131072,
                         milli_gpu_left_list=[MILLI] * 4, gpu_type="V100") for i in range(10)]
    topo = Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=2, nodes_per_rack=5)

    if variant == "cut":
        cfg = HTAFMConfig(variant="cut", pending_demands=[
            TopoDemand(milli_cpu=4000, memory_mib=8192, milli_gpu=500, gpu_number=1)])
    else:
        cfg = HTAFMConfig(variant=variant)

    sched = HTAFMScheduler(topo, cfg)
    prev = compute_tafi(topo, cfg)
    violations = []
    for i in range(n_placements):
        d = TopoDemand(milli_cpu=2000, memory_mib=4096, milli_gpu=500, gpu_number=1, name=f"d{i}")
        sched.schedule_one(d)
        cur = compute_tafi(topo, cfg)
        if cur < prev - 1e-9:
            violations.append(i)
        prev = cur
    return violations


def test_cut_variant_is_monotonic():
    """Section 8 claims TAFI is non-decreasing as VMs are added. Verified
    for TAFI-Cut: once demand no longer fits a single vertex, consuming more
    capacity can't make it fit again."""
    assert _run_and_check_monotonic("cut") == []


def test_hier_variant_is_monotonic():
    """Verified for TAFI-Hier: a vertex crossing below threshold stays below
    threshold as more capacity is consumed (thresholds/lambdas held fixed)."""
    assert _run_and_check_monotonic("hier") == []


def test_entropy_variant_violates_claimed_monotonicity():
    """IMPORTANT FINDING, not a bug: Section 8's monotonicity proof does not
    hold for TAFI-Entropy. Shannon entropy of the free-resource distribution
    can legitimately increase OR decrease as capacity is consumed, depending
    on whether the placement moves the distribution toward or away from
    uniform. This test pins down that (expected) behavior empirically so a
    future change to phi_entropy that "fixes" this doesn't silently mask
    the fact that the underlying math genuinely isn't monotonic -- see
    docs/HTAFM.md for the full writeup and a worked example."""
    violations = _run_and_check_monotonic("entropy")
    assert len(violations) > 0
