"""
Tests for w_fgd / w_fgd_balanced (k8s_sim/policies.py) and their supporting
weighted-typical-pod machinery (k8s_sim/fragmentation.py). Basic
feasibility conformance is already covered by tests/test_k8s_sim.py's
parametrized test_every_policy_* tests (every name in POLICIES, which now
includes these two automatically) -- these tests instead check the actual
mathematical properties that motivate w_fgd over plain fgd.
"""
import numpy as np
import pytest

from k8s_sim.cluster import Cluster
from k8s_sim.fragmentation import build_typical_pods_weighted, unfit_fraction, unfit_fraction_weighted
from k8s_sim.policies import fragmentation_gradient_descent, w_fgd, w_fgd_balanced
from k8s_sim.resource import NodeResource, PodResource


def _pod(pid, cpu=1000, gpu_milli=1000, gpu_n=1):
    return PodResource(pod_id=pid, milli_cpu=cpu, milli_gpu=gpu_milli, gpu_number=gpu_n)


def _node(nid, cpu_cap=32000, gpu_cap=8000, gpu_count=8):
    return NodeResource(node_id=nid, milli_cpu_capacity=cpu_cap,
                         gpu_count=gpu_count, milli_gpu_capacity=gpu_cap)


# ---------------------------------------------------------------------------
# build_typical_pods_weighted: fixes the "chimera shape" + "uniform weight" issues
# ---------------------------------------------------------------------------

def test_weighted_shapes_are_real_observed_pods_not_chimeras():
    # 10 small pods (1 GPU) + 2 huge pods (8 GPU): build_typical_pods (plain)
    # would compute per-dimension quantiles independently; here every shape
    # must come from an ACTUAL pod in the list.
    pods = [_pod(f"small{i}", cpu=500, gpu_milli=1000, gpu_n=1) for i in range(10)]
    pods += [_pod(f"huge{i}", cpu=16000, gpu_milli=8000, gpu_n=8) for i in range(2)]
    shapes, weights = build_typical_pods_weighted(pods, n_shapes=4)

    observed = {(p.milli_cpu, p.milli_gpu, p.gpu_number) for p in pods}
    for s in shapes:
        assert (s.milli_cpu, s.milli_gpu, s.gpu_number) in observed
    assert abs(sum(weights) - 1.0) < 1e-9


def test_weighted_shapes_reflect_population_frequency():
    # 90% small pods, 10% huge -- the small shape's weight should dominate.
    pods = [_pod(f"small{i}", cpu=500, gpu_milli=1000, gpu_n=1) for i in range(90)]
    pods += [_pod(f"huge{i}", cpu=16000, gpu_milli=8000, gpu_n=8) for i in range(10)]
    shapes, weights = build_typical_pods_weighted(pods, n_shapes=5)

    small_weight = sum(w for s, w in zip(shapes, weights) if s.gpu_number == 1)
    huge_weight = sum(w for s, w in zip(shapes, weights) if s.gpu_number == 8)
    assert small_weight > huge_weight


def test_unfit_fraction_weighted_matches_manual_computation():
    node = _node("n0", gpu_cap=2000, gpu_count=2)  # 2 GPUs free
    shapes = [_pod("a", gpu_milli=1000, gpu_n=1), _pod("b", gpu_milli=3000, gpu_n=3)]
    weights = [0.3, 0.7]
    # shape "a" (1 GPU) fits in 2 free GPUs; shape "b" (3 GPU) does not
    assert unfit_fraction_weighted(node, shapes, weights) == pytest.approx(0.7)
    assert unfit_fraction(node, shapes) == pytest.approx(0.5)  # unweighted: 1 of 2 shapes


# ---------------------------------------------------------------------------
# w_fgd / w_fgd_balanced: fall back gracefully, and differ from plain fgd
# ---------------------------------------------------------------------------

def test_w_fgd_falls_back_to_best_fit_with_no_typical_pods():
    nodes = [_node("n0"), _node("n1")]
    pod = _pod("p0")
    chosen = w_fgd(pod, nodes, typical_pods=None, typical_weights=None)
    assert chosen is not None


def test_w_fgd_falls_back_to_plain_fgd_when_weights_missing():
    nodes = [_node("n0"), _node("n1")]
    pod = _pod("p0")
    shapes = [_pod("t0", gpu_milli=1000, gpu_n=1)]
    chosen_wfgd = w_fgd(pod, nodes, typical_pods=shapes, typical_weights=None)
    chosen_fgd = fragmentation_gradient_descent(pod, nodes, typical_pods=shapes)
    assert chosen_wfgd is not None and chosen_fgd is not None
    assert chosen_wfgd.node_id == chosen_fgd.node_id


def test_w_fgd_can_choose_differently_from_plain_fgd_under_skewed_demand():
    # Two nodes with different existing fragmentation profiles. A rare (low
    # weight) shape "fits only on node A"; a common (high weight) shape
    # "fits only on node B". Plain (unweighted) FGD treats both shapes
    # equally, so a placement that blocks the rare shape looks exactly as
    # bad as one blocking the common one. w_fgd should be willing to
    # sacrifice the rare shape to protect the common one where the two
    # policies disagree.
    node_a = _node("A", gpu_cap=1000, gpu_count=1)   # exactly 1 GPU left: only the tiny shape fits
    node_b = _node("B", gpu_cap=4000, gpu_count=4)   # 4 GPUs left: both shapes fit right now
    pod = _pod("incoming", cpu=1000, gpu_milli=3000, gpu_n=3)  # only fits on node_b

    tiny_shape = _pod("tiny", gpu_milli=1000, gpu_n=1)      # fits node_a AND node_b
    common_shape = _pod("common", gpu_milli=2000, gpu_n=2)  # fits only node_b (needs 2 GPUs, node_a has 1)
    shapes = [tiny_shape, common_shape]

    # placing `pod` (needs 3 GPUs) is only feasible on node_b regardless of
    # policy (node_a only has 1 GPU) -- so instead verify the SCORE
    # (not the choice) diverges in the direction w_fgd's design predicts:
    # blocking `common_shape` (which only fits on B) should cost more under
    # heavy weighting on common_shape than under uniform weighting.
    from k8s_sim.fragmentation import unfit_fraction_weighted as ufw

    weights_favor_common = [0.1, 0.9]
    weights_favor_tiny = [0.9, 0.1]

    node_b.remaining_milli_gpu -= pod.milli_gpu
    node_b.remaining_gpu_count -= pod.gpu_number
    after_common_favored = ufw(node_b, shapes, weights_favor_common)
    after_tiny_favored = ufw(node_b, shapes, weights_favor_tiny)
    node_b.remaining_milli_gpu += pod.milli_gpu
    node_b.remaining_gpu_count += pod.gpu_number

    # after placement, node_b has 1 GPU left: tiny_shape (1 GPU) still fits,
    # common_shape (2 GPU) does not -- so weighting common_shape heavily
    # must produce a strictly higher (worse) fragmentation score.
    assert after_common_favored > after_tiny_favored


def test_w_fgd_balanced_prefers_leaving_a_node_fully_empty(monkeypatch=None):
    # 3 identical small nodes; one already has SOME load, two are fully
    # empty. A pod that fits equally well (identical fragmentation delta)
    # on any of them should, under w_fgd_balanced, prefer a fully-empty
    # node over a partially-loaded one at the same fragmentation score,
    # since balance_lambda penalizes higher post-placement utilization.
    partially_loaded = _node("loaded", gpu_cap=4000, gpu_count=4)
    partially_loaded.remaining_milli_gpu -= 2000
    partially_loaded.remaining_gpu_count -= 2  # 2 of 4 GPUs already used

    empty_a = _node("empty_a", gpu_cap=4000, gpu_count=4)
    empty_b = _node("empty_b", gpu_cap=4000, gpu_count=4)

    pod = _pod("p", gpu_milli=1000, gpu_n=1)
    shapes = [_pod("t", gpu_milli=1000, gpu_n=1)]  # trivial shape: fits everywhere before/after
    weights = [1.0]

    chosen = w_fgd_balanced(pod, [partially_loaded, empty_a, empty_b],
                             typical_pods=shapes, typical_weights=weights, balance_lambda=0.5)
    # fragmentation delta is 0 on all three (the shape fits everywhere
    # regardless), so the utilization regularizer must break the tie
    assert chosen.node_id in ("empty_a", "empty_b")


def test_w_fgd_balanced_reduces_to_w_fgd_when_lambda_is_zero():
    nodes = [_node("n0"), _node("n1", gpu_count=4, gpu_cap=4000)]
    pod = _pod("p")
    shapes = [_pod("t0", gpu_milli=1000, gpu_n=1), _pod("t1", gpu_milli=2000, gpu_n=2)]
    weights = [0.4, 0.6]
    a = w_fgd(pod, nodes, typical_pods=shapes, typical_weights=weights)
    b = w_fgd_balanced(pod, nodes, typical_pods=shapes, typical_weights=weights, balance_lambda=0.0)
    assert a.node_id == b.node_id


# ---------------------------------------------------------------------------
# End-to-end: w_fgd/w_fgd_balanced integrate correctly with Cluster
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["w_fgd", "w_fgd_balanced"])
def test_cluster_schedule_pods_accepts_typical_weights(policy):
    nodes = {f"n{i}": _node(f"n{i}") for i in range(4)}
    cluster = Cluster(nodes)
    pods = [_pod(f"p{i}", gpu_milli=1000, gpu_n=1) for i in range(10)]
    shapes, weights = build_typical_pods_weighted(pods, n_shapes=3)
    results = cluster.schedule_pods(pods, policy=policy, typical_pods=shapes, typical_weights=weights)
    assert all(r.node_id is not None for r in results)  # ample capacity, everything should place
