"""
Scheduling policies. Every policy has the same signature:

    policy(pod, nodes, *, rng=None, typical_pods=None, round_robin_state=None) -> Optional[NodeResource]

returning the chosen node (already checked to fit — policies must only
choose among nodes where ``pod.fits_in(node)`` is True) or None if no node
fits. ``Cluster.schedule_pod`` handles actually calling ``node.add(pod)``.

Add a new policy by writing a function with this signature and adding it
to ``POLICIES`` at the bottom of this file.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional

import numpy as np

from .fragmentation import unfit_fraction
from .resource import NodeResource, PodResource

PolicyFn = Callable[..., Optional[NodeResource]]


def _feasible(pod: PodResource, nodes: List[NodeResource]) -> List[NodeResource]:
    return [n for n in nodes if pod.fits_in(n)]


def random_fit(pod: PodResource, nodes: List[NodeResource], *, rng=None, **_) -> Optional[NodeResource]:
    """Uniform-random choice among feasible nodes. The naive baseline
    every other policy should beat."""
    feasible = _feasible(pod, nodes)
    if not feasible:
        return None
    rng = rng or np.random.default_rng()
    return feasible[rng.integers(len(feasible))]


def first_fit(pod: PodResource, nodes: List[NodeResource], **_) -> Optional[NodeResource]:
    """First node (in list order) that fits. Cheap, but tends to
    concentrate load on early-indexed nodes and ignore the rest."""
    for n in nodes:
        if pod.fits_in(n):
            return n
    return None


def best_fit(pod: PodResource, nodes: List[NodeResource], **_) -> Optional[NodeResource]:
    """Minimizes leftover capacity after placement — packs tightly, at the
    cost of potentially leaving small unusable slivers on many nodes
    (classic bin-packing fragmentation risk)."""
    feasible = _feasible(pod, nodes)
    if not feasible:
        return None

    def leftover(n):
        return (n.remaining_milli_cpu - pod.milli_cpu) + (n.remaining_milli_gpu - pod.milli_gpu)

    return min(feasible, key=leftover)


def worst_fit(pod: PodResource, nodes: List[NodeResource], **_) -> Optional[NodeResource]:
    """Maximizes leftover capacity after placement — spreads load across
    nodes, keeping headroom on any one node but using more nodes overall."""
    feasible = _feasible(pod, nodes)
    if not feasible:
        return None

    def leftover(n):
        return (n.remaining_milli_cpu - pod.milli_cpu) + (n.remaining_milli_gpu - pod.milli_gpu)

    return max(feasible, key=leftover)


_round_robin_counters: Dict[int, itertools.count] = {}


def round_robin(pod: PodResource, nodes: List[NodeResource], *, round_robin_state=None, **_) -> Optional[NodeResource]:
    """Cycles through nodes in order, skipping ones that don't fit.
    ``round_robin_state`` should be a single-element list (mutable cursor)
    shared across calls within one simulation run — see
    ``Cluster.schedule_pod`` for how this is threaded through."""
    feasible = _feasible(pod, nodes)
    if not feasible:
        return None
    if round_robin_state is None:
        round_robin_state = [0]
    idx = round_robin_state[0] % len(nodes)
    for offset in range(len(nodes)):
        candidate = nodes[(idx + offset) % len(nodes)]
        if pod.fits_in(candidate):
            round_robin_state[0] = (idx + offset + 1) % len(nodes)
            return candidate
    return None


def gpu_packing(pod: PodResource, nodes: List[NodeResource], **_) -> Optional[NodeResource]:
    """Prefers the most-already-utilized feasible node (by GPU
    utilization), to consolidate GPU usage onto fewer nodes and keep
    others fully free for large/exclusive requests."""
    feasible = _feasible(pod, nodes)
    if not feasible:
        return None
    return max(feasible, key=lambda n: n.gpu_utilization())


def least_requested(pod: PodResource, nodes: List[NodeResource], **_) -> Optional[NodeResource]:
    """Prefers the least-utilized feasible node (by combined CPU+GPU
    utilization) — spreads load to minimize any single node becoming a
    hotspot; the standard Kubernetes default-scheduler-style heuristic."""
    feasible = _feasible(pod, nodes)
    if not feasible:
        return None
    return min(feasible, key=lambda n: n.cpu_utilization() + n.gpu_utilization())


def fragmentation_gradient_descent(
    pod: PodResource, nodes: List[NodeResource], *, typical_pods: Optional[List[PodResource]] = None, **_
) -> Optional[NodeResource]:
    """Chooses the feasible node whose fragmentation score (see
    ``fragmentation.py``) increases the *least* if ``pod`` is placed there
    — i.e. picks the placement that preserves the cluster's ability to fit
    other typically-shaped pods, rather than just minimizing raw leftover
    bytes. Falls back to ``best_fit`` if no typical-pod set is given (the
    score is undefined without one).
    """
    feasible = _feasible(pod, nodes)
    if not feasible:
        return None
    if not typical_pods:
        return best_fit(pod, nodes)

    def score_after_placement(n: NodeResource) -> float:
        before = unfit_fraction(n, typical_pods)
        # simulate placement without mutating: temporarily adjust remaining
        n.remaining_milli_cpu -= pod.milli_cpu
        n.remaining_milli_gpu -= pod.milli_gpu
        n.remaining_gpu_count -= pod.gpu_number
        after = unfit_fraction(n, typical_pods)
        n.remaining_milli_cpu += pod.milli_cpu
        n.remaining_milli_gpu += pod.milli_gpu
        n.remaining_gpu_count += pod.gpu_number
        return after - before

    return min(feasible, key=score_after_placement)


POLICIES: Dict[str, PolicyFn] = {
    "random": random_fit,
    "first_fit": first_fit,
    "best_fit": best_fit,
    "worst_fit": worst_fit,
    "round_robin": round_robin,
    "gpu_packing": gpu_packing,
    "least_requested": least_requested,
    "fgd": fragmentation_gradient_descent,
}


def list_policies() -> List[str]:
    return sorted(POLICIES)


def get_policy(name: str) -> PolicyFn:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; available: {list_policies()}")
    return POLICIES[name]
