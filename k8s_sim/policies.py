"""
Port of the six scheduling scoring plugins in pkg/simulator/plugin/*.go.

Every policy is a function of the form:

    score(node: NodeResource, pod: PodResource, ctx: dict) -> float in [0, 100]

where higher is better (matches the k8s scheduler framework convention:
framework.MaxNodeScore == 100, framework.MinNodeScore == 0). `ctx` carries
whatever side information a given policy needs (e.g. FGD needs the typical
pod distribution).

These operate purely on the resource model in resource.py; the real
project's node-affinity/taint/toleration filtering, GPU-annotation parsing,
and live k8s API plumbing are intentionally left out — see README.md.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence

from .resource import NodeResource, PodResource, MAX_NODE_SCORE, MIN_NODE_SCORE, MAX_SPEC_CPU, MAX_SPEC_GPU, MILLI
from .fragmentation import TargetPod, node_gpu_share_frag_amount_score


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# 1. Random
# ---------------------------------------------------------------------------

def random_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    """Mirrors plugin.RandomScorePlugin: pick one feasible node uniformly at
    random for the whole pod (not per-node-independent). `ctx["_rng_pick"]`
    is set once per pod by the cluster driver."""
    return MAX_NODE_SCORE if node.name == ctx.get("_rng_pick") else MIN_NODE_SCORE


def random_prepare(nodes: Sequence[NodeResource], pod: PodResource, ctx: dict) -> None:
    ctx["_rng_pick"] = random.choice(nodes).name if nodes else None


# ---------------------------------------------------------------------------
# 2. Best-Fit
# ---------------------------------------------------------------------------

def best_fit_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    """Mirrors plugin.getBestFitScore: minimize leftover (free - request),
    normalized by a fixed max spec and weighted 50/50 between CPU and GPU,
    then flipped so smaller leftover -> higher score."""
    free_vec = node.to_resource_vec()
    req_vec = pod.to_resource_vec()
    max_spec = [MAX_SPEC_CPU, MAX_SPEC_GPU]
    weights = [0.5, 0.5]

    score = 0.0
    for f, r, m, w in zip(free_vec, req_vec, max_spec, weights):
        if f < r:
            return MIN_NODE_SCORE  # shouldn't happen if filtered correctly
        score += (f - r) / m * w
    return (1.0 - score) * MAX_NODE_SCORE


# ---------------------------------------------------------------------------
# 3. Dot-Product (Tetris-style packing heuristic)
# ---------------------------------------------------------------------------

def dot_product_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    """Mirrors plugin.calculateDotProductScore using the simplified
    "merge GPU dims" resource vectors [cpu, total_gpu] and NormByPod-style
    tanh normalization (favor nodes whose *residual* capacity vector aligns
    least with the pod's request vector, i.e. avoid stranding resources)."""
    node_vec = node.to_resource_vec()
    pod_vec = pod.to_resource_vec()
    dot = sum(n * p for n, p in zip(node_vec, pod_vec))
    if dot < 0:
        return MIN_NODE_SCORE
    normalized = dot / len(pod_vec)
    normalized = math.tanh(normalized / 10.0)  # squashed into [0, 1)
    # larger dot product (pod aligned with node's *largest* free dims) is
    # worse (it strands the other dimension), so invert:
    return (1.0 - normalized) * MAX_NODE_SCORE


# ---------------------------------------------------------------------------
# 4. GPU-Packing
# ---------------------------------------------------------------------------

def gpu_packing_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    """Mirrors plugin.getPackingScore. Prefers, in order:
      1. packing onto already-partially-used (shared) GPUs (score in [50,100])
      2. using free GPUs on an otherwise-used node (score in [33,50])
      3. using free GPUs on a fully-free node (score in [0,33])
    i.e. it actively tries to consolidate GPU usage rather than spread it."""
    if pod.gpu_number == 0:
        return MIN_NODE_SCORE

    fully_free = node.fully_free_gpu_num()
    if fully_free == node.gpu_number:
        # case 3: entire node is free
        score = MAX_NODE_SCORE / 3 - fully_free
        return max(score, fully_free)  # matches the (slightly odd) Go cap

    order = node.sorted_gpu_left_index_list(ascending=True)  # least-left first
    need = pod.gpu_number
    used: List[int] = []
    fully_free_used = 0
    for idx in order:
        if need == 0:
            break
        left = node.milli_gpu_left_list[idx]
        if pod.milli_gpu <= left:
            need -= 1
            used.append(idx)
            if left == MILLI:
                fully_free_used += 1
    if need != 0:
        return MIN_NODE_SCORE  # infeasible, shouldn't happen if pre-filtered

    if fully_free_used > 0:
        # case 2: had to dip into some fully-free GPUs
        score = MAX_NODE_SCORE / 2 - fully_free_used
        return max(score, MAX_NODE_SCORE / 3)

    # case 1: fits entirely on already-shared GPUs
    free_ratio_on_used = sum(node.milli_gpu_left_list[i] * 100 // MILLI for i in used)
    score = MAX_NODE_SCORE - free_ratio_on_used / 10
    return max(score, MAX_NODE_SCORE / 2)


# ---------------------------------------------------------------------------
# 5. GPU-Clustering
# ---------------------------------------------------------------------------
# The original keys "affinity" off a GPU-type tag baked into pod annotations
# (gpushareutils.GetGpuAffinityFromPodAnnotation). Here we key it off
# `pod.gpu_type` (or an explicit `affinity_key` in ctx) — same idea: try to
# co-locate pods that share an affinity tag onto the same node(s), and keep
# nodes "pure" (single affinity) where possible.

def gpu_clustering_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    if pod.gpu_number == 0:
        return MIN_NODE_SCORE

    affinity_key = ctx.get("affinity_key") or pod.gpu_type or "default"
    used_frac = (MAX_SPEC_GPU - node.total_milli_gpu_left()) / MAX_SPEC_GPU
    base = MAX_NODE_SCORE / 4 * used_frac

    if node.gpu_affinity.get(affinity_key, 0) > 0:
        if len(node.gpu_affinity) == 1:
            return base + MAX_NODE_SCORE * 3 / 4      # node already pure to this affinity
        else:
            return base + MAX_NODE_SCORE * 2 / 4       # node has this affinity among others
    else:
        if len(node.gpu_affinity) == 0:
            return base + MAX_NODE_SCORE / 4            # idle node, no affinity yet
        else:
            return base                                  # node has *different* affinities only


# ---------------------------------------------------------------------------
# 6. FGD (Fragmentation Gradient Descent)
# ---------------------------------------------------------------------------

def fgd_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    """Mirrors plugin.calculateGpuShareFragExtendScore: score a placement by
    how much it *reduces* the node's fragmentation score (gradient descent on
    fragmentation), squashed through a sigmoid into [0, 100].

    ctx must contain "typical_pods": Sequence[TargetPod].
    """
    typical_pods: Sequence[TargetPod] = ctx["typical_pods"]
    before = node_gpu_share_frag_amount_score(node, typical_pods)

    if pod.gpu_number == 1 and pod.milli_gpu < MILLI and node.gpu_number > 0:
        # fractional-GPU pod: try every GPU device, keep the best
        best = None
        for i, left in enumerate(node.milli_gpu_left_list):
            if left < pod.milli_gpu:
                continue
            trial = node.copy()
            trial.milli_cpu_left -= pod.milli_cpu
            trial.milli_gpu_left_list[i] -= pod.milli_gpu
            after = node_gpu_share_frag_amount_score(trial, typical_pods)
            frag_score = _sigmoid((before - after) / 1000.0) * MAX_NODE_SCORE
            if best is None or frag_score > best:
                best = frag_score
        return best if best is not None else MIN_NODE_SCORE
    else:
        trial = node.sub(pod)
        after = node_gpu_share_frag_amount_score(trial, typical_pods)
        return _sigmoid((before - after) / 1000.0) * MAX_NODE_SCORE


# ---------------------------------------------------------------------------

POLICIES: Dict[str, Callable[[NodeResource, PodResource, dict], float]] = {
    "random": random_score,
    "best-fit": best_fit_score,
    "dot-product": dot_product_score,
    "gpu-packing": gpu_packing_score,
    "gpu-clustering": gpu_clustering_score,
    "fgd": fgd_score,
}

# Policies that need a one-time per-pod "prepare" step before scoring nodes
# (mirrors the framework's PreScore extension point).
PREPARE_HOOKS: Dict[str, Callable[[Sequence[NodeResource], PodResource, dict], None]] = {
    "random": random_prepare,
}
