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
# 1b. First-Fit
# ---------------------------------------------------------------------------
# Classic online bin-packing heuristic: place each item in the first bin
# (here: node, in cluster-list order) that has room for it.
# Reference: D. S. Johnson, "Fast Algorithms for Bin Packing,"
# J. Comput. Syst. Sci., 8(3):272-314, 1974 (and his 1973 MIT PhD thesis,
# where First-Fit and First-Fit-Decreasing were introduced/analyzed).

def first_fit_prepare(nodes: Sequence[NodeResource], pod: PodResource, ctx: dict) -> None:
    ctx["_ff_order"] = {n.name: i for i, n in enumerate(nodes)}


def first_fit_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    order = ctx.get("_ff_order", {})
    idx = order.get(node.name, 0)
    return max(MAX_NODE_SCORE - idx, MIN_NODE_SCORE)


# ---------------------------------------------------------------------------
# 1c. Worst-Fit
# ---------------------------------------------------------------------------
# The mirror image of Best-Fit: place each item in the bin that will have
# the MOST room left over, spreading load rather than consolidating it.
# Reference: E. G. Coffman Jr., M. R. Garey, D. S. Johnson,
# "Approximation Algorithms for Bin Packing: A Survey," in Approximation
# Algorithms for NP-Hard Problems, PWS Publishing, 1996 (surveys First-Fit,
# Best-Fit, Worst-Fit and their decreasing variants).

def worst_fit_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    free_vec = node.to_resource_vec()
    req_vec = pod.to_resource_vec()
    max_spec = [MAX_SPEC_CPU, MAX_SPEC_GPU]
    weights = [0.5, 0.5]

    score = 0.0
    for f, r, m, w in zip(free_vec, req_vec, max_spec, weights):
        if f < r:
            return MIN_NODE_SCORE
        score += (f - r) / m * w
    return min(score * MAX_NODE_SCORE, MAX_NODE_SCORE)  # larger leftover -> higher score


# ---------------------------------------------------------------------------
# 1d. Round-Robin
# ---------------------------------------------------------------------------
# Cycle through feasible nodes in order, one pod per node per turn, rather
# than scoring by resource state at all. A standard fairness/simplicity
# baseline in scheduling theory (e.g. round-robin CPU scheduling, Kleinrock's
# time-sharing analysis) and a common naive baseline in cluster-scheduler
# evaluations (e.g. used alongside Random in Grandl et al., "Multi-Resource
# Packing for Cluster Schedulers," SIGCOMM 2014).
#
# Needs persistent state ACROSS pod placements (not just within one pod's
# scoring call), so it reads/writes ctx["_cluster_state"], which
# Cluster.schedule_pod populates from a dict that lives on the Cluster
# instance itself and survives across calls.

def round_robin_prepare(nodes: Sequence[NodeResource], pod: PodResource, ctx: dict) -> None:
    state = ctx.setdefault("_cluster_state", {})
    counter = state.get("rr_counter", 0)
    if nodes:
        ctx["_rr_pick"] = nodes[counter % len(nodes)].name
        state["rr_counter"] = counter + 1
    else:
        ctx["_rr_pick"] = None


def round_robin_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    return MAX_NODE_SCORE if node.name == ctx.get("_rr_pick") else MIN_NODE_SCORE


# ---------------------------------------------------------------------------
# 1e. DRF-inspired (Dominant Resource Fairness)
# ---------------------------------------------------------------------------
# DRF equalizes each tenant's *dominant share* (their largest fractional
# share of any resource) across a shared cluster. We repurpose the same
# principle at placement time as a load-balancing heuristic: prefer the node
# whose dominant resource share (max across CPU/GPU of used/capacity) would
# end up LOWEST after placing the pod, i.e. keep every node's most
# constrained resource as balanced as possible across the cluster.
# Reference: A. Ghodsi, M. Zaharia, B. Hindman, A. Konwinski, S. Shenker,
# I. Stoica, "Dominant Resource Fairness: Fair Allocation of Multiple
# Resource Types," NSDI 2011.

def drf_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    cpu_used_after = (node.milli_cpu_capacity - node.milli_cpu_left) + pod.milli_cpu
    cpu_share = cpu_used_after / node.milli_cpu_capacity if node.milli_cpu_capacity else 0.0

    if node.gpu_number > 0:
        gpu_cap = node.gpu_number * MILLI
        gpu_used_after = (gpu_cap - node.total_milli_gpu_left()) + pod.total_milli_gpu()
        gpu_share = gpu_used_after / gpu_cap if gpu_cap else 0.0
    else:
        gpu_share = 0.0

    dominant_share = max(cpu_share, gpu_share)
    return max(0.0, (1.0 - dominant_share)) * MAX_NODE_SCORE


# ---------------------------------------------------------------------------
# 1f. Least-Requested-Priority (Kubernetes default scheduler baseline)
# ---------------------------------------------------------------------------
# The default kube-scheduler strategy: prefer the node with the most free
# capacity *proportionally*, spreading pods across the cluster rather than
# packing them. This is the LeastAllocated strategy of the built-in
# NodeResourcesFit score plugin.
# Reference: Kubernetes documentation, "Scheduler Configuration - Scheduling
# Plugins / NodeResourcesFit," https://kubernetes.io/docs/reference/scheduling/config/#scheduling-plugins

def least_requested_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    cpu_free_frac = (node.milli_cpu_left - pod.milli_cpu) / node.milli_cpu_capacity \
        if node.milli_cpu_capacity else 0.0
    if node.gpu_number > 0:
        gpu_cap = node.gpu_number * MILLI
        gpu_free_frac = (node.total_milli_gpu_left() - pod.total_milli_gpu()) / gpu_cap if gpu_cap else 0.0
        avg_free_frac = (cpu_free_frac + gpu_free_frac) / 2.0
    else:
        avg_free_frac = cpu_free_frac
    return max(0.0, min(1.0, avg_free_frac)) * MAX_NODE_SCORE


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
    "first-fit": first_fit_score,
    "worst-fit": worst_fit_score,
    "round-robin": round_robin_score,
    "best-fit": best_fit_score,
    "dot-product": dot_product_score,
    "drf": drf_score,
    "least-requested": least_requested_score,
    "gpu-packing": gpu_packing_score,
    "gpu-clustering": gpu_clustering_score,
    "fgd": fgd_score,
}

# Policies that need a one-time per-pod "prepare" step before scoring nodes
# (mirrors the framework's PreScore extension point).
PREPARE_HOOKS: Dict[str, Callable[[Sequence[NodeResource], PodResource, dict], None]] = {
    "random": random_prepare,
    "first-fit": first_fit_prepare,
    "round-robin": round_robin_prepare,
}

# ==================== HTAFM integration ====================
from .htafm import Topology, TopoDemand, HTAFMScheduler, HTAFMConfig
from .topology import TopoVertex
from .resource import NodeResource, PodResource

def _build_topology_from_nodes(nodes: List[NodeResource]) -> Topology:
    """Convert a list of NodeResource objects to a HTAFM Topology.
       Returns a topology with vertices as a dict (id -> vertex) and no hyperedges."""
    vertices_dict = {}
    for node in nodes:
        # ---- CPU ----
        milli_cpu_cap = getattr(node, 'milli_cpu_capacity', 0)

        # ---- Memory ----
        memory_mib_cap = getattr(node, 'memory_mib_capacity', None)
        if memory_mib_cap is None:
            memory_mib_cap = getattr(node, 'memory_capacity_mib', 0)

        # ---- GPU per‑device capacity ----
        gpu_count = getattr(node, 'gpu_number', 0)
        if gpu_count > 0:
            per_card_milli = getattr(node, 'gpu_capacity_per_card_milli', None)
            if per_card_milli is None:
                total_milli = getattr(node, 'milli_gpu_capacity', None)
                if total_milli is not None:
                    per_card_milli = total_milli // gpu_count
                else:
                    per_card_milli = 1000   # default: 1 GPU = 1000 milli
        else:
            per_card_milli = 0
        milli_gpu_left_list = [per_card_milli] * gpu_count

        vertex = TopoVertex(
            id=node.name,
            node_name=node.name,
            socket_id="",
            rack_id="",
            milli_cpu_capacity=milli_cpu_cap,
            milli_cpu_left=milli_cpu_cap,
            memory_mib_capacity=memory_mib_cap,
            memory_mib_left=memory_mib_cap,
            milli_gpu_left_list=milli_gpu_left_list,
            gpu_type=getattr(node, 'gpu_type', '')
        )
        # Store in dict with id as key
        vertices_dict[vertex.id] = vertex

    # Return Topology with dict of vertices and empty dict for edges
    return Topology(vertices_dict, {})

def htafm_scorer(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    """Dummy scorer: HTAFM does its own scoring internally."""
    return MAX_NODE_SCORE   # maximum score, so it's always eligible

def htafm_schedule(nodes: List[NodeResource], pods: List[PodResource],
                   typical_pods=None):
    # Build topology
    topology = _build_topology_from_nodes(nodes)
    config = HTAFMConfig(variant="cut")
    scheduler = HTAFMScheduler(topology, config)

    # Convert pods to TopoDemand
    demands = []
    for pod in pods:
        cpu_milli = getattr(pod, 'milli_cpu_request', getattr(pod, 'milli_cpu', 0))
        mem_mib = getattr(pod, 'memory_request_mib', getattr(pod, 'memory_mib', 0))
        gpu_num = getattr(pod, 'gpu_number', 0)
        if gpu_num > 0:
            milli_per_gpu = getattr(pod, 'milli_gpu_per_card', None)
            if milli_per_gpu is None:
                total_milli = getattr(pod, 'milli_gpu_request', None)
                if total_milli is not None:
                    milli_per_gpu = total_milli // gpu_num
                else:
                    milli_per_gpu = 1000
        else:
            milli_per_gpu = 0

        demand = TopoDemand(
            name=pod.name,
            milli_cpu=cpu_milli,
            memory_mib=mem_mib,
            milli_gpu=milli_per_gpu,
            gpu_number=gpu_num,
            gpu_type=getattr(pod, 'gpu_type', '')
        )
        demands.append(demand)

    result = scheduler.schedule(demands)

    # Map vertex ID (which is node.name) back to node name
    vertex_to_node = {v.id: v.id for v in topology.vertices.values()}

    scheduled = []
    unscheduled = list(result.unscheduled)
    for name in result.scheduled:
        vertex_id = result.placement.get(name)
        if vertex_id is not None:
            node_name = vertex_to_node.get(vertex_id)
            if node_name:
                scheduled.append((name, node_name))
            else:
                unscheduled.append(name)
        else:
            unscheduled.append(name)

    unscheduled = list(set(unscheduled))

    class HTAFMScheduleResult:
        def __init__(self, scheduled, unscheduled):
            self.scheduled = scheduled
            self.unscheduled = unscheduled

    return HTAFMScheduleResult(scheduled, unscheduled)

# Register HTAFM policy (score function is dummy; actual scheduling is intercepted)
POLICIES["htafm"] = htafm_scorer