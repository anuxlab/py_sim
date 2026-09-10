"""
H-TAFM: Hypergraph-based Topology-Aware Fragmentation Metric.

Implements the three TAFI variants (Cut, Entropy, Hierarchical) from the
methodology and the gradient-descent scheduler (Algorithm 5) on top of the
Topology model in topology.py.

Lineage / citations (as given in the source methodology document):
  - Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads
    with Fragmentation Gradient Descent," USENIX ATC 2023 -- H-TAFM extends
    FGD's gradient-descent placement principle from a flat GPU-only metric
    to a topology-aware, multi-resource one.
  - Mishra & Bellur, "De-Fragmenting the Cloud," arXiv:1506.07020, 2015 --
    multi-dimensional fragmentation measurement predates FGD; H-TAFM's
    Cut/Entropy variants formalize this with provable properties (Sec. 8).
  - Wang & Wu, "Survey on Hypergraph Algorithms... in Cloud Computing,"
    J. Comput. Sci. Technol. 41(2), 2026 -- hypergraphs for multi-resource
    VM placement.
  - Papp, Anegg, Yzelman, "Partitioning Hypergraphs is Hard," SPAA 2023 --
    motivates the hierarchical NUMA/Socket/Server/Rack weighting.
  - Christensen et al., "Multidimensional Bin Packing and Other Related
    Problems: A Survey," 2016 -- vector bin-packing framing of VM placement.

RESOLVED AMBIGUITIES (documented per docs/HTAFM.md):
  1. NUMA-level hyperedges are omitted as vertex-granularity singletons
     (see topology.py's module docstring) -- this is a decoding decision,
     not a missing feature.
  2. TAFI-Hier's formula sums lambda_level-weighted frag_level ACROSS ALL
     LEVELS *inside* phi(e, Pi), which would double-count against the outer
     Sum_{e in E} w(e) that already iterates per-level edges. We resolve
     this by having phi_hier(e) use only e's own level's threshold/lambda,
     letting the outer sum-over-all-edges perform the "aggregate across
     levels" the inner formula also describes. This avoids double-counting
     while preserving the stated per-level threshold/lambda design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .topology import Topology, TopoVertex, TopoDemand, HyperEdge

VARIANTS = ("cut", "entropy", "hier")

DEFAULT_LEVEL_THRESHOLD_FRAC = 0.5   # Sec. 3, Variant 3: "a VM requiring 50% of capacity"
DEFAULT_LEVEL_LAMBDA = {"socket": 1.0, "server": 1.0, "rack": 1.0}


def _free_of(v: TopoVertex, override: Optional[Dict[str, Tuple[float, float, float]]]) -> Tuple[float, float, float]:
    if override is not None and v.id in override:
        return override[v.id]
    return v.free_vector()


def _demand_leq(d: TopoDemand, agg: Tuple[float, float, float]) -> bool:
    return d.milli_cpu <= agg[0] and d.memory_mib <= agg[1] and d.total_milli_gpu() <= agg[2]


def _vertex_fits_free(free: Tuple[float, float, float], cap_gpu_devices: int, d: TopoDemand) -> bool:
    """Approximates TopoVertex.fits() using only the aggregated free vector
    (cpu, memory, total_gpu_milli) -- used inside phi_cut's Condition B,
    which per the methodology checks against a vertex's free CAPACITY
    (a scalar/vector comparison), not its per-device GPU packing. Per-device
    packing nuances are handled by the scheduler's actual placement step
    (TopoVertex.fits), not by the fragmentation metric itself."""
    return d.milli_cpu <= free[0] and d.memory_mib <= free[1] and d.total_milli_gpu() <= free[2]


# ---------------------------------------------------------------------------
# Variant 1: TAFI-Cut
# ---------------------------------------------------------------------------

def phi_cut(topology: Topology, edge: HyperEdge, pending_demands: Sequence[TopoDemand],
            override: Optional[Dict[str, Tuple[float, float, float]]] = None) -> int:
    verts = [topology.vertices[vid] for vid in edge.vertex_ids]
    frees = [_free_of(v, override) for v in verts]
    F_e = (sum(f[0] for f in frees), sum(f[1] for f in frees), sum(f[2] for f in frees))

    for d in pending_demands:
        if not _demand_leq(d, F_e):
            continue  # Condition A fails: not even the aggregate can satisfy it
        # Condition B: no single vertex individually satisfies it
        if all(not _vertex_fits_free(f, 0, d) for f in frees):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Variant 2: TAFI-Entropy
# ---------------------------------------------------------------------------

def phi_entropy(topology: Topology, edge: HyperEdge,
                 override: Optional[Dict[str, Tuple[float, float, float]]] = None) -> float:
    verts = [topology.vertices[vid] for vid in edge.vertex_ids]
    free_norm = []
    for v in verts:
        free = _free_of(v, override)
        cap = v.capacity_vector()
        s = 0.0
        for f, c in zip(free, cap):
            if c > 0:
                s += f / c
        free_norm.append(s)

    total = sum(free_norm)
    if total <= 0:
        return 0.0
    p = [f / total for f in free_norm if f > 0]
    return -sum(pi * math.log(pi) for pi in p)


# ---------------------------------------------------------------------------
# Variant 3: TAFI-Hier
# ---------------------------------------------------------------------------

def phi_hier(topology: Topology, edge: HyperEdge,
              threshold_frac: float = DEFAULT_LEVEL_THRESHOLD_FRAC,
              level_lambda: Optional[Dict[str, float]] = None,
              override: Optional[Dict[str, Tuple[float, float, float]]] = None) -> float:
    level_lambda = level_lambda or DEFAULT_LEVEL_LAMBDA
    verts = [topology.vertices[vid] for vid in edge.vertex_ids]

    fragmented = 0
    for v in verts:
        free = _free_of(v, override)
        cap = v.capacity_vector()
        # "free space has fallen below this threshold, making it essentially
        # unusable" -- a vertex counts as fragmented if ANY of its resource
        # dimensions has dropped below threshold_frac of capacity (the
        # tightest dimension determines usability for a "typical" future VM).
        ratios = [f / c for f, c in zip(free, cap) if c > 0]
        if ratios and min(ratios) < threshold_frac:
            fragmented += 1

    frag_level = fragmented / len(verts) if verts else 0.0
    return level_lambda.get(edge.level, 1.0) * frag_level


# ---------------------------------------------------------------------------
# TAFI dispatcher + gradient (Delta TAFI) computation
# ---------------------------------------------------------------------------

@dataclass
class HTAFMConfig:
    variant: str = "cut"                                   # "cut" | "entropy" | "hier"
    pending_demands: Sequence[TopoDemand] = ()               # used by "cut"
    threshold_frac: float = DEFAULT_LEVEL_THRESHOLD_FRAC      # used by "hier"
    level_lambda: Optional[Dict[str, float]] = None            # used by "hier"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown H-TAFM variant {self.variant!r}, expected one of {VARIANTS}")


def _phi(topology: Topology, edge: HyperEdge, cfg: HTAFMConfig,
         override: Optional[Dict[str, Tuple[float, float, float]]] = None) -> float:
    if cfg.variant == "cut":
        return float(phi_cut(topology, edge, cfg.pending_demands, override))
    elif cfg.variant == "entropy":
        return phi_entropy(topology, edge, override)
    else:
        return phi_hier(topology, edge, cfg.threshold_frac, cfg.level_lambda, override)


def compute_tafi(topology: Topology, cfg: HTAFMConfig) -> float:
    """Full TAFI(H, Pi) = Sum_{e in E} w(e) * phi(e, Pi). O(|E| * avg_edge_size)
    -- use for validation/tests, not the hot scheduling path (see delta_tafi)."""
    return sum(edge.weight * _phi(topology, edge, cfg) for edge in topology.edges.values())


def delta_tafi(topology: Topology, vertex_id: str, demand: TopoDemand, cfg: HTAFMConfig) -> float:
    """The incremental Delta-TAFI(H, Pi, v, p) from Sec. 4, computed by only
    touching the hyperedges containing `vertex_id` (Sec. 5's stated
    optimization), and without mutating any state: the target vertex's
    hypothetical post-placement free vector is passed via `override` to
    the phi_* functions rather than actually placed."""
    vertex = topology.vertices[vertex_id]
    touched = topology.edges_touching(vertex_id)
    if not touched:
        return 0.0  # isolated vertex (no socket/server/rack peers) -- placing here can't fragment anything

    before = sum(e.weight * _phi(topology, e, cfg) for e in touched)

    free = vertex.free_vector()
    hyp_free = (free[0] - demand.milli_cpu, free[1] - demand.memory_mib, free[2] - demand.total_milli_gpu())
    override = {vertex_id: hyp_free}
    after = sum(e.weight * _phi(topology, e, cfg, override=override) for e in touched)

    return after - before


# ---------------------------------------------------------------------------
# The scheduler itself: Algorithm 5 from the methodology
# ---------------------------------------------------------------------------

@dataclass
class HTAFMResult:
    scheduled: List[str]
    unscheduled: List[str]
    placement: Dict[str, str]  # demand name -> vertex id


class HTAFMScheduler:
    """Wraps a Topology and schedules a stream of TopoDemands one at a time,
    each time picking the feasible vertex that minimizes Delta-TAFI
    (Algorithm 5, Sec. 5). If no vertex fits, the methodology says "activate
    a new physical server" -- since this is an offline capacity-constrained
    simulation over a fixed, already-loaded cluster (consistent with how
    the rest of k8s_sim and the original Go simulator work -- neither adds
    hardware mid-run), that case is instead reported as unscheduled, the
    same convention k8s_sim.cluster.Cluster uses."""

    def __init__(self, topology: Topology, cfg: HTAFMConfig):
        self.topology = topology
        self.cfg = cfg

    def schedule_one(self, demand: TopoDemand) -> Optional[str]:
        candidates = [v for v in self.topology.vertices.values() if v.fits(demand)]
        if not candidates:
            return None
        best_id, best_delta = None, None
        for v in candidates:
            d = delta_tafi(self.topology, v.id, demand, self.cfg)
            if best_delta is None or d < best_delta:
                best_id, best_delta = v.id, d
        self.topology.vertices[best_id].place(demand)
        return best_id

    def schedule(self, demands: Sequence[TopoDemand]) -> HTAFMResult:
        result = HTAFMResult(scheduled=[], unscheduled=[], placement={})
        for d in demands:
            name = d.name or f"demand-{len(result.scheduled) + len(result.unscheduled)}"
            vid = self.schedule_one(d)
            if vid is None:
                result.unscheduled.append(name)
            else:
                result.scheduled.append(name)
                result.placement[name] = vid
        return result

    def unallocated_gpus(self) -> float:
        """Section 7's primary metric: idle GPU capacity (in whole-GPU
        units) that remains after scheduling."""
        return sum(sum(v.milli_gpu_left_list) for v in self.topology.vertices.values()) / 1000.0
