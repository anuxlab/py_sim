"""
Fragmentation scoring, in the spirit of Alibaba's Fragmentation Gradient
Descent (FGD) approach: rather than scoring a node's leftover capacity in
the abstract, score it by how many *representative pod shapes actually
seen in this workload* could still be placed in what's left. Resources
left over that can't fit ANY typical shape are fragmented in the sense
that actually matters — they can't be used by the workload you have,
regardless of how large the raw number looks.

``build_typical_pods`` derives that representative shape set directly from
a pod list (real, loaded, or gputrace-generated) rather than a fixed
hardcoded set of sizes, so fragmentation scoring is always relative to the
workload actually being scheduled.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .resource import NodeResource, PodResource


def build_typical_pods(pods: List[PodResource], n_quantiles: int = 5) -> List[PodResource]:
    """Derive a small representative set of pod shapes from a pod list, by
    binning GPU-requesting pods' (milli_cpu, milli_gpu, gpu_number) into
    quantiles. Falls back to a couple of fixed CPU-only shapes if the
    workload has no GPU pods at all.
    """
    gpu_pods = [p for p in pods if p.gpu_number > 0 or p.milli_gpu > 0]
    if not gpu_pods:
        cpu_vals = sorted(p.milli_cpu for p in pods) or [1000]
        qs = np.quantile(cpu_vals, np.linspace(0.2, 0.8, n_quantiles))
        return [PodResource(pod_id=f"typical_{i}", milli_cpu=int(q)) for i, q in enumerate(qs)]

    cpu = np.array([p.milli_cpu for p in gpu_pods])
    gpu_milli = np.array([p.milli_gpu if p.milli_gpu else p.gpu_number * 1000 for p in gpu_pods])
    gpu_n = np.array([max(p.gpu_number, 1) for p in gpu_pods])

    qs = np.linspace(0.1, 0.9, n_quantiles)
    typical = []
    for i, q in enumerate(qs):
        typical.append(
            PodResource(
                pod_id=f"typical_{i}",
                milli_cpu=int(np.quantile(cpu, q)),
                milli_gpu=int(np.quantile(gpu_milli, q)),
                gpu_number=int(round(np.quantile(gpu_n, q))),
            )
        )
    return typical


def build_typical_pods_weighted(pods: List[PodResource], n_shapes: int = 5):
    """
    Like build_typical_pods, but fixes two specific, checkable weaknesses in
    that function that w_fgd (see policies.py) exploits:

    1. UNWEIGHTED SHAPES: build_typical_pods's 5 shapes are all weighted
       equally when scored (unfit_fraction is a plain mean). A shape near
       the 90th percentile represents far less of the real workload than
       one near the median, but FGD's gradient treats a placement that
       blocks the rare shape identically to one that blocks the common
       one. Here each shape gets weight = the fraction of the *actual pod
       population* closest to it (a proper stratified-sampling weight, not
       a fixed 1/n).

    2. PER-DIMENSION QUANTILES ("chimera" shapes): build_typical_pods takes
       the q-th quantile of milli_cpu, milli_gpu, and gpu_number
       *independently*. If CPU and GPU demand aren't perfectly correlated
       (they aren't, in every gputrace scenario we've measured), the
       resulting "typical pod" at quantile q is a coordinatewise chimera
       that may not resemble any pod actually in the workload. Here, each
       shape is the *medoid* (an actual observed pod, real joint
       cpu+gpu+gpu_number combination) of a demand-magnitude stratum, so
       every typical shape is guaranteed to be something real.

    Returns (shapes, weights) -- weights sum to 1.0, same length as shapes.
    Falls back to build_typical_pods()'s CPU-only path (uniform weights)
    when there are no GPU pods, for the same reason that function does.
    """
    gpu_pods = [p for p in pods if p.gpu_number > 0 or p.milli_gpu > 0]
    if not gpu_pods:
        shapes = build_typical_pods(pods, n_quantiles=n_shapes)
        return shapes, [1.0 / len(shapes)] * len(shapes) if shapes else []

    # composite demand magnitude per pod, normalized so CPU and GPU
    # contribute comparably regardless of their raw unit scales
    cpu = np.array([p.milli_cpu for p in gpu_pods], dtype=float)
    gpu_milli = np.array([p.milli_gpu if p.milli_gpu else p.gpu_number * 1000 for p in gpu_pods], dtype=float)
    cpu_norm = cpu / (cpu.max() or 1.0)
    gpu_norm = gpu_milli / (gpu_milli.max() or 1.0)
    magnitude = cpu_norm + gpu_norm

    order = np.argsort(magnitude)
    strata = np.array_split(order, n_shapes)  # n_shapes equal-count magnitude strata

    shapes, weights = [], []
    for i, stratum_idx in enumerate(strata):
        if len(stratum_idx) == 0:
            continue
        stratum_mag = magnitude[stratum_idx]
        medoid_local = stratum_idx[np.argmin(np.abs(stratum_mag - np.median(stratum_mag)))]
        medoid_pod = gpu_pods[medoid_local]
        shapes.append(PodResource(
            pod_id=f"typical_w{i}", milli_cpu=medoid_pod.milli_cpu,
            milli_gpu=medoid_pod.milli_gpu, gpu_number=medoid_pod.gpu_number,
            gpu_type=medoid_pod.gpu_type,
        ))
        weights.append(len(stratum_idx) / len(gpu_pods))
    return shapes, weights


def unfit_fraction_weighted(node: NodeResource, typical_pods: List[PodResource],
                             weights: List[float]) -> float:
    """Weighted analogue of unfit_fraction: sum of weights of typical_pods
    that don't fit, rather than a plain unweighted mean. Weights should sum
    to ~1.0 (as build_typical_pods_weighted's do), so this stays comparable
    in range ([0, 1]) to the unweighted score."""
    if not typical_pods:
        return 0.0
    return sum(w for p, w in zip(typical_pods, weights) if not p.fits_in(node))



def unfit_fraction(node: NodeResource, typical_pods: List[PodResource]) -> float:
    """Fraction of ``typical_pods`` that could NOT be placed in ``node``'s
    *current remaining* capacity. 0.0 = no fragmentation (everything
    representative still fits); 1.0 = fully fragmented (nothing
    representative fits, even though raw numbers may be nonzero)."""
    if not typical_pods:
        return 0.0
    cannot_fit = sum(1 for p in typical_pods if not p.fits_in(node))
    return cannot_fit / len(typical_pods)


def cluster_fragmentation_score(nodes: List[NodeResource], typical_pods: List[PodResource]) -> float:
    """Mean unfit-fraction across all nodes with nonzero remaining GPU
    capacity (fully-empty and fully-packed nodes both contribute — empty
    nodes at 0.0, full nodes near 1.0 automatically)."""
    if not nodes:
        return 0.0
    scores = [unfit_fraction(n, typical_pods) for n in nodes]
    return float(np.mean(scores))
