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
