"""
Port of pkg/utils/frag.go — the fragmentation model behind the FGD
(Fragmentation Gradient Descent) policy from the USENIX ATC'23 paper
"Beware of Fragmentation".

For a node and a hypothetical ("typical") pod spec, the node is classified
into one of 7 buckets depending on whether the node currently has enough
free CPU / GPU-memory to satisfy that pod:

    Q1_LACK_BOTH   - not enough CPU AND not enough GPU
    Q2_LACK_GPU    - enough CPU, not enough GPU
    Q3_SATISFIED   - enough CPU AND enough GPU (good fit)
    Q4_LACK_CPU    - enough GPU, not enough CPU
    XL_SATISFIED   - pod wants no GPU, node has enough CPU
    XR_LACK_CPU    - pod wants no GPU, node lacks CPU
    NO_ACCESS      - GPU type mismatch

All buckets except Q3_SATISFIED are considered "fragmented" idle GPU memory:
capacity that is sitting idle but can't actually be used by a representative
pod shape. FGD scores a placement by how much it reduces this fragmented
amount, summed across a distribution of "typical" pod shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .resource import NodeResource, PodResource

Q1_LACK_BOTH = "q1_lack_both"
Q2_LACK_GPU = "q2_lack_gpu"
Q3_SATISFIED = "q3_satisfied"
Q4_LACK_CPU = "q4_lack_cpu"
XL_SATISFIED = "xl_satisfied"
XR_LACK_CPU = "xr_lack_cpu"
NO_ACCESS = "no_access"

FRAG_TYPES = [Q1_LACK_BOTH, Q2_LACK_GPU, Q3_SATISFIED, Q4_LACK_CPU,
              XL_SATISFIED, XR_LACK_CPU, NO_ACCESS]


@dataclass
class TargetPod:
    """A representative pod shape and how frequently it occurs in the
    workload (0.0-1.0). Mirrors simontype.TargetPod."""

    pod: PodResource
    percentage: float


def get_gpu_frag_milli(node: NodeResource, pod: PodResource) -> int:
    """Sum of milli-gpu on GPUs that are individually too small for `pod`."""
    return sum(left for left in node.milli_gpu_left_list if left < pod.milli_gpu)


def get_node_pod_frag(node: NodeResource, pod: PodResource) -> str:
    """Classify (node, pod) into one of the 7 fragmentation buckets.
    Mirrors utils.GetNodePodFrag."""
    if pod.milli_gpu == 0 and pod.gpu_number == 0:
        return XL_SATISFIED if node.milli_cpu_left >= pod.milli_cpu else XR_LACK_CPU

    if not node.is_accessible_to(pod):
        return NO_ACCESS

    if node.can_host_on_gpu_memory(pod):
        return Q3_SATISFIED if node.milli_cpu_left >= pod.milli_cpu else Q4_LACK_CPU
    else:
        return Q2_LACK_GPU if node.milli_cpu_left >= pod.milli_cpu else Q1_LACK_BOTH


def node_gpu_share_frag_amount(node: NodeResource, typical_pods: Sequence[TargetPod]) -> Dict[str, float]:
    """The expected *amount* (in milli-gpu) of fragmented capacity on this
    node, broken down by bucket, averaged over the typical-pod distribution.
    Mirrors utils.NodeGpuShareFragAmount.

    Special case for Q3 (satisfied): only the portion of idle GPU-memory that
    sits on devices too small to fit the pod is "wasted"; the rest is
    genuinely usable, so it's still counted as Q3 (not fragmented).
    """
    amount = {t: 0.0 for t in FRAG_TYPES}
    gpu_milli_left_total = node.total_milli_gpu_left()

    for tp in typical_pods:
        freq = tp.percentage
        if not (0.0 <= freq <= 1.0):
            continue
        frag_type = get_node_pod_frag(node, tp.pod)
        if frag_type == Q3_SATISFIED:
            gpu_frag_milli = get_gpu_frag_milli(node, tp.pod)
            amount[Q2_LACK_GPU] += freq * gpu_frag_milli
            amount[Q3_SATISFIED] += freq * (gpu_milli_left_total - gpu_frag_milli)
        else:
            amount[frag_type] += freq * gpu_milli_left_total
    return amount


def frag_amount_sum_except_q3(amount: Dict[str, float]) -> float:
    return sum(v for k, v in amount.items() if k != Q3_SATISFIED)


def node_gpu_share_frag_amount_score(node: NodeResource, typical_pods: Sequence[TargetPod]) -> float:
    """The scalar fragmentation score for a node: total fragmented milli-gpu,
    summed over all buckets except Q3 (satisfied/usable). Lower is better.
    Mirrors utils.NodeGpuShareFragAmountScore."""
    return frag_amount_sum_except_q3(node_gpu_share_frag_amount(node, typical_pods))


def cluster_frag_ratio(nodes: Sequence[NodeResource], typical_pods: Sequence[TargetPod]) -> float:
    """Fraction of all idle GPU-memory across the cluster that is fragmented.
    Mirrors plugin.PreFilterFragGpuRatio. Handy top-line metric for comparing
    policies."""
    total_frag = 0.0
    total_idle = 0.0
    for node in nodes:
        amount = node_gpu_share_frag_amount(node, typical_pods)
        total_frag += frag_amount_sum_except_q3(amount)
        total_idle += sum(amount.values())
    if total_idle == 0:
        return 0.0
    return total_frag / total_idle


def build_typical_pods_uniform(pod_shapes: Sequence[PodResource]) -> List[TargetPod]:
    """Convenience helper: build a TargetPod distribution that weights each
    given pod shape equally. For frequency-weighted construction from a real
    workload trace, replicate GetTypicalPods from frag.go instead."""
    if not pod_shapes:
        return []
    w = 1.0 / len(pod_shapes)
    return [TargetPod(pod=p, percentage=w) for p in pod_shapes]
