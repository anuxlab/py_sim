"""
k8s_sim - a standalone Python re-implementation of the core scheduling logic
from hkust-adsl/kubernetes-scheduler-simulator ("Simon").

This package ports the resource model, fragmentation math, and the six
scoring policies (Random, Best-Fit, Dot-Product/Tetris, GPU-Packing,
GPU-Clustering, FGD) out of the original Go + real-k8s-scheduler-framework
code into plain, dependency-free Python so they can be run and experimented
with outside of Kubernetes entirely. See README.md for details and for the
simplifications made relative to the original.
"""

from .resource import PodResource, NodeResource
from .fragmentation import TargetPod, node_gpu_share_frag_amount_score, get_node_pod_frag
from .cluster import Cluster
from . import policies
from . import trace
from . import experiment

__all__ = [
    "PodResource",
    "NodeResource",
    "TargetPod",
    "node_gpu_share_frag_amount_score",
    "get_node_pod_frag",
    "Cluster",
    "policies",
    "trace",
    "experiment",
]
