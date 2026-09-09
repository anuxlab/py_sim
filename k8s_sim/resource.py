"""
Core resource model.

``PodResource`` is a scheduling request; ``NodeResource`` is a capacity
pool that can accept/release pods. Everything in this package (policies,
cluster, fragmentation scoring, the event-driven runtime) is built on top
of these two types and nothing else — including gputrace-derived
workloads, once ``gputrace_bridge.py`` has converted them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PodResource:
    """A scheduling request.

    milli_cpu / milli_gpu are in milli-units (1000 = one whole CPU core or
    one whole GPU device), following Kubernetes' own resource-quantity
    convention — this is what lets ``gpu_number=1, milli_gpu=250`` express
    "a quarter of one GPU device" (MPS/MIG/time-slicing-style sharing)
    without a separate fractional-GPU type.
    """

    pod_id: str
    milli_cpu: int
    milli_gpu: int = 0
    gpu_number: int = 0
    gpu_type: str = ""
    user: str = ""

    def fits_in(self, node: "NodeResource") -> bool:
        if self.milli_cpu > node.remaining_milli_cpu:
            return False
        if self.gpu_number > node.remaining_gpu_count:
            return False
        if self.milli_gpu > node.remaining_milli_gpu:
            return False
        if self.gpu_type and node.gpu_type and self.gpu_type != node.gpu_type:
            return False
        return True


@dataclass
class NodeResource:
    """A cluster node's capacity pool. GPU capacity is tracked at two
    granularities simultaneously: whole-device count (``gpu_count`` /
    ``remaining_gpu_count``, for pods that need N whole devices) and
    milli-units of the *currently partially-used* device
    (``remaining_milli_gpu``, for fractional/shared requests) — see
    ``add()`` for how the two interact.
    """

    node_id: str
    milli_cpu_capacity: int
    gpu_count: int = 0
    gpu_type: str = ""
    milli_gpu_capacity: int = field(default=0)  # defaults to gpu_count*1000 if unset

    def __post_init__(self):
        if self.milli_gpu_capacity == 0 and self.gpu_count:
            self.milli_gpu_capacity = self.gpu_count * 1000
        self.remaining_milli_cpu = self.milli_cpu_capacity
        self.remaining_gpu_count = self.gpu_count
        self.remaining_milli_gpu = self.milli_gpu_capacity
        self.allocated: Dict[str, PodResource] = {}

    def add(self, pod: PodResource) -> bool:
        """Attempt to place ``pod`` on this node. Returns True and mutates
        remaining capacity on success; returns False (no mutation) if it
        doesn't fit."""
        if not pod.fits_in(self):
            return False
        self.remaining_milli_cpu -= pod.milli_cpu
        self.remaining_milli_gpu -= pod.milli_gpu
        self.remaining_gpu_count -= pod.gpu_number
        self.allocated[pod.pod_id] = pod
        return True

    def remove(self, pod_id: str) -> Optional[PodResource]:
        """Release a previously-placed pod's resources back to the node.
        Returns the released PodResource, or None if it wasn't here."""
        pod = self.allocated.pop(pod_id, None)
        if pod is None:
            return None
        self.remaining_milli_cpu += pod.milli_cpu
        self.remaining_milli_gpu += pod.milli_gpu
        self.remaining_gpu_count += pod.gpu_number
        return pod

    def cpu_utilization(self) -> float:
        if self.milli_cpu_capacity == 0:
            return 0.0
        return 1 - (self.remaining_milli_cpu / self.milli_cpu_capacity)

    def gpu_utilization(self) -> float:
        if self.milli_gpu_capacity == 0:
            return 0.0
        return 1 - (self.remaining_milli_gpu / self.milli_gpu_capacity)
