"""
Port of pkg/type/resource.go

Resource units follow the original project's convention:
  - CPU is tracked in "milli-cpu" (1000m == 1 core), same as Kubernetes.
  - GPU is tracked per-device in "milli-gpu" (1000m == 1 whole GPU), which is
    how the original models GPU-sharing / fractional-GPU requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

MILLI = 1000  # 1 whole GPU == 1000 "milli-gpu", matches gpushareutils.MILLI

# Used to normalize Best-Fit / Dot-Product scores, matches gpushareutils consts.
MAX_SPEC_CPU = 128_000  # milli-cpu
MAX_SPEC_GPU = 8_000    # milli-gpu (8 whole GPUs)

MAX_NODE_SCORE = 100  # framework.MaxNodeScore
MIN_NODE_SCORE = 0    # framework.MinNodeScore


@dataclass(frozen=True)
class PodResource:
    """A pod's resource request. Mirrors simontype.PodResource."""

    milli_cpu: int
    milli_gpu: int = 0      # per-GPU milli-gpu request (0-1000)
    gpu_number: int = 0     # number of GPUs requested
    gpu_type: str = ""      # "" means "no GPU type constraint"
    name: str = ""          # convenience, not in the original struct

    def total_milli_gpu(self) -> int:
        return self.milli_gpu * self.gpu_number

    def is_gpu_share(self) -> bool:
        """True if this is a fractional-GPU ("GPU-share") pod."""
        return self.gpu_number == 1 and self.milli_gpu < MILLI

    def to_resource_vec(self) -> List[float]:
        """[milli_cpu, total_milli_gpu] - used by Best-Fit / Dot-Product."""
        return [float(self.milli_cpu), float(self.total_milli_gpu())]

    def repr(self) -> str:
        return (f"<CPU: {self.milli_cpu/1000:6.2f}, GPU: {self.gpu_number} x "
                f"{{{self.milli_gpu:4d}}}m ({self.gpu_type})>")


@dataclass
class NodeResource:
    """A node's current free capacity. Mirrors simontype.NodeResource."""

    name: str
    milli_cpu_left: int
    milli_cpu_capacity: int
    milli_gpu_left_list: List[int] = field(default_factory=list)  # per-GPU device
    gpu_type: str = ""
    gpu_affinity: dict = field(default_factory=dict)  # affinity_key -> count of pods

    @property
    def gpu_number(self) -> int:
        return len(self.milli_gpu_left_list)

    def total_milli_gpu_left(self) -> int:
        return sum(self.milli_gpu_left_list)

    def fully_free_gpu_num(self) -> int:
        return sum(1 for g in self.milli_gpu_left_list if g == MILLI)

    def sorted_gpu_left_index_list(self, ascending: bool = True) -> List[int]:
        idx = list(range(len(self.milli_gpu_left_list)))
        idx.sort(key=lambda i: self.milli_gpu_left_list[i], reverse=not ascending)
        return idx

    def to_resource_vec(self) -> List[float]:
        """[milli_cpu_left, total_milli_gpu_left]."""
        return [float(self.milli_cpu_left), float(self.total_milli_gpu_left())]

    def copy(self) -> "NodeResource":
        return NodeResource(
            name=self.name,
            milli_cpu_left=self.milli_cpu_left,
            milli_cpu_capacity=self.milli_cpu_capacity,
            milli_gpu_left_list=list(self.milli_gpu_left_list),
            gpu_type=self.gpu_type,
            gpu_affinity=dict(self.gpu_affinity),
        )

    def is_accessible_to(self, pod: PodResource) -> bool:
        """Mirrors utils.IsNodeAccessibleToPodByType (simplified: exact/blank match)."""
        if not pod.gpu_type:
            return True
        if not self.gpu_type:
            return False  # pod wants a GPU type, node is CPU-only
        pod_types = pod.gpu_type.split("|")
        return self.gpu_type in pod_types

    def can_host_on_gpu_memory(self, pod: PodResource) -> bool:
        """Mirrors utils.CanNodeHostPodOnGpuMemory: can enough distinct GPUs
        each supply pod.milli_gpu to satisfy pod.gpu_number?"""
        if pod.gpu_number == 0:
            return True
        need = pod.gpu_number
        for left in self.milli_gpu_left_list:
            if left >= pod.milli_gpu:
                need -= 1
                if need <= 0:
                    return True
        return False

    def fits(self, pod: PodResource) -> bool:
        """Filter step: cpu capacity + gpu type + gpu capacity."""
        if self.milli_cpu_left < pod.milli_cpu:
            return False
        if not self.is_accessible_to(pod):
            return False
        if pod.gpu_number > 0 and not self.can_host_on_gpu_memory(pod):
            return False
        return True

    def sub(self, pod: PodResource, gpu_ids: Optional[List[int]] = None) -> "NodeResource":
        """Allocate `pod` onto this node, returning a NEW NodeResource.
        Mirrors simontype.NodeResource.Sub: without explicit gpu_ids, GPUs are
        packed onto the *least sufficient* device first (ascending sort),
        i.e. best-fit-style bin packing at the GPU-device level."""
        out, _ = self.sub_with_gpu_ids(pod, gpu_ids)
        return out

    def sub_with_gpu_ids(self, pod: PodResource,
                          gpu_ids: Optional[List[int]] = None) -> "tuple[NodeResource, List[int]]":
        """Same as sub(), but also returns exactly which GPU device indices
        were used -- needed by anything that must later release the SAME
        devices (e.g. k8s_sim.simulation's departure handling), since without
        recording this, a later add() with no gpu_ids would re-derive
        "least sufficient first" against whatever the node's state happens
        to be *then*, which may no longer match what was actually freed."""
        out = self.copy()
        if out.milli_cpu_left < pod.milli_cpu:
            raise ValueError(f"node {self.name} lacks CPU for pod {pod.repr()}")
        out.milli_cpu_left -= pod.milli_cpu

        need = pod.gpu_number
        if need == 0:
            return out, []

        order = gpu_ids if gpu_ids is not None else out.sorted_gpu_left_index_list(ascending=True)
        used: List[int] = []
        for i in order:
            if need <= 0:
                break
            if pod.milli_gpu <= out.milli_gpu_left_list[i]:
                out.milli_gpu_left_list[i] -= pod.milli_gpu
                used.append(i)
                need -= 1
        if need > 0:
            raise ValueError(f"node {self.name} failed to accommodate pod {pod.repr()} "
                              f"({need} GPU requests left)")
        return out, used

    def add(self, pod: PodResource, gpu_ids: Optional[List[int]] = None) -> "NodeResource":
        """Release `pod`'s resources back onto the node (e.g. on completion/eviction)."""
        out = self.copy()
        out.milli_cpu_left += pod.milli_cpu
        if out.milli_cpu_left > out.milli_cpu_capacity:
            out.milli_cpu_left -= pod.milli_cpu
            raise ValueError(f"releasing pod {pod.repr()} would exceed node {self.name} cpu capacity")

        need = pod.gpu_number
        if need == 0:
            return out
        order = gpu_ids if gpu_ids is not None else out.sorted_gpu_left_index_list(ascending=True)
        for i in order:
            if need <= 0:
                break
            if out.milli_gpu_left_list[i] + pod.milli_gpu <= MILLI:
                out.milli_gpu_left_list[i] += pod.milli_gpu
                need -= 1
        if need > 0:
            raise ValueError(f"node {self.name} failed to release pod {pod.repr()}")
        return out

    def repr(self) -> str:
        s = f"{self.name}<CPU: {self.milli_cpu_left/1000:6.2f}/{self.milli_cpu_capacity/1000:6.2f}"
        s += f", GPU ({self.gpu_type}): {self.gpu_number}"
        if self.gpu_number > 0:
            s += " x %dm, Left:" % MILLI
            s += " " + " ".join(f"{g}m" for g in self.milli_gpu_left_list)
        s += ">"
        return s
