"""
A minimal filter -> score -> bind scheduling loop, standing in for the real
project's simulator.go (which drives the actual k8s scheduler framework
against a fake API server). This is the piece that's specific to running
inside Kubernetes in the original; here it's just plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .resource import NodeResource, PodResource
from .fragmentation import TargetPod, cluster_frag_ratio
from . import policies as policy_mod


@dataclass
class ScheduleResult:
    scheduled: List[str] = field(default_factory=list)     # pod names placed
    unscheduled: List[str] = field(default_factory=list)   # pod names that didn't fit anywhere
    placement: Dict[str, str] = field(default_factory=dict)  # pod name -> node name


class Cluster:
    def __init__(self, nodes: Sequence[NodeResource]):
        self.nodes: Dict[str, NodeResource] = {n.name: n.copy() for n in nodes}
        self._policy_state: Dict[str, dict] = {}  # per-policy persistent scratchpad (e.g. round-robin counter)

    def node_list(self) -> List[NodeResource]:
        return list(self.nodes.values())

    def feasible_nodes(self, pod: PodResource) -> List[NodeResource]:
        return [n for n in self.nodes.values() if n.fits(pod)]

    def schedule_pod(self, pod: PodResource, policy: str,
                      typical_pods: Optional[Sequence[TargetPod]] = None,
                      affinity_key: Optional[str] = None) -> Optional[str]:
        """Try to place a single pod. Returns the chosen node name, or None
        if no node is feasible."""
        score_fn = policy_mod.POLICIES[policy]
        candidates = self.feasible_nodes(pod)
        if not candidates:
            return None

        ctx: dict = {}
        if typical_pods is not None:
            ctx["typical_pods"] = typical_pods
        if affinity_key is not None:
            ctx["affinity_key"] = affinity_key
        ctx["_cluster_state"] = self._policy_state.setdefault(policy, {})

        prepare = policy_mod.PREPARE_HOOKS.get(policy)
        if prepare:
            prepare(candidates, pod, ctx)

        best_node, best_score = None, None
        for node in candidates:
            s = score_fn(node, pod, ctx)
            if best_score is None or s > best_score:
                best_node, best_score = node, s

        chosen = self.nodes[best_node.name]
        self.nodes[best_node.name] = chosen.sub(pod)

        # bookkeeping for gpu-clustering's affinity map
        if pod.gpu_number > 0:
            key = affinity_key or pod.gpu_type or "default"
            self.nodes[best_node.name].gpu_affinity[key] = \
                self.nodes[best_node.name].gpu_affinity.get(key, 0) + 1

        return best_node.name

    def schedule_pod_tracked(self, pod: PodResource, policy: str,
                              typical_pods: Optional[Sequence[TargetPod]] = None,
                              affinity_key: Optional[str] = None) -> Optional["tuple[str, List[int]]"]:
        """Same decision logic as schedule_pod, but also returns exactly
        which GPU device indices were used, so a caller (k8s_sim.simulation)
        can release precisely those devices later via release_pod() --
        needed once pods can depart independently of arrival order, which
        schedule_pod's plain node.sub() doesn't support (see
        NodeResource.sub_with_gpu_ids's docstring)."""
        score_fn = policy_mod.POLICIES[policy]
        candidates = self.feasible_nodes(pod)
        if not candidates:
            return None

        ctx: dict = {}
        if typical_pods is not None:
            ctx["typical_pods"] = typical_pods
        if affinity_key is not None:
            ctx["affinity_key"] = affinity_key
        ctx["_cluster_state"] = self._policy_state.setdefault(policy, {})

        prepare = policy_mod.PREPARE_HOOKS.get(policy)
        if prepare:
            prepare(candidates, pod, ctx)

        best_node, best_score = None, None
        for node in candidates:
            s = score_fn(node, pod, ctx)
            if best_score is None or s > best_score:
                best_node, best_score = node, s

        chosen = self.nodes[best_node.name]
        new_node, gpu_ids = chosen.sub_with_gpu_ids(pod)
        self.nodes[best_node.name] = new_node

        if pod.gpu_number > 0:
            key = affinity_key or pod.gpu_type or "default"
            self.nodes[best_node.name].gpu_affinity[key] = \
                self.nodes[best_node.name].gpu_affinity.get(key, 0) + 1

        return best_node.name, gpu_ids

    def release_pod(self, pod: PodResource, node_name: str, gpu_ids: Sequence[int]) -> None:
        """Release a previously-placed pod's resources back onto the named
        node, using the SAME gpu_ids schedule_pod_tracked recorded for it."""
        self.nodes[node_name] = self.nodes[node_name].add(pod, gpu_ids=list(gpu_ids))

    def schedule_pods(self, pods: Sequence[PodResource], policy: str,
                       typical_pods: Optional[Sequence[TargetPod]] = None) -> ScheduleResult:
        """Schedule a batch of pods, one at a time, in the given order
        (this is what SchedulePods does in simulator.go)."""
        result = ScheduleResult()
        for pod in pods:
            node_name = self.schedule_pod(pod, policy, typical_pods=typical_pods)
            pod_id = pod.name or pod.repr()
            if node_name is None:
                result.unscheduled.append(pod_id)
            else:
                result.scheduled.append(pod_id)
                result.placement[pod_id] = node_name
        return result

    def fragmentation_ratio(self, typical_pods: Sequence[TargetPod]) -> float:
        """Cluster-wide fraction of idle GPU memory that's fragmented (lower
        is better). Mirrors plugin.PreFilterFragGpuRatio."""
        return cluster_frag_ratio(self.node_list(), typical_pods)

    def utilization(self) -> Dict[str, float]:
        """Simple headline stats: fraction of CPU / GPU capacity in use."""
        cpu_cap = sum(n.milli_cpu_capacity for n in self.nodes.values())
        cpu_left = sum(n.milli_cpu_left for n in self.nodes.values())
        gpu_cap = sum(n.gpu_number for n in self.nodes.values()) * 1000
        gpu_left = sum(n.total_milli_gpu_left() for n in self.nodes.values())
        return {
            "cpu_utilization": 1 - cpu_left / cpu_cap if cpu_cap else 0.0,
            "gpu_utilization": 1 - gpu_left / gpu_cap if gpu_cap else 0.0,
        }
