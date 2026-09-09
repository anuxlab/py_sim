"""
``Cluster``: a named collection of nodes plus static (snapshot)
scheduling. This is the original mode this package was built around —
schedule a bag of pods against a fixed node set, in submission order, with
no notion of simulated time. It's the right tool for questions purely
about packing quality ("does this policy fragment less than that one on
this exact pod mix").

It is NOT the right tool for questions about arrival dynamics (bursty vs.
smooth arrivals, queueing delay under load, preemption/eviction over time)
— for those, use ``event_runtime.EventDrivenRunner`` instead, which wraps
this same ``Cluster``/``NodeResource`` machinery in a discrete-event time
loop. See that module's docstring for why the distinction matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .fragmentation import cluster_fragmentation_score
from .policies import get_policy
from .resource import NodeResource, PodResource


@dataclass
class ScheduleResult:
    pod_id: str
    node_id: Optional[str]  # None if rejected (no feasible node)


class Cluster:
    def __init__(self, nodes: Dict[str, NodeResource]):
        self.nodes = nodes
        self._round_robin_state = [0]

    @property
    def node_list(self) -> List[NodeResource]:
        return list(self.nodes.values())

    def schedule_pod(self, pod: PodResource, policy: str = "first_fit",
                      typical_pods: Optional[List[PodResource]] = None,
                      rng: Optional[np.random.Generator] = None) -> Optional[str]:
        """Attempt to place a single pod using the named policy. Returns
        the chosen node_id, or None if no node had room."""
        fn = get_policy(policy)
        chosen = fn(
            pod, self.node_list,
            rng=rng, typical_pods=typical_pods,
            round_robin_state=self._round_robin_state,
        )
        if chosen is None:
            return None
        chosen.add(pod)
        return chosen.node_id

    def schedule_pods(self, pods: List[PodResource], policy: str = "first_fit",
                       typical_pods: Optional[List[PodResource]] = None,
                       seed: int = 0) -> List[ScheduleResult]:
        """Static batch scheduling: place every pod in ``pods``, in list
        order, against the cluster's current state. Does not model time —
        pods are never released. Use a fresh ``Cluster`` (or
        ``reset_cluster``, see trace.py) per run if you want to compare
        policies on identical starting conditions.
        """
        rng = np.random.default_rng(seed)
        results = []
        for pod in pods:
            node_id = self.schedule_pod(pod, policy=policy, typical_pods=typical_pods, rng=rng)
            results.append(ScheduleResult(pod_id=pod.pod_id, node_id=node_id))
        return results

    def fragmentation_score(self, typical_pods: List[PodResource]) -> float:
        return cluster_fragmentation_score(self.node_list, typical_pods)

    def total_gpu_utilization(self) -> float:
        total_cap = sum(n.milli_gpu_capacity for n in self.node_list)
        if total_cap == 0:
            return 0.0
        total_used = sum(n.milli_gpu_capacity - n.remaining_milli_gpu for n in self.node_list)
        return total_used / total_cap
