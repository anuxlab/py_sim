"""
Loaders for k8s_sim's native, time-agnostic CSV format — the same
``pods.csv`` / ``nodes.csv`` shape that ``gputrace export --format k8s_sim``
produces (see ``gputrace_bridge.py`` for the time-aware loader that reads
the same files but preserves ``submit_time``/``duration`` for the
event-driven runtime instead of discarding them).

nodes.csv columns: node_id, milli_cpu_capacity, gpu_count, milli_gpu_capacity, gpu_type
pods.csv columns:  pod_id, milli_cpu, milli_gpu, gpu_number, gpu_type[, user][, submit_time][, duration]
(submit_time/duration are ignored here if present — this loader is for
static scheduling only.)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .resource import NodeResource, PodResource


def load_nodes_csv(path: str | Path) -> Dict[str, NodeResource]:
    nodes: Dict[str, NodeResource] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            nodes[row["node_id"]] = NodeResource(
                node_id=row["node_id"],
                milli_cpu_capacity=int(float(row["milli_cpu_capacity"])),
                gpu_count=int(float(row.get("gpu_count", 0) or 0)),
                milli_gpu_capacity=int(float(row.get("milli_gpu_capacity", 0) or 0)),
                gpu_type=row.get("gpu_type", "") or "",
            )
    return nodes


def load_pods_csv(path: str | Path) -> List[PodResource]:
    pods: List[PodResource] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            pods.append(
                PodResource(
                    pod_id=row["pod_id"],
                    milli_cpu=int(float(row["milli_cpu"])),
                    milli_gpu=int(float(row.get("milli_gpu", 0) or 0)),
                    gpu_number=int(float(row.get("gpu_number", 0) or 0)),
                    gpu_type=row.get("gpu_type", "") or "",
                    user=row.get("user", "") or "",
                )
            )
    return pods


def reset_cluster(nodes: Dict[str, NodeResource]) -> Dict[str, NodeResource]:
    """Rebuild a fresh, fully-empty copy of ``nodes`` with identical
    capacities — use this between runs that should start from the same
    cluster state (e.g. comparing policies head-to-head)."""
    return {
        node_id: NodeResource(
            node_id=n.node_id,
            milli_cpu_capacity=n.milli_cpu_capacity,
            gpu_count=n.gpu_count,
            milli_gpu_capacity=n.milli_gpu_capacity,
            gpu_type=n.gpu_type,
        )
        for node_id, n in nodes.items()
    }
