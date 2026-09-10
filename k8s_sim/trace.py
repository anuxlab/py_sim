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
import os
from pathlib import Path
from typing import Dict, List

from .resource import NodeResource, PodResource

# Convenience constants pointing at the real trace data shipped in data/csv/
# (1523 nodes / 8152 pods from a live heterogeneous GPU cluster -- see
# data/ORIGINAL_DATA_README.md). These are additive: load_nodes_csv() /
# load_pods_csv() below always take an explicit path and never depended on
# these, so restoring them doesn't change either function's behavior. They
# exist so callers (demo scripts, k8s_sim.topology.load_topology_from_csv)
# don't have to hardcode the data/ layout themselves.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csv")
DEFAULT_POD_TRACE = "openb_pod_list_default.csv"
DEFAULT_NODE_TRACE = "openb_node_list_all_node.csv"
GPU_NODE_TRACE = "openb_node_list_gpu_node.csv"


def list_available_traces() -> List[str]:
    """Every openb_pod_list_*.csv shipped under data/csv/, for scripts that
    want to let the user pick a trace interactively (see trace_demo.py)."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if f.startswith("openb_pod_list_"))


def _clean_gpu_spec(raw: str) -> str:
    raw = (raw or "").strip()
    return "" if raw.lower() in ("", "nan", "none") else raw


def load_real_trace_nodes(path: str | Path | None = None) -> Dict[str, NodeResource]:
    """Read an openb_node_list_*.csv (the RAW Alibaba trace format shipped in
    data/csv/ -- columns: sn, cpu_milli, memory_mib, gpu, model) into the
    current unified NodeResource dict shape, i.e. the same return shape as
    load_nodes_csv() below, just from the real trace's own column names
    instead of a gputrace-export nodes.csv. Restores the "run against our
    real bundled data" capability that load_nodes_csv()'s rewrite (for
    gputrace-export CSVs) otherwise no longer covers -- see
    data/ORIGINAL_DATA_README.md and docs/HTAFM.md for what this data is.
    """
    path = path or os.path.join(DATA_DIR, DEFAULT_NODE_TRACE)
    nodes: Dict[str, NodeResource] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            gpu_count = int(row["gpu"])
            nodes[row["sn"]] = NodeResource(
                node_id=row["sn"],
                milli_cpu_capacity=int(row["cpu_milli"]),
                gpu_count=gpu_count,
                milli_gpu_capacity=gpu_count * 1000,
                gpu_type=_clean_gpu_spec(row.get("model", "")),
            )
    return nodes


def load_real_trace_pods(path: str | Path | None = None, limit: int | None = None,
                          sample: bool = False, seed: int = 0) -> List[PodResource]:
    """Read an openb_pod_list_*.csv (RAW Alibaba trace format -- columns:
    name, cpu_milli, memory_mib, num_gpu, gpu_milli, gpu_spec, qos,
    pod_phase, creation_time, deletion_time, scheduled_time) into a flat
    PodResource list, the same return shape load_pods_csv() below returns
    for gputrace-export pods.csv files.

    By default (sample=False) rows come back in file order and `limit`
    takes a *prefix*. In several shipped traces (e.g.
    openb_pod_list_gpushare100.csv) GPU-requesting pods are concentrated
    later in the file, so a small chronological prefix can be entirely
    CPU-only. Pass sample=True for a uniform random subset instead --
    usually what you want for a quick, representative demo run.
    """
    path = path or os.path.join(DATA_DIR, DEFAULT_POD_TRACE)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if limit is not None and limit < len(rows):
        if sample:
            import random
            rows = random.Random(seed).sample(rows, limit)
        else:
            rows = rows[:limit]

    pods = []
    for row in rows:
        num_gpu = int(row["num_gpu"])
        gpu_milli = int(row["gpu_milli"]) if row.get("gpu_milli", "") not in ("", None) else 0
        pods.append(PodResource(
            pod_id=row["name"],
            milli_cpu=int(row["cpu_milli"]),
            milli_gpu=gpu_milli if num_gpu > 0 else 0,
            gpu_number=num_gpu,
            gpu_type=_clean_gpu_spec(row.get("gpu_spec", "")),
        ))
    return pods


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
