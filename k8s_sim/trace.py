"""
Loader for the real production traces shipped in data/csv/, taken verbatim
from hkust-adsl/kubernetes-scheduler-simulator's data/ directory (1523 nodes,
8152 pods from a real heterogeneous GPU cluster; see data/ORIGINAL_DATA_README.md).

This replaces the YAML-generation step the original does via
data/pod_csv_to_yaml.py + prepare_input.sh — we read the CSVs directly into
our own NodeResource / PodResource objects, no intermediate k8s YAML needed.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .resource import NodeResource, PodResource, MILLI
from .fragmentation import TargetPod

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csv")

DEFAULT_POD_TRACE = "openb_pod_list_default.csv"
DEFAULT_NODE_TRACE = "openb_node_list_all_node.csv"
GPU_NODE_TRACE = "openb_node_list_gpu_node.csv"


@dataclass
class TracePod:
    """A pod from the trace, resource request plus its lifecycle timestamps
    (seconds), as-is from the CSV. `pod` is ready to feed into Cluster.schedule_pod."""

    pod: PodResource
    qos: str
    pod_phase: str
    creation_time: int
    deletion_time: int
    scheduled_time: int


def _clean_gpu_spec(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.lower() in ("", "nan", "none"):
        return ""
    return raw


def load_nodes_csv(path: Optional[str] = None) -> List[NodeResource]:
    """Read an openb_node_list_*.csv file into NodeResource objects.
    Columns: sn, cpu_milli, memory_mib, gpu, model"""
    path = path or os.path.join(DATA_DIR, DEFAULT_NODE_TRACE)
    nodes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cpu_milli = int(row["cpu_milli"])
            gpu_num = int(row["gpu"])
            gpu_type = _clean_gpu_spec(row.get("model", ""))
            nodes.append(NodeResource(
                name=row["sn"],
                milli_cpu_left=cpu_milli,
                milli_cpu_capacity=cpu_milli,
                milli_gpu_left_list=[MILLI] * gpu_num,
                gpu_type=gpu_type,
            ))
    return nodes


def load_pods_csv(path: Optional[str] = None, limit: Optional[int] = None,
                   sample: bool = False, seed: int = 0) -> List[TracePod]:
    """Read an openb_pod_list_*.csv file into TracePod objects.
    Columns: name, cpu_milli, memory_mib, num_gpu, gpu_milli, gpu_spec, qos,
             pod_phase, creation_time, deletion_time, scheduled_time

    By default (sample=False) rows are returned in file order (creation
    order in the original trace) and `limit` takes a *prefix*. Note that in
    several of the shipped traces (e.g. openb_pod_list_gpushare100.csv),
    GPU-requesting pods are concentrated later in the file, so a small
    chronological prefix can be entirely CPU-only pods. Pass sample=True to
    instead draw a uniform random subset of `limit` rows across the whole
    file — usually what you want for a quick, representative demo run.
    """
    path = path or os.path.join(DATA_DIR, DEFAULT_POD_TRACE)
    rows = []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

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
        pod = PodResource(
            milli_cpu=int(row["cpu_milli"]),
            milli_gpu=gpu_milli if num_gpu > 0 else 0,
            gpu_number=num_gpu,
            gpu_type=_clean_gpu_spec(row.get("gpu_spec", "")),
            name=row["name"],
        )
        pods.append(TracePod(
            pod=pod,
            qos=row.get("qos", ""),
            pod_phase=row.get("pod_phase", ""),
            creation_time=int(row["creation_time"]) if row.get("creation_time", "") not in ("", None) else -1,
            deletion_time=int(row["deletion_time"]) if row.get("deletion_time", "") not in ("", None) else -1,
            scheduled_time=int(row["scheduled_time"]) if row.get("scheduled_time", "") not in ("", None) else -1,
        ))
    return pods


def list_available_traces() -> List[str]:
    """All openb_pod_list_*.csv variants shipped in data/csv (different
    workload mixes: CPU-heavy, GPU-share-heavy, multi-GPU-heavy, GPU-type-
    constrained, etc. — see data/ORIGINAL_DATA_README.md)."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if f.startswith("openb_pod_list_"))


def build_typical_pods(trace_pods: Sequence[TracePod],
                        popularity_threshold: float = 60.0,
                        involve_cpu_pods: bool = True) -> List[TargetPod]:
    """Port of utils.GetTypicalPods: derive a frequency-weighted distribution
    of representative pod shapes from a real trace, keeping only the most
    common shapes that together cover `popularity_threshold` percent of pods
    (weighted count), then renormalizing their percentages to sum to 1.0.

    This is what FGD (and fragmentation_ratio / cluster.schedule_pods'
    typical_pods argument) should be fed when working off real trace data,
    instead of the hand-specified shapes in demo.py.
    """
    counts: dict = {}
    total = 0.0
    for tp in trace_pods:
        p = tp.pod
        if not involve_cpu_pods and p.gpu_number == 0:
            continue
        key = (p.milli_cpu, p.milli_gpu, p.gpu_number, p.gpu_type)
        counts[key] = counts.get(key, 0.0) + 1.0
        total += 1.0

    if total == 0:
        return []

    # sort shapes by decreasing frequency
    shapes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    expected = popularity_threshold * total / 100.0
    cum = 0.0
    kept = []
    for key, cnt in shapes:
        if cum >= expected:
            break
        kept.append((key, cnt))
        cum += cnt

    result = []
    for key, cnt in kept:
        cpu, gpu_milli, gpu_num, gpu_type = key
        result.append(TargetPod(
            pod=PodResource(milli_cpu=cpu, milli_gpu=gpu_milli, gpu_number=gpu_num, gpu_type=gpu_type),
            percentage=cnt / cum if cum else 0.0,
        ))
    return result
