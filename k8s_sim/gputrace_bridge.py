"""
Bridge between gputrace and k8s_sim.

This is the piece that didn't exist before: ``trace.py``'s loaders discard
``submit_time``/``duration`` because static scheduling has no notion of
time. But gputrace's whole value proposition — bursty vs. diurnal vs.
flash-crowd arrival *dynamics* — lives entirely in those two columns. A
scenario fed through the static loader is indistinguishable from any other
scenario with the same pod-size mix, regardless of how it was meant to
stress the scheduler. This module preserves the time dimension and hands
it to ``event_runtime.EventDrivenRunner``, which is what actually lets
scenario diversity show up in the results.

Two entry points:

    load_gputrace_export(pods_csv, nodes_csv)
        reads files written by `gputrace export --format k8s_sim`

    load_gputrace_dataframe(df, n_nodes=0, gpus_per_node=8)
        converts an in-memory gputrace unified-schema DataFrame directly,
        without a CSV round trip — the field mapping is duplicated here
        (rather than importing gputrace) so k8s_sim has no hard dependency
        on the gputrace package; the two implementations are intentionally
        kept in lockstep with gputrace/exporters/k8s_sim.py's
        _pods_frame/_synthesize_nodes and should be changed together.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .resource import NodeResource, PodResource


@dataclass
class TimedPodEvent:
    submit_time: float
    duration: float
    pod: PodResource


def load_gputrace_export(pods_csv: str | Path, nodes_csv: str | Path) -> Tuple[Dict[str, NodeResource], List[TimedPodEvent]]:
    """Read the pods.csv/nodes.csv pair written by
    ``gputrace export --format k8s_sim``, preserving submit_time/duration.
    Returns (nodes, events) with events sorted by submit_time — the input
    ``EventDrivenRunner.run()`` expects.
    """
    nodes: Dict[str, NodeResource] = {}
    with open(nodes_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            nodes[row["node_id"]] = NodeResource(
                node_id=row["node_id"],
                milli_cpu_capacity=int(float(row["milli_cpu_capacity"])),
                gpu_count=int(float(row.get("gpu_count", 0) or 0)),
                milli_gpu_capacity=int(float(row.get("milli_gpu_capacity", 0) or 0)),
                gpu_type=row.get("gpu_type", "") or "",
            )

    events: List[TimedPodEvent] = []
    with open(pods_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            pod = PodResource(
                pod_id=row["pod_id"],
                milli_cpu=int(float(row["milli_cpu"])),
                milli_gpu=int(float(row.get("milli_gpu", 0) or 0)),
                gpu_number=int(float(row.get("gpu_number", 0) or 0)),
                gpu_type=row.get("gpu_type", "") or "",
                user=row.get("user", "") or "",
            )
            events.append(
                TimedPodEvent(
                    submit_time=float(row["submit_time"]),
                    duration=float(row["duration"]),
                    pod=pod,
                )
            )
    events.sort(key=lambda e: e.submit_time)
    return nodes, events


def load_gputrace_dataframe(df, n_nodes: int = 0, gpus_per_node: int = 8,
                             target_utilization: float = 0.6) -> Tuple[Dict[str, NodeResource], List[TimedPodEvent]]:
    """Same conversion as ``load_gputrace_export``, but directly from an
    in-memory pandas DataFrame in gputrace's unified schema — skips the
    CSV round trip, useful when generating and simulating in the same
    Python process (e.g. inside ``experiment.py``'s sweep loop).
    """
    import numpy as np
    import pandas as pd

    if "gpu_milli" in df.columns and df["gpu_milli"].notna().any():
        gpu_milli_per_device = df["gpu_milli"].fillna(1000.0).to_numpy()
    else:
        gpu_milli_per_device = np.full(len(df), 1000.0)

    events: List[TimedPodEvent] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        num_gpu = max(row["num_gpu"], 0)
        milli_gpu = int(round(num_gpu * gpu_milli_per_device[pos]))
        pod = PodResource(
            pod_id=str(row["job_id"]),
            milli_cpu=int(round(row["num_cpu"] * 1000)),
            milli_gpu=milli_gpu,
            gpu_number=int(round(num_gpu)),
            gpu_type=row["gpu_type"] or "",
            user=str(row["user"]),
        )
        events.append(TimedPodEvent(submit_time=float(row["submit_time"]), duration=float(row["duration"]), pod=pod))
    events.sort(key=lambda e: e.submit_time)

    gpu_type_counts = df.loc[df["num_gpu"] > 0, "gpu_type"].value_counts()
    if gpu_type_counts.empty:
        gpu_type_counts = pd.Series({"A100": 1})
    types = gpu_type_counts.index.tolist()
    weights = (gpu_type_counts / gpu_type_counts.sum()).tolist()

    total_gpu_demand = df["num_gpu"].sum()
    total_cpu_demand = df["num_cpu"].sum()
    total_gpu_capacity = max(total_gpu_demand / max(target_utilization, 1e-6), gpus_per_node)
    n_nodes_sized = max(1, int(np.ceil(total_gpu_capacity / gpus_per_node)))
    n_nodes = n_nodes if n_nodes else n_nodes_sized
    cpu_per_node = max(4, int(np.ceil((total_cpu_demand / max(target_utilization, 1e-6)) / n_nodes)))

    rng = np.random.default_rng(0)
    assigned_types = rng.choice(types, size=n_nodes, p=weights)
    nodes = {
        f"node_{i:04d}": NodeResource(
            node_id=f"node_{i:04d}",
            milli_cpu_capacity=cpu_per_node * 1000,
            gpu_count=gpus_per_node,
            milli_gpu_capacity=gpus_per_node * 1000,
            gpu_type=str(assigned_types[i]),
        )
        for i in range(n_nodes)
    }
    return nodes, events
