"""
Benchmark sweep driver.

Runs every registered policy against every gputrace scenario export found
under a directory, at multiple scales and seeds, in time-driven mode
(``EventDrivenRunner``) — this is the mode that actually exercises the
arrival-dynamics differences between scenarios; see
``event_runtime.py``/``gputrace_bridge.py`` docstrings for why static mode
can't. A static-mode sweep is also available for pure packing-quality
comparisons where time genuinely doesn't matter.

Expects a directory laid out as gputrace's ``generate-all`` + per-scenario
``export`` would produce:

    traces/
      bursty_arrivals/pods.csv
      bursty_arrivals/nodes.csv
      diurnal_pattern/pods.csv
      diurnal_pattern/nodes.csv
      ...

CLI:
    python -m k8s_sim.experiment --traces-dir traces/ --policies fgd,best_fit,random \
        --seeds 1,2,3 --scale 1.0,0.5,2.0 --out results.csv --mode time-driven
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from .cluster import Cluster
from .event_runtime import EventDrivenRunner, RunConfig
from .fragmentation import build_typical_pods, build_typical_pods_weighted
from .gputrace_bridge import load_gputrace_export
from .policies import list_policies
from .trace import reset_cluster

_WEIGHT_AWARE_POLICIES = {"w_fgd", "w_fgd_balanced"}


def _scale_events(events, factor: float):
    """Scale offered load by ``factor`` by subsampling (factor<1) or
    duplicating-with-jitter (factor>1) the event list — a cheap way to
    sweep contention level without regenerating traces from gputrace. For
    factor > 1, duplicated copies are offset by a small time jitter within
    each original inter-arrival gap so they don't all land at literally
    the same instant.
    """
    import numpy as np

    if factor == 1.0:
        return list(events)
    rng = np.random.default_rng(0)
    if factor < 1.0:
        n_keep = max(1, int(len(events) * factor))
        idx = np.sort(rng.choice(len(events), size=n_keep, replace=False))
        return [events[i] for i in idx]

    # factor > 1: duplicate with small jitter, preserve sortedness
    from .gputrace_bridge import TimedPodEvent
    from .resource import PodResource

    extra_needed = int(len(events) * (factor - 1))
    out = list(events)
    for i in range(extra_needed):
        src = events[i % len(events)]
        jitter = rng.uniform(0, 1.0)
        dup_pod = PodResource(
            pod_id=f"{src.pod.pod_id}_dup{i}",
            milli_cpu=src.pod.milli_cpu, milli_gpu=src.pod.milli_gpu,
            gpu_number=src.pod.gpu_number, gpu_type=src.pod.gpu_type, user=src.pod.user,
        )
        out.append(TimedPodEvent(submit_time=src.submit_time + jitter, duration=src.duration, pod=dup_pod))
    out.sort(key=lambda e: e.submit_time)
    return out


def run_time_driven_sweep(traces_dir: Path, policies: List[str], seeds: List[int],
                           scales: List[float]) -> List[dict]:
    rows = []
    scenario_dirs = sorted(p for p in traces_dir.iterdir() if (p / "pods.csv").exists())
    if not scenario_dirs:
        print(f"no scenario exports found under {traces_dir} (expected <scenario>/pods.csv + nodes.csv)",
              file=sys.stderr)

    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        base_nodes, base_events = load_gputrace_export(scenario_dir / "pods.csv", scenario_dir / "nodes.csv")

        for scale in scales:
            events = _scale_events(base_events, scale)
            typical_pods = build_typical_pods([e.pod for e in events])
            w_shapes, w_weights = build_typical_pods_weighted([e.pod for e in events])

            for policy in policies:
                weight_aware = policy in _WEIGHT_AWARE_POLICIES
                for seed in seeds:
                    nodes = reset_cluster(base_nodes)
                    cluster = Cluster(nodes)
                    runner = EventDrivenRunner(cluster, RunConfig(policy=policy, seed=seed))
                    result = runner.run(
                        events,
                        typical_pods=w_shapes if weight_aware else typical_pods,
                        typical_weights=w_weights if weight_aware else None,
                    )

                    rows.append({
                        "scenario": scenario,
                        "policy": policy,
                        "scale": scale,
                        "seed": seed,
                        "n_submitted": result.n_submitted,
                        "n_admitted": result.n_admitted,
                        "rejection_rate": result.rejection_rate,
                        "mean_wait_time": result.mean_wait_time,
                        "p95_wait_time": result.p95_wait_time,
                        "makespan": result.makespan,
                        "final_fragmentation_score": cluster.fragmentation_score(typical_pods),
                        "final_gpu_utilization": cluster.total_gpu_utilization(),
                    })
                    print(f"  {scenario:28s} {policy:16s} scale={scale:<5} seed={seed}  "
                          f"reject={result.rejection_rate:.3f} mean_wait={result.mean_wait_time:.1f}",
                          file=sys.stderr)
    return rows


def run_static_sweep(traces_dir: Path, policies: List[str], seeds: List[int]) -> List[dict]:
    """Packing-quality-only comparison: schedule every scenario's pods in
    submission order with no time dimension. Included for completeness /
    comparison against the time-driven results — this is what the
    original snapshot-only design could measure.
    """
    from .trace import load_nodes_csv, load_pods_csv

    rows = []
    scenario_dirs = sorted(p for p in traces_dir.iterdir() if (p / "pods.csv").exists())
    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        base_nodes = load_nodes_csv(scenario_dir / "nodes.csv")
        pods = load_pods_csv(scenario_dir / "pods.csv")
        typical_pods = build_typical_pods(pods)
        w_shapes, w_weights = build_typical_pods_weighted(pods)

        for policy in policies:
            weight_aware = policy in _WEIGHT_AWARE_POLICIES
            for seed in seeds:
                nodes = reset_cluster(base_nodes)
                cluster = Cluster(nodes)
                results = cluster.schedule_pods(
                    pods, policy=policy,
                    typical_pods=w_shapes if weight_aware else typical_pods,
                    typical_weights=w_weights if weight_aware else None,
                    seed=seed,
                )
                n_admitted = sum(1 for r in results if r.node_id is not None)
                rows.append({
                    "scenario": scenario,
                    "policy": policy,
                    "seed": seed,
                    "n_submitted": len(pods),
                    "n_admitted": n_admitted,
                    "rejection_rate": 1 - n_admitted / len(pods) if pods else 0.0,
                    "fragmentation_score": cluster.fragmentation_score(typical_pods),
                    "gpu_utilization": cluster.total_gpu_utilization(),
                })
    return rows


def _write_csv(rows: List[dict], out_path: Path) -> None:
    if not rows:
        print("no results to write", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces-dir", required=True, type=Path)
    p.add_argument("--policies", default=",".join(list_policies()))
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--scale", default="1.0")
    p.add_argument("--mode", choices=["time-driven", "static"], default="time-driven")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    policies = args.policies.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.mode == "time-driven":
        scales = [float(s) for s in args.scale.split(",")]
        rows = run_time_driven_sweep(args.traces_dir, policies, seeds, scales)
    else:
        rows = run_static_sweep(args.traces_dir, policies, seeds)

    _write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
