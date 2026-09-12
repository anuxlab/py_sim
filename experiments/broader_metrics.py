"""
Broader evaluation metrics for GPU-scheduling algorithm comparison, beyond
rejection_rate/mean_wait_time/fragmentation_score (already in
k8s_sim.experiment / full_scenario_matrix.py). These are metrics the
scheduling-research and MLOps/SRE communities commonly report and that a
single mean/rejection-rate number can hide entirely:

  - p50 / p99 wait time (not just mean and p95): mean can look fine while a
    long tail of jobs waits far longer -- p99 is standard in latency SLOs.
  - SLA violation rate: fraction of ADMITTED jobs whose wait time exceeds a
    threshold (default 60s). A policy can have a low mean wait and still
    violate SLOs for a meaningful minority of jobs; this is the metric that
    catches that, which p95/p99 alone only implies.
  - Jain's Fairness Index (Jain, Chiu & Hawe, 1984 -- the standard fairness
    metric in networking/scheduling literature) over PER-USER mean wait
    time (time-driven) or per-user admission rate (static/H-TAFM, which has
    no time dimension): J(x) = (sum x)^2 / (n * sum x^2), in (1/n, 1], 1.0 =
    perfectly equal treatment across users. A policy that minimizes overall
    rejection by starving a few users can look great on rejection_rate and
    terrible on this.
  - GPU-hours wasted: (1 - gpu_utilization) * total_gpu_capacity * makespan,
    in GPU-hours -- ties idle capacity to an actual cost-relevant unit
    instead of a dimensionless percentage.
  - Packing efficiency: admitted GPU-demand / allocated GPU-capacity at
    steady state -- distinguishes "idle because nothing wants to run" from
    "idle because what's running is badly packed", which raw utilization
    alone conflates.

Usage:
    python3 experiments/broader_metrics.py --traces-dir traces \
        --policies fgd,w_fgd,w_fgd_balanced,best_fit,least_requested \
        --seeds 1,2,3 --out broader_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from k8s_sim.cluster import Cluster
from k8s_sim.event_runtime import EventDrivenRunner, RunConfig
from k8s_sim.fragmentation import build_typical_pods, build_typical_pods_weighted
from k8s_sim.gputrace_bridge import load_gputrace_export
from k8s_sim.trace import load_nodes_csv, load_pods_csv, reset_cluster

_WEIGHT_AWARE = {"w_fgd", "w_fgd_balanced"}
SLA_THRESHOLD_S = 60.0


def jains_fairness_index(values: List[float]) -> float:
    """Jain, Chiu & Hawe (1984). 1.0 = perfectly fair (all equal),
    approaches 1/n = maximally unfair (all resource to one entity)."""
    values = [v for v in values if v is not None]
    if not values:
        return 1.0
    n = len(values)
    s = sum(values)
    if s == 0:
        return 1.0  # everyone got exactly zero -> vacuously "fair"
    return (s ** 2) / (n * sum(v ** 2 for v in values))


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]


def time_driven_broader_metrics(traces_dir: Path, policies: List[str], seeds: List[int],
                                 sla_threshold: float = SLA_THRESHOLD_S) -> List[dict]:
    rows = []
    scenario_dirs = sorted(p for p in traces_dir.iterdir() if (p / "pods.csv").exists())
    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        base_nodes, base_events = load_gputrace_export(scenario_dir / "pods.csv", scenario_dir / "nodes.csv")
        typical_pods = build_typical_pods([e.pod for e in base_events])
        w_shapes, w_weights = build_typical_pods_weighted([e.pod for e in base_events])
        user_of = {e.pod.pod_id: e.pod.user for e in base_events}
        gpu_of = {e.pod.pod_id: (e.pod.gpu_number or (e.pod.milli_gpu / 1000.0)) for e in base_events}
        total_gpu = sum(n.gpu_count for n in base_nodes.values())

        for policy in policies:
            weight_aware = policy in _WEIGHT_AWARE
            for seed in seeds:
                nodes = reset_cluster(base_nodes)
                cluster = Cluster(nodes)
                runner = EventDrivenRunner(cluster, RunConfig(policy=policy, seed=seed))
                result = runner.run(
                    base_events,
                    typical_pods=w_shapes if weight_aware else typical_pods,
                    typical_weights=w_weights if weight_aware else None,
                )

                waits = sorted(result.admitted_wait_times)
                sla_violations = sum(1 for w in waits if w > sla_threshold)
                n_admitted = result.n_admitted

                per_user_wait: Dict[str, List[float]] = defaultdict(list)
                for o in result.outcomes:
                    if o.admitted and o.wait_time is not None:
                        per_user_wait[user_of.get(o.pod_id, "unknown")].append(o.wait_time)
                per_user_mean_wait = [sum(v) / len(v) for v in per_user_wait.values()]

                makespan_h = result.makespan / 3600.0

                # TIME-WEIGHTED average GPU utilization, computed from each admitted
                # job's actual occupied interval -- NOT cluster.total_gpu_utilization()
                # after the run, which reflects an instantaneous snapshot at
                # simulation-end. For any trace that runs to completion (the normal
                # case), essentially every job has already finished and released by
                # then, so that snapshot reads ~0 regardless of how busy the cluster
                # was throughout the run -- a real, previously-unnoticed measurement
                # gap in k8s_sim.experiment's `final_gpu_utilization` column, not
                # something specific to these algorithms. GPU-hours actually used =
                # sum over admitted jobs of (occupied duration x GPUs held);
                # utilization = that divided by (makespan x total cluster GPUs).
                gpu_hours_used = 0.0
                for o in result.outcomes:
                    if o.admitted and o.start_time is not None and o.completion_time is not None:
                        occupied_h = (o.completion_time - o.start_time) / 3600.0
                        gpu_hours_used += occupied_h * gpu_of.get(o.pod_id, 0.0)
                total_gpu_hours_available = total_gpu * makespan_h
                avg_gpu_utilization = (gpu_hours_used / total_gpu_hours_available
                                        if total_gpu_hours_available > 0 else 0.0)
                gpu_hours_wasted = max(total_gpu_hours_available - gpu_hours_used, 0.0)

                rows.append({
                    "scenario": scenario, "algorithm": policy, "mode": "time-driven", "seed": seed,
                    "n_submitted": result.n_submitted, "n_admitted": n_admitted,
                    "rejection_rate": result.rejection_rate,
                    "p50_wait_time": percentile(waits, 0.50),
                    "p95_wait_time": percentile(waits, 0.95),
                    "p99_wait_time": percentile(waits, 0.99),
                    "sla_violation_rate": sla_violations / n_admitted if n_admitted else 0.0,
                    "jains_fairness_per_user_wait": jains_fairness_index(per_user_mean_wait),
                    "avg_gpu_utilization_time_weighted": avg_gpu_utilization,
                    "gpu_hours_wasted": gpu_hours_wasted,
                    "gpu_hours_used": gpu_hours_used,
                    "makespan_hours": makespan_h,
                })
    return rows


def static_broader_metrics(traces_dir: Path, policies: List[str]) -> List[dict]:
    """Static/batch-mode equivalent, for algorithms with no time dimension
    (relevant for w_fgd/w_fgd_balanced too, since they're directly
    comparable to H-TAFM here) -- fairness measured over per-user admission
    RATE (no wait times exist in a snapshot), plus packing efficiency."""
    rows = []
    scenario_dirs = sorted(p for p in traces_dir.iterdir() if (p / "pods.csv").exists())
    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        base_nodes = load_nodes_csv(scenario_dir / "nodes.csv")
        pods = load_pods_csv(scenario_dir / "pods.csv")
        typical_pods = build_typical_pods(pods)
        w_shapes, w_weights = build_typical_pods_weighted(pods)
        total_gpu_capacity_milli = sum(n.milli_gpu_capacity for n in base_nodes.values())
        total_demand_milli = sum(p.milli_gpu for p in pods)

        for policy in policies:
            weight_aware = policy in _WEIGHT_AWARE
            nodes = reset_cluster(base_nodes)
            cluster = Cluster(nodes)
            results = cluster.schedule_pods(
                pods, policy=policy,
                typical_pods=w_shapes if weight_aware else typical_pods,
                typical_weights=w_weights if weight_aware else None,
            )
            admitted_ids = {r.pod_id for r in results if r.node_id is not None}

            per_user_total: Dict[str, int] = defaultdict(int)
            per_user_admitted: Dict[str, int] = defaultdict(int)
            for p in pods:
                per_user_total[p.user] += 1
                if p.pod_id in admitted_ids:
                    per_user_admitted[p.user] += 1
            per_user_rate = [per_user_admitted[u] / per_user_total[u] for u in per_user_total]

            allocated_milli = sum(n.milli_gpu_capacity - n.remaining_milli_gpu for n in cluster.node_list)
            admitted_demand_milli = sum(p.milli_gpu for p in pods if p.pod_id in admitted_ids)
            packing_efficiency = admitted_demand_milli / allocated_milli if allocated_milli else 0.0

            rows.append({
                "scenario": scenario, "algorithm": policy, "mode": "static",
                "n_submitted": len(pods), "n_admitted": len(admitted_ids),
                "rejection_rate": 1 - len(admitted_ids) / len(pods) if pods else 0.0,
                "jains_fairness_per_user_admission": jains_fairness_index(per_user_rate),
                "packing_efficiency": packing_efficiency,
                "gpu_utilization": cluster.total_gpu_utilization(),
                "offered_load_ratio": total_demand_milli / total_gpu_capacity_milli if total_gpu_capacity_milli else 0.0,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True, type=Path)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sla-threshold", type=float, default=SLA_THRESHOLD_S)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-static", type=Path, default=None)
    args = ap.parse_args()

    policies = args.policies.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    td_rows = time_driven_broader_metrics(args.traces_dir, policies, seeds, args.sla_threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(td_rows[0].keys()))
        w.writeheader()
        w.writerows(td_rows)
    print(f"wrote {len(td_rows)} time-driven rows -> {args.out}", file=sys.stderr)

    if args.out_static:
        static_rows = static_broader_metrics(args.traces_dir, policies)
        with open(args.out_static, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(static_rows[0].keys()))
            w.writeheader()
            w.writerows(static_rows)
        print(f"wrote {len(static_rows)} static rows -> {args.out_static}", file=sys.stderr)


if __name__ == "__main__":
    main()
