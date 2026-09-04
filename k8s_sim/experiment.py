"""
Experiment / benchmark-suite driver, standing in for the original project's
experiments/run_scripts + scripts/analysis.py + experiments/analysis
(merge_*.py) pipeline: run bin/simon over many (policy, workload, scale,
seed) combinations, dump a "Cluster Analysis Results" block per run, then
merge everything into a handful of tidy CSVs for plotting.

Here it's one Python module: `run_experiment` produces the per-run analysis
(mirrors the log block the original prints after every `simon apply`), and
`run_benchmark_suite` sweeps a grid of configs and writes tidy CSVs to
experiments/results/ that `experiments/plot.py` consumes directly -- no log
scraping required.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

from .cluster import Cluster
from .resource import NodeResource
from .fragmentation import node_gpu_share_frag_amount, FRAG_TYPES
from .trace import load_nodes_csv, load_pods_csv, build_typical_pods, list_available_traces, DATA_DIR
from . import policies as policy_mod

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "experiments", "results")

# Curated subset of data/csv/openb_pod_list_*.csv, one representative per
# workload-mix category (mirrors the four alloc-bar categories in the
# original repo's experiments/plot/plot_openb_*_alloc_bar.py scripts).
WORKLOAD_CATEGORIES: Dict[str, str] = {
    "default": "openb_pod_list_default.csv",
    "cpu-heavy": "openb_pod_list_cpu300.csv",
    "gpu-share-heavy": "openb_pod_list_gpushare100.csv",
    "multi-gpu-heavy": "openb_pod_list_multigpu50.csv",
    "gpu-type-constrained": "openb_pod_list_gpuspec33.csv",
}


@dataclass
class RunConfig:
    policy: str
    workload: str            # trace filename, e.g. "openb_pod_list_gpushare100.csv"
    n_pods: int
    seed: int = 0
    sample: bool = True


@dataclass
class RunResult:
    policy: str
    workload: str
    n_pods: int
    seed: int
    scheduled: int
    unscheduled: int
    alloc_ratio_cpu: float
    alloc_ratio_gpu: float
    alloc_ratio_milli_gpu: float
    frag_ratio: float
    # fragmentation bucket breakdown, as percentages of total idle GPU-milli
    frag_pct: Dict[str, float] = field(default_factory=dict)
    elapsed_sec: float = 0.0


def run_experiment(config: RunConfig,
                    nodes: Optional[Sequence[NodeResource]] = None) -> RunResult:
    """Run one (policy, workload, scale, seed) experiment and produce the
    same headline numbers the original's 'Cluster Analysis Results' log
    block reports: allocation ratios per resource, and the Q1-Q4/XL/XR/
    NoAccess fragmentation breakdown."""
    t0 = time.time()

    node_list = list(nodes) if nodes is not None else load_nodes_csv()
    workload_path = config.workload if os.path.isabs(config.workload) else os.path.join(DATA_DIR, config.workload)
    trace_pods = load_pods_csv(path=workload_path, limit=config.n_pods, sample=config.sample, seed=config.seed)
    pods = [tp.pod for tp in trace_pods]
    typical = build_typical_pods(trace_pods)

    cluster = Cluster(node_list)
    result = cluster.schedule_pods(pods, config.policy, typical_pods=typical)
    util = cluster.utilization()
    frag_ratio = cluster.fragmentation_ratio(typical)

    # aggregate fragmentation bucket breakdown across all nodes, as % of total idle gpu-milli
    totals = {t: 0.0 for t in FRAG_TYPES}
    for node in cluster.node_list():
        amt = node_gpu_share_frag_amount(node, typical)
        for k, v in amt.items():
            totals[k] += v
    grand_total = sum(totals.values())
    frag_pct = {k: (v / grand_total * 100.0 if grand_total else 0.0) for k, v in totals.items()}

    milli_gpu_alloc = util["gpu_utilization"]  # already a fraction of total milli-gpu capacity

    return RunResult(
        policy=config.policy,
        workload=config.workload,
        n_pods=config.n_pods,
        seed=config.seed,
        scheduled=len(result.scheduled),
        unscheduled=len(result.unscheduled),
        alloc_ratio_cpu=util["cpu_utilization"],
        alloc_ratio_gpu=util["gpu_utilization"],
        alloc_ratio_milli_gpu=milli_gpu_alloc,
        frag_ratio=frag_ratio,
        frag_pct=frag_pct,
        elapsed_sec=time.time() - t0,
    )


def print_analysis(r: RunResult) -> None:
    """Pretty-print in the same spirit as the original's
    'Cluster Analysis Results' console block."""
    print(f"========== Cluster Analysis Results ({r.policy} / {r.workload}, n={r.n_pods}, seed={r.seed}) ==========")
    print("Allocation Ratio:")
    print(f"    MilliCpu : {r.alloc_ratio_cpu*100:5.1f}%")
    print(f"    Gpu      : {r.alloc_ratio_gpu*100:5.1f}%")
    for t in FRAG_TYPES:
        print(f"{t:14s}: {r.frag_pct.get(t, 0.0):5.2f}%")
    print(f"frag_ratio (non-Q3 share of idle GPU): {r.frag_ratio*100:5.2f}%")
    print(f"scheduled={r.scheduled} unscheduled={r.unscheduled} elapsed={r.elapsed_sec:.2f}s")
    print("=" * 60)


def run_benchmark_suite(policies: Optional[Sequence[str]] = None,
                         workloads: Optional[Sequence[str]] = None,
                         pod_counts: Sequence[int] = (300,),
                         seeds: Sequence[int] = (0, 1, 2),
                         node_sample_size: Optional[int] = None,
                         node_sample_seed: int = 0,
                         out_path: Optional[str] = None,
                         verbose: bool = True) -> List[RunResult]:
    """Sweep a grid of (policy x workload x n_pods x seed) and write a tidy
    CSV to experiments/results/benchmark_<timestamp>.csv (or `out_path`).
    Node list is loaded once and reused (fresh Cluster copy per run) for speed.
    Returns the list of RunResult so callers can also pass it straight to
    experiments/plot.py without touching disk.

    The full 1523-node cluster makes FGD's per-GPU trial scoring the
    dominant cost (O(candidates x gpus x typical_pods) per pod). For a quick
    sweep across many (policy, workload) combinations, set
    `node_sample_size` to a smaller random subset of nodes -- see
    `experiments/run_benchmark.py` for curated fast-vs-full presets.
    """
    policies = list(policies) if policies is not None else list(policy_mod.POLICIES.keys())
    workloads = list(workloads) if workloads is not None else list_available_traces()

    nodes = load_nodes_csv()
    if node_sample_size is not None and node_sample_size < len(nodes):
        import random
        nodes = random.Random(node_sample_seed).sample(nodes, node_sample_size)

    all_results: List[RunResult] = []

    total = len(policies) * len(workloads) * len(pod_counts) * len(seeds)
    i = 0
    for workload in workloads:
        for n_pods in pod_counts:
            for seed in seeds:
                for policy in policies:
                    i += 1
                    cfg = RunConfig(policy=policy, workload=workload, n_pods=n_pods, seed=seed)
                    r = run_experiment(cfg, nodes=nodes)
                    all_results.append(r)
                    if verbose:
                        print(f"[{i}/{total}] {policy:<16} {workload:<32} n={n_pods:<5} seed={seed} "
                              f"-> frag_ratio={r.frag_ratio*100:5.1f}% "
                              f"sched={r.scheduled}/{r.scheduled + r.unscheduled}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = out_path or os.path.join(RESULTS_DIR, f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    _write_results_csv(all_results, out_path)
    # also refresh a stable "latest" pointer that plot.py defaults to
    _write_results_csv(all_results, os.path.join(RESULTS_DIR, "latest.csv"))
    if verbose:
        print(f"\nWrote {len(all_results)} rows to {out_path}")
    return all_results


def _write_results_csv(results: Sequence[RunResult], path: str) -> None:
    if not results:
        return
    frag_keys = list(results[0].frag_pct.keys())
    fieldnames = ["policy", "workload", "n_pods", "seed", "scheduled", "unscheduled",
                  "alloc_ratio_cpu", "alloc_ratio_gpu", "alloc_ratio_milli_gpu",
                  "frag_ratio", "elapsed_sec"] + [f"frag_pct_{k}" for k in frag_keys]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            frag_pct = row.pop("frag_pct")
            for k, v in frag_pct.items():
                row[f"frag_pct_{k}"] = v
            writer.writerow(row)
