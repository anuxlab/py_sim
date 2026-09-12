"""
Run every scheduling approach this repo has -- all 8 node-granularity
policies (k8s_sim.policies.POLICIES) AND all 3 H-TAFM variants
(k8s_sim.htafm, NUMA-vertex granularity) -- against every scenario gputrace
generates, and write one consolidated results CSV.

COMPARABILITY NOTE (read this before interpreting results): the node
policies and H-TAFM are fundamentally different placement engines (whole-
node bin-packing vs. NUMA-vertex hypergraph scoring), so this script uses
STATIC mode (schedule_pods runs is submission order, no arrivals/departures
over time) for both, since that's the only mode H-TAFM has -- it's a
one-shot placement algorithm, not a time-driven simulator. Node policies
also have a separate, richer TIME-DRIVEN comparison available via
`k8s_sim.experiment --mode time-driven` (rejection rate under realistic
timing, wait times) that H-TAFM cannot participate in; that's a second,
separate comparison this script doesn't attempt to merge into the same
table, to avoid implying a false apples-to-apples between modes.

MEMORY ASSUMPTION: gputrace's exported pods.csv/nodes.csv schema doesn't
carry a memory field (see gputrace's k8s_sim exporter). H-TAFM needs one
(it's a genuinely 3-dimensional CPU+Memory+GPU placement engine). We assume
8 MiB per milli-cpu on BOTH node capacity and pod demand -- the same ratio
k8s_sim.topology._memory_capacity_of already documents as its fallback for
node capacity when memory isn't tracked -- so this is a labeled, consistent
ASSUMPTION, not measured data. Real memory numbers would change H-TAFM's
absolute fragmentation scores; the *relative* comparison across scenarios
and variants should be far less sensitive to this, since the same ratio
is applied everywhere.

Usage:
    python3 experiments/full_scenario_matrix.py --traces-dir traces --out full_matrix_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from k8s_sim import policies as policy_mod
from k8s_sim.cluster import Cluster
from k8s_sim.fragmentation import build_typical_pods, build_typical_pods_weighted
from k8s_sim.htafm import VARIANTS as HTAFM_VARIANTS
from k8s_sim.htafm import HTAFMConfig, HTAFMScheduler
from k8s_sim.topology import MILLI, Topology, TopoDemand
from k8s_sim.trace import load_nodes_csv, load_pods_csv, reset_cluster

MEM_PER_MILLI_CPU = 8  # MiB per milli-cpu; see module docstring
_WEIGHT_AWARE_POLICIES = {"w_fgd", "w_fgd_balanced"}


@dataclass
class _NamedNode:
    """Adapts a k8s_sim.resource.NodeResource (.node_id) to what
    Topology.from_nodes expects (.name) -- see that method's docstring."""
    name: str
    milli_cpu_capacity: int
    milli_gpu_left_list: List[int]
    gpu_type: str


def _to_topo_demand(pod) -> TopoDemand:
    return TopoDemand(
        milli_cpu=pod.milli_cpu,
        memory_mib=pod.milli_cpu * MEM_PER_MILLI_CPU,
        milli_gpu=pod.milli_gpu,
        gpu_number=pod.gpu_number,
        gpu_type=pod.gpu_type,
        name=pod.pod_id,
    )


def _to_topology(nodes_dict) -> Topology:
    named = [_NamedNode(name=n.node_id, milli_cpu_capacity=n.milli_cpu_capacity,
                         milli_gpu_left_list=[MILLI] * n.gpu_count, gpu_type=n.gpu_type)
             for n in nodes_dict.values()]
    return Topology.from_nodes(named)


def run_node_policies(scenario: str, nodes_master, pods, typical, policies: List[str]) -> List[dict]:
    rows = []
    w_shapes, w_weights = build_typical_pods_weighted(pods)
    for policy in policies:
        weight_aware = policy in _WEIGHT_AWARE_POLICIES
        nodes = reset_cluster(nodes_master)
        cluster = Cluster(nodes)
        results = cluster.schedule_pods(
            pods, policy=policy,
            typical_pods=w_shapes if weight_aware else typical,
            typical_weights=w_weights if weight_aware else None,
        )
        n_admitted = sum(1 for r in results if r.node_id is not None)
        rows.append({
            "scenario": scenario, "algorithm": policy, "granularity": "node",
            "n_submitted": len(pods), "n_admitted": n_admitted,
            "rejection_rate": 1 - n_admitted / len(pods) if pods else 0.0,
            # always scored against the SAME standard, unweighted typical set
            # for every algorithm -- see module docstring's comparability note
            "fragmentation_or_unallocated": cluster.fragmentation_score(typical),
            "gpu_utilization": cluster.total_gpu_utilization(),
        })
    return rows


def run_htafm_variants(scenario: str, nodes_master, pods) -> List[dict]:
    rows = []
    demands = [_to_topo_demand(p) for p in pods]
    total_gpus = sum(n.gpu_count for n in nodes_master.values())
    for variant in HTAFM_VARIANTS:
        topo = _to_topology(nodes_master)  # fresh topology per variant
        cfg = (HTAFMConfig(variant="cut", pending_demands=demands[:5])
               if variant == "cut" else HTAFMConfig(variant=variant))
        sched = HTAFMScheduler(topo, cfg)
        result = sched.schedule(demands)
        n_admitted = len(result.scheduled)
        unallocated = sched.unallocated_gpus()
        rows.append({
            "scenario": scenario, "algorithm": f"h-tafm-{variant}", "granularity": "numa",
            "n_submitted": len(pods), "n_admitted": n_admitted,
            "rejection_rate": 1 - n_admitted / len(pods) if pods else 0.0,
            "fragmentation_or_unallocated": unallocated,
            "gpu_utilization": (total_gpus - unallocated) / total_gpus if total_gpus else 0.0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--policies", default=",".join(policy_mod.POLICIES))
    args = ap.parse_args()

    policies = args.policies.split(",")
    scenario_dirs = sorted(p for p in args.traces_dir.iterdir() if (p / "pods.csv").exists())
    if not scenario_dirs:
        print(f"no scenario exports found under {args.traces_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows: List[dict] = []
    for sd in scenario_dirs:
        scenario = sd.name
        nodes_master = load_nodes_csv(sd / "nodes.csv")
        pods = load_pods_csv(sd / "pods.csv")
        typical = build_typical_pods(pods)

        node_rows = run_node_policies(scenario, nodes_master, pods, typical, policies)
        htafm_rows = run_htafm_variants(scenario, nodes_master, pods)
        all_rows.extend(node_rows)
        all_rows.extend(htafm_rows)
        print(f"{scenario:28s} n_pods={len(pods):6d}  "
              f"node-policy reject range=[{min(r['rejection_rate'] for r in node_rows):.3f}, "
              f"{max(r['rejection_rate'] for r in node_rows):.3f}]  "
              f"h-tafm reject range=[{min(r['rejection_rate'] for r in htafm_rows):.3f}, "
              f"{max(r['rejection_rate'] for r in htafm_rows):.3f}]", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
