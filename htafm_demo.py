"""
Benchmark H-TAFM's 3 variants (Cut/Entropy/Hier) against each other, and
against the existing node-granularity policies (FGD, Best-Fit, Random),
using the metrics from Section 7 of the methodology: unallocated GPUs and
acceptance rate. Utilization is reported too since it's needed to sanity
check the comparison is apples-to-apples (same workload, same total GPU
capacity, just different placement granularity).

NOTE ON COMPARABILITY: H-TAFM places demands on NUMA-vertex granularity
(k8s_sim.topology) while the other policies place on whole-node granularity
(k8s_sim.cluster/policies). They're independent placement engines evaluated
on the same trace, not variations of one engine -- see docs/HTAFM.md.

Run: python3 htafm_demo.py [n_demands]
"""

import csv
import sys

from k8s_sim import Cluster
from k8s_sim.htafm import HTAFMConfig, HTAFMScheduler
from k8s_sim.topology import TopoDemand, load_topology_from_csv
from k8s_sim.trace import DEFAULT_POD_TRACE, load_real_trace_nodes, load_real_trace_pods, reset_cluster
from k8s_sim.trace import DATA_DIR as _DATA_DIR


def _load_demands(n_demands: int) -> list:
    """H-TAFM needs `memory_mib` per pod (Section 1.1's multi-resource
    demand vector), which PodResource doesn't carry -- read it straight
    from the raw trace CSV rather than through load_real_trace_pods()."""
    path = f"{_DATA_DIR}/{DEFAULT_POD_TRACE}"
    demands = []
    with open(path, newline="") as f:
        for row in list(csv.DictReader(f))[:n_demands]:
            num_gpu = int(row["num_gpu"])
            gpu_milli = int(row["gpu_milli"]) if row.get("gpu_milli", "") not in ("", None) else 0
            demands.append(TopoDemand(
                milli_cpu=int(row["cpu_milli"]),
                memory_mib=int(row["memory_mib"]),
                milli_gpu=gpu_milli if num_gpu > 0 else 0,
                gpu_number=num_gpu,
                gpu_type=(row.get("gpu_spec") or "").strip(),
                name=row["name"],
            ))
    return demands


def run_htafm(variant: str, demands: list, **topo_kwargs) -> dict:
    topo = load_topology_from_csv(**topo_kwargs)

    if variant == "cut":
        pending = [TopoDemand(milli_cpu=d.milli_cpu, memory_mib=d.memory_mib,
                               milli_gpu=d.milli_gpu, gpu_number=d.gpu_number)
                   for d in demands[:5]]  # small representative sample as "pending" reference set
        cfg = HTAFMConfig(variant="cut", pending_demands=pending)
    else:
        cfg = HTAFMConfig(variant=variant)

    sched = HTAFMScheduler(topo, cfg)
    result = sched.schedule(demands)
    total_gpu = sum(len(v.milli_gpu_left_list) for v in topo.vertices.values())
    return {
        "policy": f"h-tafm-{variant}",
        "scheduled": len(result.scheduled),
        "unscheduled": len(result.unscheduled),
        "unallocated_gpus": sched.unallocated_gpus(),
        "total_gpus": total_gpu,
        "acceptance_rate": len(result.scheduled) / max(1, len(demands)),
    }


def run_baseline(policy: str, nodes_master: dict, pods: list, typical: list) -> dict:
    nodes = reset_cluster(nodes_master)
    cluster = Cluster(nodes)
    results = cluster.schedule_pods(pods, policy, typical_pods=typical)
    n_scheduled = sum(1 for r in results if r.node_id is not None)
    total_gpu = sum(n.gpu_count for n in cluster.node_list)
    unallocated = sum(n.remaining_milli_gpu for n in cluster.node_list) / 1000.0
    return {
        "policy": policy,
        "scheduled": n_scheduled,
        "unscheduled": len(results) - n_scheduled,
        "unallocated_gpus": unallocated,
        "total_gpus": total_gpu,
        "acceptance_rate": n_scheduled / max(1, len(pods)),
    }


def main():
    n_demands = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    demands = _load_demands(n_demands)
    nodes_master = load_real_trace_nodes()
    pods = load_real_trace_pods(limit=n_demands)  # same prefix as _load_demands, for apples-to-apples
    from k8s_sim.fragmentation import build_typical_pods
    typical = build_typical_pods(pods)

    rows = []
    for variant in ["cut", "entropy", "hier"]:
        print(f"running h-tafm-{variant}...", file=sys.stderr)
        rows.append(run_htafm(variant, demands, nodes_per_rack=40))
    for policy in ["random", "best_fit", "fgd"]:
        print(f"running {policy}...", file=sys.stderr)
        rows.append(run_baseline(policy, nodes_master, pods, typical))

    header = f"{'policy':<16}{'scheduled':>10}{'unscheduled':>13}{'accept_rate':>13}{'unalloc_gpus':>14}{'total_gpus':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['policy']:<16}{r['scheduled']:>10}{r['unscheduled']:>13}"
              f"{r['acceptance_rate']*100:>12.1f}%{r['unallocated_gpus']:>14.1f}{r['total_gpus']:>12.0f}")

    print("\nNote: h-tafm-* places on NUMA-vertex granularity (multi-resource: CPU+Memory+GPU,")
    print("topology-aware); the others place on whole-node granularity (GPU-focused).")
    print("unallocated_gpus is the common ground metric from Section 7 of the methodology.")


if __name__ == "__main__":
    main()
