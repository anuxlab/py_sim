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

import sys

from k8s_sim.topology import load_topology_from_csv, TopoDemand
from k8s_sim.htafm import HTAFMScheduler, HTAFMConfig, compute_tafi
from k8s_sim.trace import load_nodes_csv, load_pods_csv, build_typical_pods
from k8s_sim import Cluster


def run_htafm(variant, trace_pods, n_demands, **topo_kwargs):
    topo = load_topology_from_csv(**topo_kwargs)
    demands = [
        TopoDemand(milli_cpu=tp.pod.milli_cpu, memory_mib=0, milli_gpu=tp.pod.milli_gpu,
                   gpu_number=tp.pod.gpu_number, gpu_type=tp.pod.gpu_type, name=tp.pod.name)
        for tp in trace_pods[:n_demands]
    ]
    # pull memory_mib back out of the raw CSV rows (PodResource doesn't carry it)
    import csv
    import os
    from k8s_sim.trace import DATA_DIR, DEFAULT_POD_TRACE
    mem_by_name = {}
    with open(os.path.join(DATA_DIR, DEFAULT_POD_TRACE), newline="") as f:
        for row in csv.DictReader(f):
            mem_by_name[row["name"]] = int(row["memory_mib"])
    for d in demands:
        d.memory_mib = mem_by_name.get(d.name, 0)

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


def run_baseline(policy, nodes, trace_pods, n_demands, typical):
    pods = [tp.pod for tp in trace_pods[:n_demands]]
    cluster = Cluster(nodes)
    result = cluster.schedule_pods(pods, policy, typical_pods=typical)
    total_gpu = sum(n.gpu_number for n in cluster.node_list())
    unallocated = sum(n.total_milli_gpu_left() for n in cluster.node_list()) / 1000.0
    return {
        "policy": policy,
        "scheduled": len(result.scheduled),
        "unscheduled": len(result.unscheduled),
        "unallocated_gpus": unallocated,
        "total_gpus": total_gpu,
        "acceptance_rate": len(result.scheduled) / max(1, len(pods)),
    }


def main():
    n_demands = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    trace_pods = load_pods_csv(limit=n_demands, sample=True, seed=0)
    nodes = load_nodes_csv()
    typical = build_typical_pods(trace_pods)

    rows = []
    for variant in ["cut", "entropy", "hier"]:
        print(f"running h-tafm-{variant}...", file=sys.stderr)
        rows.append(run_htafm(variant, trace_pods, n_demands, nodes_per_rack=40))
    for policy in ["random", "best-fit", "fgd"]:
        print(f"running {policy}...", file=sys.stderr)
        rows.append(run_baseline(policy, nodes, trace_pods, n_demands, typical))

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
