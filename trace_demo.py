"""
Run all 6 scheduling policies against the REAL production trace data shipped
in data/csv/ (from the original repo's data/ directory: 1523 nodes, 8152 pods
from a live heterogeneous GPU cluster).

Run:  python3 trace_demo.py [trace_file] [n_pods]
e.g.  python3 trace_demo.py openb_pod_list_gpushare100.csv 500
"""

import sys

from k8s_sim import Cluster, policies as policy_mod
from k8s_sim.trace import (
    load_nodes_csv, load_pods_csv, build_typical_pods,
    list_available_traces, GPU_NODE_TRACE,
)


def main():
    trace_file = sys.argv[1] if len(sys.argv) > 1 else None
    n_pods = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print("Available pod traces in data/csv/:")
    for t in list_available_traces():
        print(f"  - {t}")
    print()

    nodes_master = load_nodes_csv()  # openb_node_list_all_node.csv, 1523 nodes
    trace_pods = load_pods_csv(
        path=None if trace_file is None else f"data/csv/{trace_file}",
        limit=n_pods,
        sample=True,  # uniform random subset, so GPU pods aren't left out
    )
    pods = [tp.pod for tp in trace_pods]
    typical = build_typical_pods(trace_pods)

    print(f"Loaded {len(nodes_master)} nodes and {len(pods)} pods "
          f"({'default trace' if trace_file is None else trace_file}).")
    print(f"Derived {len(typical)} typical pod shapes covering >=60% of the workload.\n")

    header = f"{'policy':<15}{'scheduled':>10}{'unscheduled':>13}{'cpu_util':>10}{'gpu_util':>10}{'frag_ratio':>12}"
    print(header)
    print("-" * len(header))
    for policy in policy_mod.POLICIES:
        cluster = Cluster(nodes_master)  # fresh copy of the cluster per policy
        result = cluster.schedule_pods(pods, policy, typical_pods=typical)
        util = cluster.utilization()
        frag = cluster.fragmentation_ratio(typical)
        print(f"{policy:<15}{len(result.scheduled):>10}{len(result.unscheduled):>13}"
              f"{util['cpu_utilization']*100:>9.1f}%{util['gpu_utilization']*100:>9.1f}%"
              f"{frag*100:>11.1f}%")


if __name__ == "__main__":
    main()
