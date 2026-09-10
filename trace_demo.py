"""
Run all scheduling policies against the REAL production trace data shipped
in data/csv/ (1523 nodes, 8152 pods from a live heterogeneous GPU cluster --
see data/ORIGINAL_DATA_README.md).

Run:  python3 trace_demo.py [trace_file] [n_pods]
e.g.  python3 trace_demo.py openb_pod_list_gpushare100.csv 500
"""

import sys

from k8s_sim import Cluster
from k8s_sim import policies as policy_mod
from k8s_sim.fragmentation import build_typical_pods
from k8s_sim.trace import (
    DATA_DIR, list_available_traces, load_real_trace_nodes, load_real_trace_pods, reset_cluster,
)


def main():
    trace_file = sys.argv[1] if len(sys.argv) > 1 else None
    n_pods = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print("Available pod traces in data/csv/:")
    for t in list_available_traces():
        print(f"  - {t}")
    print()

    nodes_master = load_real_trace_nodes()  # openb_node_list_all_node.csv, 1523 nodes
    pod_path = None if trace_file is None else f"{DATA_DIR}/{trace_file}"
    pods = load_real_trace_pods(path=pod_path, limit=n_pods, sample=True, seed=0)
    typical = build_typical_pods(pods)

    print(f"Loaded {len(nodes_master)} nodes and {len(pods)} pods "
          f"({'default trace' if trace_file is None else trace_file}).")
    print(f"Derived {len(typical)} typical pod shapes from this workload.\n")

    header = f"{'policy':<15}{'scheduled':>10}{'unscheduled':>13}{'cpu_util':>10}{'gpu_util':>10}{'frag_score':>12}"
    print(header)
    print("-" * len(header))
    for policy in policy_mod.POLICIES:
        # fresh copy of the cluster per policy, so every policy starts from
        # the same, fully-empty capacity
        nodes = reset_cluster(nodes_master)
        cluster = Cluster(nodes)
        results = cluster.schedule_pods(pods, policy, typical_pods=typical)
        n_scheduled = sum(1 for r in results if r.node_id is not None)
        n_unscheduled = len(results) - n_scheduled
        cpu_util = sum(n.cpu_utilization() for n in cluster.node_list) / len(cluster.node_list)
        gpu_util = cluster.total_gpu_utilization()
        frag = cluster.fragmentation_score(typical)
        print(f"{policy:<15}{n_scheduled:>10}{n_unscheduled:>13}"
              f"{cpu_util*100:>9.1f}%{gpu_util*100:>9.1f}%{frag*100:>11.1f}%")


if __name__ == "__main__":
    main()
