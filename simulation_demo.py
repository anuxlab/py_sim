"""
Time-driven simulation demo: replays a real trace slice with actual arrival
timing and departures, against a chosen policy, and reports all 7
performance metrics (utilization, throughput, waiting time, fairness,
starvation, scheduling latency, interference).

This is a genuinely different simulation mode from demo.py/trace_demo.py
(which do static, one-shot batch placement with no time dimension) --
see k8s_sim/simulation.py's module docstring.

Run: python3 simulation_demo.py [n_pods] [n_nodes]
"""

import sys
import json
import random

from k8s_sim.trace import load_nodes_csv, load_pods_csv, build_typical_pods
from k8s_sim import Cluster
from k8s_sim.simulation import TimeDrivenSimulator


def run_one(policy, nodes, trace_pods, typical, starvation_threshold_sec):
    cluster = Cluster(nodes)
    sim = TimeDrivenSimulator(cluster, policy=policy, typical_pods=typical,
                               starvation_threshold_sec=starvation_threshold_sec)
    metrics = sim.run(trace_pods)
    return metrics.summary()


def main():
    n_pods = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    n_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    all_nodes = load_nodes_csv()
    gpu_nodes = [n for n in all_nodes if n.gpu_number > 0]
    nodes = random.Random(0).sample(gpu_nodes, min(n_nodes, len(gpu_nodes)))

    trace_pods = load_pods_csv(limit=n_pods, sample=True, seed=0)
    typical = build_typical_pods(trace_pods)

    policies = ["random", "best-fit", "fgd"]
    print(f"{n_nodes} GPU nodes, {n_pods} pods, comparing: {policies}\n")

    header = (f"{'policy':<12}{'cpu_util':>10}{'gpu_util':>10}{'throughput/hr':>14}"
              f"{'wait_p50':>10}{'wait_p95':>10}{'starved':>9}{'fairness':>10}"
              f"{'sched_lat_ms':>13}{'interfer.':>10}")
    print(header)
    print("-" * len(header))

    s = None
    for policy in policies:
        s = run_one(policy, nodes, trace_pods, typical, starvation_threshold_sec=1800.0)
        u = s["cluster_utilization"]
        w = s["job_waiting_time_sec"]
        lat = s["scheduling_latency_ms"]
        interf = s["interference_intensity"]
        print(f"{policy:<12}{u['cpu_mean']*100:>9.1f}%{u['gpu_mean']*100:>9.1f}%"
              f"{s['job_throughput_per_hour']:>14.2f}"
              f"{w['p50']:>10.0f}{w['p95']:>10.0f}{s['starvation_count']:>9d}"
              f"{s['fairness_jains_index']:>10.3f}{lat['mean']:>13.3f}"
              f"{interf['time_weighted_mean']:>10.4f}")

    print("\nFull summary for the last policy run (JSON, utilization series omitted):")
    s_full = dict(s)
    s_full["cluster_utilization"] = {k: v for k, v in s["cluster_utilization"].items() if k != "series"}
    s_full["cluster_utilization"]["n_samples"] = len(s["cluster_utilization"]["series"])
    print(json.dumps(s_full, indent=2))

    print("\nCaveats: fairness uses 10 SYNTHETIC tenants (the trace has no real user IDs);")
    print("interference_intensity is an uncalibrated co-tenancy proxy, not a validated")
    print("slowdown model. See k8s_sim/metrics.py's module docstring for details.")


if __name__ == "__main__":
    main()
