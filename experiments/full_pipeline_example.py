"""
End-to-end example: gputrace -> k8s_sim, in one process, no CSV files.

Requires both packages installed:
    pip install -e ../gputrace
    pip install -e .[gputrace]

Run:
    python experiments/full_pipeline_example.py
"""

from __future__ import annotations

import gputrace as gt
from k8s_sim import Cluster, EventDrivenRunner, RunConfig, build_typical_pods
from k8s_sim.gputrace_bridge import load_gputrace_dataframe


def main():
    # 1. Fit distributions to a source trace. Swap in a real one:
    #    fit = gt.analyze_file("alibaba2020", "pai_task_table_sample.csv")
    fit = gt.reference_fit()

    scenarios = ["baseline", "bursty_arrivals", "flash_crowd", "diurnal_pattern", "high_contention"]
    print(f"{'scenario':22s} {'policy':14s} {'reject':>8s} {'mean_wait':>10s} {'p95_wait':>10s} {'makespan':>10s}")

    for scenario in scenarios:
        df = gt.generate(scenario, fit, n_jobs=3000, seed=1, rate=3.0)
        nodes, events = load_gputrace_dataframe(df, n_nodes=20, gpus_per_node=8)
        typical_pods = build_typical_pods([e.pod for e in events])

        for policy in ["fgd", "best_fit", "random"]:
            # fresh cluster per policy so results are comparable
            fresh_nodes, _ = load_gputrace_dataframe(df, n_nodes=20, gpus_per_node=8)
            cluster = Cluster(fresh_nodes)
            result = EventDrivenRunner(cluster, RunConfig(policy=policy, seed=1)).run(
                events, typical_pods=typical_pods
            )
            print(f"{scenario:22s} {policy:14s} {result.rejection_rate:8.3f} "
                  f"{result.mean_wait_time:10.1f} {result.p95_wait_time:10.1f} {result.makespan:10.1f}")


if __name__ == "__main__":
    main()
