"""
Demo: build a GPU-sharing cluster, generate a synthetic workload of typical
pod shapes, and compare all scheduling policies on scheduling success rate,
resource utilization, and resulting fragmentation.

Run:  python3 demo.py
"""

import random

from k8s_sim import Cluster, NodeResource, PodResource
from k8s_sim import policies as policy_mod
from k8s_sim.fragmentation import build_typical_pods


def make_cluster(num_nodes: int = 20, gpus_per_node: int = 8,
                  cpu_cores_per_node: int = 96, gpu_type: str = "V100") -> Cluster:
    nodes = {}
    for i in range(num_nodes):
        node_id = f"node-{i:02d}"
        nodes[node_id] = NodeResource(
            node_id=node_id,
            milli_cpu_capacity=cpu_cores_per_node * 1000,
            gpu_count=gpus_per_node,
            milli_gpu_capacity=gpus_per_node * 1000,
            gpu_type=gpu_type,
        )
    return Cluster(nodes)


# A representative pod-shape mix, loosely modeled after the kind of
# GPU-sharing traces the original paper studies: mostly small fractional-GPU
# inference/training jobs, plus some full-GPU and multi-GPU training jobs.
POD_SHAPES = [
    # (milli_cpu, milli_gpu, gpu_number, weight)
    (500, 250, 1, 25),    # small GPU-share inference pod (1/4 GPU)
    (1000, 500, 1, 20),   # medium GPU-share pod (1/2 GPU)
    (2000, 1000, 1, 25),  # full single-GPU job
    (4000, 1000, 2, 15),  # 2-GPU job
    (8000, 1000, 4, 10),  # 4-GPU job
    (2000, 0, 0, 5),      # CPU-only job
]


def generate_workload(n_pods: int, gpu_type: str = "V100", seed: int = 0) -> list:
    rng = random.Random(seed)
    weights = [w for *_, w in POD_SHAPES]
    pods = []
    for i in range(n_pods):
        cpu, gpu_milli, gpu_num, _ = rng.choices(POD_SHAPES, weights=weights, k=1)[0]
        pods.append(PodResource(
            pod_id=f"pod-{i:04d}",
            milli_cpu=cpu, milli_gpu=gpu_milli, gpu_number=gpu_num,
            gpu_type=gpu_type if gpu_num > 0 else "",
        ))
    rng.shuffle(pods)
    return pods


def run_policy(policy: str, n_pods: int = 85) -> dict:
    cluster = make_cluster()
    pods = generate_workload(n_pods)
    # Derive representative shapes straight from the workload being
    # scheduled (quantile-binned), rather than a hand-specified list --
    # see fragmentation.build_typical_pods's docstring for why.
    typical = build_typical_pods(pods)

    results = cluster.schedule_pods(pods, policy, typical_pods=typical)
    n_scheduled = sum(1 for r in results if r.node_id is not None)
    n_unscheduled = len(results) - n_scheduled

    cpu_util = sum(n.cpu_utilization() for n in cluster.node_list) / len(cluster.node_list)
    gpu_util = cluster.total_gpu_utilization()
    frag = cluster.fragmentation_score(typical)

    return {
        "policy": policy,
        "scheduled": n_scheduled,
        "unscheduled": n_unscheduled,
        "cpu_util": cpu_util,
        "gpu_util": gpu_util,
        "frag_score": frag,
    }


def main():
    policies = list(policy_mod.POLICIES.keys())
    rows = [run_policy(p) for p in policies]

    header = f"{'policy':<15}{'scheduled':>10}{'unscheduled':>13}{'cpu_util':>10}{'gpu_util':>10}{'frag_score':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['policy']:<15}{r['scheduled']:>10}{r['unscheduled']:>13}"
              f"{r['cpu_util']*100:>9.1f}%{r['gpu_util']*100:>9.1f}%{r['frag_score']*100:>11.1f}%")

    print("\nLower frag_score is better: it's the mean fraction of representative")
    print("pod shapes that could NOT be placed in each node's leftover capacity.")
    print("FGD is designed to directly minimize this quantity.")


if __name__ == "__main__":
    main()
