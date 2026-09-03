"""
Demo: build a GPU-sharing cluster, generate a synthetic workload of typical
pod shapes, and compare all 6 scheduling policies on scheduling success rate,
resource utilization, and resulting fragmentation.

Run:  python3 demo.py
"""

import random

from k8s_sim import PodResource, NodeResource, TargetPod, Cluster
from k8s_sim.fragmentation import build_typical_pods_uniform


def make_cluster(num_nodes: int = 20, gpus_per_node: int = 8,
                  cpu_cores_per_node: int = 96, gpu_type: str = "V100") -> Cluster:
    nodes = []
    for i in range(num_nodes):
        nodes.append(NodeResource(
            name=f"node-{i:02d}",
            milli_cpu_left=cpu_cores_per_node * 1000,
            milli_cpu_capacity=cpu_cores_per_node * 1000,
            milli_gpu_left_list=[1000] * gpus_per_node,
            gpu_type=gpu_type,
        ))
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
            milli_cpu=cpu, milli_gpu=gpu_milli, gpu_number=gpu_num,
            gpu_type=gpu_type if gpu_num > 0 else "",
            name=f"pod-{i:04d}",
        ))
    rng.shuffle(pods)
    return pods


def typical_pods_from_shapes(gpu_type: str = "V100") -> list:
    total_w = sum(w for *_, w in POD_SHAPES)
    return [
        TargetPod(
            pod=PodResource(milli_cpu=cpu, milli_gpu=gpu_milli, gpu_number=gpu_num,
                             gpu_type=gpu_type if gpu_num > 0 else ""),
            percentage=w / total_w,
        )
        for cpu, gpu_milli, gpu_num, w in POD_SHAPES
    ]


def run_policy(policy: str, n_pods: int = 85) -> dict:
    cluster = make_cluster()
    pods = generate_workload(n_pods)
    typical = typical_pods_from_shapes()

    result = cluster.schedule_pods(pods, policy, typical_pods=typical)
    util = cluster.utilization()
    frag = cluster.fragmentation_ratio(typical)

    return {
        "policy": policy,
        "scheduled": len(result.scheduled),
        "unscheduled": len(result.unscheduled),
        "cpu_util": util["cpu_utilization"],
        "gpu_util": util["gpu_utilization"],
        "frag_ratio": frag,
    }


def main():
    policies = ["random", "best-fit", "dot-product", "gpu-packing", "gpu-clustering", "fgd"]
    rows = [run_policy(p) for p in policies]

    header = f"{'policy':<15}{'scheduled':>10}{'unscheduled':>13}{'cpu_util':>10}{'gpu_util':>10}{'frag_ratio':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['policy']:<15}{r['scheduled']:>10}{r['unscheduled']:>13}"
              f"{r['cpu_util']*100:>9.1f}%{r['gpu_util']*100:>9.1f}%{r['frag_ratio']*100:>11.1f}%")

    print("\nLower frag_ratio is better: it's the fraction of idle GPU memory")
    print("that's stranded (too small on every device to fit a typical pod).")
    print("FGD is designed to directly minimize this quantity.")


if __name__ == "__main__":
    main()
