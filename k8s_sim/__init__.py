"""
k8s_sim: a GPU-cluster scheduling-policy simulator, with two modes.

Static / snapshot mode (the original design): schedule a fixed bag of pods
against a fixed node set, no notion of time. Good for pure packing-quality
questions.

    from k8s_sim import Cluster, load_nodes_csv, load_pods_csv, build_typical_pods
    nodes = load_nodes_csv("nodes.csv")
    pods = load_pods_csv("pods.csv")
    cluster = Cluster(nodes)
    results = cluster.schedule_pods(pods, policy="fgd", typical_pods=build_typical_pods(pods))
    print(cluster.fragmentation_score(build_typical_pods(pods)))

Time-driven / event mode (new — see event_runtime.py's docstring for why
this exists): schedule a gputrace-derived arrival trace over simulated
time, with queueing and release-on-completion.

    from k8s_sim import Cluster, load_gputrace_export, EventDrivenRunner, RunConfig, build_typical_pods
    nodes, events = load_gputrace_export("pods.csv", "nodes.csv")
    cluster = Cluster(nodes)
    runner = EventDrivenRunner(cluster, RunConfig(policy="fgd"))
    result = runner.run(events, typical_pods=build_typical_pods([e.pod for e in events]))
    print(result.rejection_rate, result.mean_wait_time, result.p95_wait_time)
"""

from .resource import NodeResource, PodResource
from .fragmentation import build_typical_pods, unfit_fraction, cluster_fragmentation_score
from .policies import POLICIES, get_policy, list_policies
from .cluster import Cluster, ScheduleResult
from .trace import load_nodes_csv, load_pods_csv, reset_cluster
from .gputrace_bridge import TimedPodEvent, load_gputrace_export, load_gputrace_dataframe
from .event_runtime import EventDrivenRunner, RunConfig, RunResult, PodOutcome

__all__ = [
    "NodeResource", "PodResource",
    "build_typical_pods", "unfit_fraction", "cluster_fragmentation_score",
    "POLICIES", "get_policy", "list_policies",
    "Cluster", "ScheduleResult",
    "load_nodes_csv", "load_pods_csv", "reset_cluster",
    "TimedPodEvent", "load_gputrace_export", "load_gputrace_dataframe",
    "EventDrivenRunner", "RunConfig", "RunResult", "PodOutcome",
]
