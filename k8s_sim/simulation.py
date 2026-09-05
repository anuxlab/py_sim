"""
Discrete-event, time-driven cluster simulation.

Everything else in k8s_sim (Cluster.schedule_pods, experiment.run_experiment)
is a *static, one-shot* simulation: hand it a batch of pods, it places them
all with no notion of time, queueing, or departure. That's the right tool
for measuring fragmentation (the paper's own methodology), but it can't
produce time-based metrics -- utilization *over time*, throughput, waiting
time, starvation, or interference all require an actual clock, an arrival/
departure timeline, and a queue for pods that don't fit yet.

This module adds that: a classic discrete-event simulation (DES) --
a priority queue of (arrival | departure) events, a FIFO pending queue for
pods that arrived but don't fit yet, and a retry of the queue on every
event (not just arrivals), since a departure can free up room for
something already waiting.

TRACE REPLAY METHODOLOGY: each pod's real *creation_time* (from the trace)
is used as its arrival time. Each pod's real *duration* (deletion_time -
scheduled_time, i.e. how long it actually ran once started in the real
cluster) is preserved and replayed against OUR scheduler's own start time --
so if our scheduler starts a pod later than the real trace did (because our
synthetic cluster is more contended, or a different policy queued it), the
pod still runs for its real recorded duration, just shifted in time. This is
the standard trace-replay approach: preserve intrinsic job properties
(resource demand, run length), replay timing against the policy under test.
Pods with no usable duration (deletion_time <= scheduled_time, or either
timestamp missing/-1 in the trace) fall back to `default_duration_sec`
(documented, configurable) -- see `k8s_sim.metrics` for how this affects
throughput/utilization interpretation.
"""

from __future__ import annotations

import heapq
import time as _time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .cluster import Cluster
from .resource import PodResource
from .fragmentation import TargetPod
from .trace import TracePod
from .metrics import MetricsCollector

ARRIVAL = "arrival"
DEPARTURE = "departure"

DEFAULT_DURATION_SEC = 3600.0  # 1 hour fallback when the trace has no usable duration for a pod


@dataclass(order=True)
class _Event:
    time: float
    seq: int                     # tiebreaker for heap stability (avoids comparing pods)
    kind: str = field(compare=False)
    pod: PodResource = field(compare=False)
    gpu_ids: Optional[List[int]] = field(compare=False, default=None)
    node_name: Optional[str] = field(compare=False, default=None)


def _job_duration(tp: TracePod, default_duration_sec: float) -> float:
    if tp.scheduled_time >= 0 and tp.deletion_time > tp.scheduled_time:
        return float(tp.deletion_time - tp.scheduled_time)
    return default_duration_sec


class TimeDrivenSimulator:
    """Replays a trace against `policy` with real arrival timing, a FIFO
    pending queue, and departures that release resources -- enabling
    utilization-over-time, throughput, waiting time, starvation, and
    (optionally) interference metrics. Scheduling latency (wall-clock time
    per decision) is measured directly regardless of simulated time.
    """

    def __init__(self, cluster: Cluster, policy: str,
                 typical_pods: Optional[Sequence[TargetPod]] = None,
                 default_duration_sec: float = DEFAULT_DURATION_SEC,
                 starvation_threshold_sec: float = 3600.0,
                 tenant_key_fn: Optional[Callable[[PodResource], str]] = None):
        self.cluster = cluster
        self.policy = policy
        self.typical_pods = typical_pods
        self.default_duration_sec = default_duration_sec
        self.metrics = MetricsCollector(
            starvation_threshold_sec=starvation_threshold_sec,
            tenant_key_fn=tenant_key_fn,
        )
        self._pending: List[Tuple[float, int, PodResource]] = []  # (arrival_time, seq, pod), FIFO by arrival
        self._seq = 0
        self._durations: dict = {}  # pod.name -> duration_sec (PodResource is frozen, can't stash on it)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _try_schedule(self, pod: PodResource, arrival_time: float, now: float,
                       heap: List[_Event]) -> bool:
        t0 = _time.perf_counter()
        placed = self.cluster.schedule_pod_tracked(pod, self.policy, typical_pods=self.typical_pods)
        latency = _time.perf_counter() - t0
        self.metrics.record_scheduling_attempt(latency)

        if placed is None:
            return False
        node_name, gpu_ids = placed
        wait = now - arrival_time
        self.metrics.record_start(pod, arrival_time=arrival_time, start_time=now, wait_time=wait,
                                   node_name=node_name, gpu_ids=gpu_ids)

        duration = self._durations.get(pod.name, self.default_duration_sec)
        heapq.heappush(heap, _Event(time=now + duration, seq=self._next_seq(), kind=DEPARTURE,
                                     pod=pod, gpu_ids=gpu_ids, node_name=node_name))
        return True

    def run(self, trace_pods: Sequence[TracePod], horizon_sec: Optional[float] = None) -> MetricsCollector:
        """Replay `trace_pods` (sorted by creation_time internally) as an
        arrival stream. `horizon_sec`, if given, caps simulated time --
        anything still pending at the horizon is counted as never-scheduled
        rather than run forever."""
        heap: List[_Event] = []
        ordered = sorted(trace_pods, key=lambda tp: tp.creation_time)
        base_time = ordered[0].creation_time if ordered else 0.0

        for tp in ordered:
            arrival = float(tp.creation_time - base_time)
            pod = tp.pod
            self._durations[pod.name] = _job_duration(tp, self.default_duration_sec)
            heapq.heappush(heap, _Event(time=arrival, seq=self._next_seq(), kind=ARRIVAL, pod=pod))
            self.metrics.record_arrival(pod, arrival_time=arrival)

        now = 0.0
        while heap:
            ev = heapq.heappop(heap)
            now = ev.time
            if horizon_sec is not None and now > horizon_sec:
                break

            if ev.kind == DEPARTURE:
                self.cluster.release_pod(ev.pod, ev.node_name, ev.gpu_ids)
                self.metrics.record_departure(ev.pod, departure_time=now,
                                               node_name=ev.node_name, gpu_ids=ev.gpu_ids)
            else:
                self._pending.append((ev.time, ev.seq, ev.pod))

            # retry the FIFO pending queue: a departure may free room for
            # something already waiting; a fresh arrival is itself a first attempt.
            progressed = True
            while progressed and self._pending:
                progressed = False
                for idx, (arrival_time, seq, pod) in enumerate(self._pending):
                    if self._try_schedule(pod, arrival_time, now, heap):
                        del self._pending[idx]
                        progressed = True
                        break

            util = self.cluster.utilization()
            self.metrics.record_utilization_sample(now, util["cpu_utilization"], util["gpu_utilization"])

        # anything left pending at the end of the event stream never got scheduled
        for arrival_time, _, pod in self._pending:
            self.metrics.record_never_scheduled(pod, last_time=now)

        return self.metrics
