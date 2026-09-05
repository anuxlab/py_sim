"""
Performance metrics for k8s_sim.simulation.TimeDrivenSimulator.

Seven metrics, as requested:
  1. Cluster utilization    - time-weighted average + full series
  2. Job throughput         - completed jobs per unit simulated time
  3. Job waiting time       - submission -> start, distribution
  4. Fairness               - Jain's index over per-tenant resource-time
  5. Starvation count       - jobs stuck waiting past a threshold (or never scheduled)
  6. Scheduling latency     - real wall-clock time per placement decision
  7. Interference intensity - degree of GPU-sharing co-tenancy, proxy metric

None of these exist in a *static* one-shot batch simulation (the rest of
k8s_sim) -- they all require the time dimension k8s_sim.simulation adds.

IMPORTANT CAVEATS (read before citing these numbers):
  - The trace has no real user/tenant identifiers, so fairness needs a
    tenant assignment. The default `tenant_key_fn` buckets pods into 10
    SYNTHETIC tenants via a stable hash of pod name -- purely so the metric
    has something non-trivial to compute over. Pass your own `tenant_key_fn`
    for anything you'd actually report.
  - Interference intensity is a coarse, UNCALIBRATED proxy: "average number
    of extra pods co-resident on the same physical GPU device, weighted by
    time." It is motivated by the well-documented fact that GPU-sharing
    causes real performance interference (see Xiao et al., "Gandiva:
    Introspective Cluster Scheduling for Deep Learning," OSDI 2018; Xiao et
    al., "AntMan: Dynamic Scaling on GPU Clusters for Deep Learning," OSDI
    2020), but the trace has no profiling data (achieved throughput,
    slowdown) to calibrate an actual performance-interference model against.
    Treat it as "how much sharing is happening," not "how much slower jobs
    actually ran."
  - Job duration comes from the real trace's (deletion_time - scheduled_time)
    where available, replayed against our own scheduler's start time (see
    k8s_sim.simulation's module docstring) -- so all duration-derived metrics
    (throughput, utilization-over-time) inherit whatever noise/estimation
    error exists in the trace's own timestamps, plus our default_duration_sec
    fallback for pods missing usable timestamps.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .resource import PodResource


def default_tenant_key_fn(pod: PodResource) -> str:
    """SYNTHETIC tenant assignment (the trace has no real one) -- stable
    hash of pod name into 10 buckets, purely to give the fairness metric
    non-trivial groups to compute over. Replace via tenant_key_fn for
    anything meaningful."""
    h = int(hashlib.sha1(pod.name.encode()).hexdigest(), 16)
    return f"tenant-{h % 10}"


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _jains_fairness_index(values: List[float]) -> float:
    """J(x) = (sum x_i)^2 / (n * sum x_i^2). 1.0 = perfectly fair (everyone
    got the same total), 1/n = maximally unfair (one tenant got everything)."""
    values = [v for v in values if v > 0]
    n = len(values)
    if n == 0:
        return 1.0
    s1 = sum(values)
    s2 = sum(v * v for v in values)
    if s2 == 0:
        return 1.0
    return (s1 * s1) / (n * s2)


@dataclass
class MetricsCollector:
    starvation_threshold_sec: float = 3600.0
    tenant_key_fn: Optional[Callable[[PodResource], str]] = None

    _arrivals: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    _starts: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    _completions: List[Tuple[str, float, float, float]] = field(default_factory=list)
    _never_scheduled: List[Tuple[str, float]] = field(default_factory=list)
    _scheduling_latencies_sec: List[float] = field(default_factory=list)
    _utilization_samples: List[Tuple[float, float, float]] = field(default_factory=list)
    _tenant_resource_time: Dict[str, float] = field(default_factory=dict)

    _device_occupancy: Dict[Tuple[str, int], int] = field(default_factory=dict)
    _interference_samples: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self._tenant_key_fn = self.tenant_key_fn or default_tenant_key_fn

    def record_arrival(self, pod: PodResource, arrival_time: float) -> None:
        tenant = self._tenant_key_fn(pod)
        self._arrivals[pod.name] = (arrival_time, tenant)

    def record_scheduling_attempt(self, latency_sec: float) -> None:
        self._scheduling_latencies_sec.append(latency_sec)

    def record_start(self, pod: PodResource, arrival_time: float, start_time: float,
                      wait_time: float, node_name: Optional[str] = None,
                      gpu_ids: Optional[List[int]] = None) -> None:
        self._starts[pod.name] = (start_time, wait_time)
        if node_name is not None and gpu_ids and pod.is_gpu_share():
            for gi in gpu_ids:
                key = (node_name, gi)
                self._device_occupancy[key] = self._device_occupancy.get(key, 0) + 1
            self._sample_interference(start_time)

    def record_departure(self, pod: PodResource, departure_time: float,
                          node_name: Optional[str] = None,
                          gpu_ids: Optional[List[int]] = None) -> None:
        start_time, _wait = self._starts.get(pod.name, (departure_time, 0.0))
        duration = max(departure_time - start_time, 0.0)
        self._completions.append((pod.name, start_time, departure_time, duration))

        _arrival_time, tenant = self._arrivals.get(pod.name, (start_time, "unknown"))
        resource_units = pod.milli_cpu + pod.total_milli_gpu()
        self._tenant_resource_time[tenant] = self._tenant_resource_time.get(tenant, 0.0) + resource_units * duration

        if node_name is not None and gpu_ids and pod.is_gpu_share():
            for gi in gpu_ids:
                key = (node_name, gi)
                if key in self._device_occupancy:
                    self._device_occupancy[key] = max(0, self._device_occupancy[key] - 1)
            self._sample_interference(departure_time)

    def record_never_scheduled(self, pod: PodResource, last_time: float) -> None:
        arrival_time, _tenant = self._arrivals.get(pod.name, (last_time, "unknown"))
        self._never_scheduled.append((pod.name, last_time - arrival_time))

    def record_utilization_sample(self, time: float, cpu_util: float, gpu_util: float) -> None:
        self._utilization_samples.append((time, cpu_util, gpu_util))

    def _sample_interference(self, time: float) -> None:
        excess = sum(max(0, c - 1) for c in self._device_occupancy.values())
        self._interference_samples.append((time, float(excess)))

    @staticmethod
    def _time_weighted_mean(samples: List[Tuple[float, float]]) -> float:
        if not samples:
            return 0.0
        if len(samples) == 1:
            return samples[0][1]
        samples = sorted(samples, key=lambda s: s[0])
        total_time = samples[-1][0] - samples[0][0]
        if total_time <= 0:
            return statistics.fmean(v for _, v in samples)
        acc = 0.0
        for (t0, v0), (t1, _v1) in zip(samples, samples[1:]):
            acc += v0 * (t1 - t0)
        return acc / total_time

    def summary(self) -> Dict[str, object]:
        wait_times = [w for _, w in self._starts.values()]
        never_waits = [w for _, w in self._never_scheduled]
        all_waits = wait_times + never_waits

        cpu_series = [(t, c) for (t, c, _g) in self._utilization_samples]
        gpu_series = [(t, g) for (t, _c, g) in self._utilization_samples]

        completed = len(self._completions)
        if self._utilization_samples:
            horizon = max(t for t, _, _ in self._utilization_samples) - \
                min(t for t, _, _ in self._utilization_samples)
        else:
            horizon = 0.0
        throughput_per_hour = (completed / horizon * 3600.0) if horizon > 0 else 0.0

        starved = sum(1 for w in wait_times if w > self.starvation_threshold_sec) + len(self._never_scheduled)

        latencies_ms = [x * 1000.0 for x in self._scheduling_latencies_sec]

        return {
            "cluster_utilization": {
                "cpu_mean": self._time_weighted_mean(cpu_series),
                "gpu_mean": self._time_weighted_mean(gpu_series),
                "series": self._utilization_samples,
            },
            "job_throughput_per_hour": throughput_per_hour,
            "jobs_completed": completed,
            "jobs_never_scheduled": len(self._never_scheduled),
            "job_waiting_time_sec": {
                "mean": statistics.fmean(all_waits) if all_waits else 0.0,
                "p50": _percentile(all_waits, 0.50),
                "p95": _percentile(all_waits, 0.95),
                "p99": _percentile(all_waits, 0.99),
                "max": max(all_waits) if all_waits else 0.0,
            },
            "fairness_jains_index": _jains_fairness_index(list(self._tenant_resource_time.values())),
            "num_tenants": len(self._tenant_resource_time),
            "starvation_count": starved,
            "scheduling_latency_ms": {
                "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
                "p95": _percentile(latencies_ms, 0.95),
                "max": max(latencies_ms) if latencies_ms else 0.0,
            },
            "interference_intensity": {
                "time_weighted_mean": self._time_weighted_mean(self._interference_samples),
                "max": max((v for _, v in self._interference_samples), default=0.0),
            },
        }
