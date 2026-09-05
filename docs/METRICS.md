# Performance metrics system: implementation notes

Everything in `demo.py`, `trace_demo.py`, `k8s_sim/experiment.py`, and H-TAFM's
`htafm_demo.py` is a **static, one-shot batch simulation**: hand it a list of
pods, it places all of them with no notion of time, queueing, or departure.
That's the right tool for fragmentation analysis (the FGD paper's own
methodology), but none of the 7 requested metrics can be computed without an
actual clock, an arrival/departure timeline, and a queue for pods that don't
fit yet. This adds that layer.

## New files

| File | Contents |
|---|---|
| `k8s_sim/simulation.py` | `TimeDrivenSimulator` -- a discrete-event simulation (DES): a priority queue of arrival/departure events, a FIFO pending queue, retried on every event |
| `k8s_sim/metrics.py` | `MetricsCollector` -- records events, computes all 7 metrics in `.summary()` |
| `simulation_demo.py` | Runs it against real trace data, compares policies |
| `tests/test_simulation.py`, `tests/test_metrics.py` | 23 new tests |

## A correctness fix this required

`NodeResource.sub()` picks which specific GPU device(s) to use internally
(least-sufficient-first) but previously threw that choice away. That's fine
for one-shot batch placement (nothing is ever released), but it's wrong for
a simulation with departures: if you don't record *which* devices a pod
used, releasing it later (`NodeResource.add()` with no `gpu_ids`) re-derives
"least sufficient first" against whatever the node's state happens to be
*then* -- which may no longer correspond to what was actually freed,
silently corrupting capacity accounting over many arrival/departure cycles.

Fixed via `NodeResource.sub_with_gpu_ids()` (returns the indices used) and
`Cluster.schedule_pod_tracked()` / `Cluster.release_pod()` (the tracked
placement/release pair `TimeDrivenSimulator` uses). `Cluster.schedule_pod()`
itself is untouched -- this is purely additive, all 157 pre-existing tests
still pass unmodified.

## Trace replay methodology

Each pod's real `creation_time` is its arrival time. Each pod's real
duration (`deletion_time - scheduled_time`, i.e. how long it actually ran
once started in the real cluster) is preserved and replayed against *our*
scheduler's own start time -- so if our policy starts a job later than the
real trace did, the job still runs for its real recorded length, just
shifted. This is standard trace-replay practice: keep intrinsic job
properties (demand, run length) fixed, replay timing against the policy
under test. Pods with no usable trace duration (`deletion_time <=
scheduled_time`, or a missing/`-1` timestamp) fall back to
`default_duration_sec` (1 hour by default, configurable).

## The 7 metrics

| Metric | How it's computed | Caveat |
|---|---|---|
| Cluster utilization | Time-weighted mean of CPU/GPU utilization samples taken on every event, plus the full series for plotting | None -- this one's exact given the simulation |
| Job throughput | Completed jobs divided by simulated horizon, per hour | Depends on the trace-replay duration methodology above |
| Job waiting time | `start_time - arrival_time` per job; mean/p50/p95/p99/max | Jobs never scheduled by the end contribute their full `last_time - arrival_time` too |
| Fairness | Jain's index `(sum x)^2 / (n * sum x^2)` over per-tenant resource-time (`milli_cpu + milli_gpu` times duration) | **The trace has no real user/tenant IDs.** Default `tenant_key_fn` buckets pods into 10 synthetic tenants via a stable hash of pod name, purely so the metric has non-trivial groups. Pass your own `tenant_key_fn` for anything you'd actually report |
| Starvation count | Jobs that waited past `starvation_threshold_sec` (default 1h) + jobs never scheduled at all | Threshold is a config knob, pick one appropriate to your workload |
| Scheduling latency | Real wall-clock time (`time.perf_counter()`) around each placement decision, mean/p95/max in ms | Measures *our* Python implementation's speed, not a claim about real k8s scheduler latency |
| Interference intensity | Time-weighted mean of "excess co-tenancy" (sum over GPU devices of `max(0, occupants-1)`) among concurrently-running GPU-**share** pods only | **Uncalibrated proxy.** Motivated by real GPU-sharing interference (Xiao et al., "Gandiva," OSDI 2018; Xiao et al., "AntMan," OSDI 2020) but the trace has no profiling/slowdown data to calibrate an actual performance model against. Reads as "how much sharing is happening," not "how much slower jobs ran" |

## Why utilization/waiting/starvation might all show as ~0 by default

The real trace's arrivals are naturally spread out: sampling 150 pods from
`openb_pod_list_default.csv` spans on the order of ~2,200 simulated hours,
with a mean inter-arrival gap (~15h) similar to mean job duration (~17h) --
so with even a handful of GPU nodes, concurrent demand rarely exceeds
capacity and there's nothing to queue for. **This isn't a bug -- it reflects
the real trace's actual arrival sparsity.** To see genuine contention
(nonzero waiting time, starvation, interference), shrink the node pool:

```bash
python3 simulation_demo.py 150 1     # 1 GPU node -- clear contention: ~51% GPU util, p95 wait ~700s, 6 starved
python3 simulation_demo.py 150 3     # 3 GPU nodes -- default, much lighter contention
```

## Running it

```python
from k8s_sim.trace import load_nodes_csv, load_pods_csv, build_typical_pods
from k8s_sim import Cluster
from k8s_sim.simulation import TimeDrivenSimulator

nodes = load_nodes_csv()[:5]
trace_pods = load_pods_csv(limit=200, sample=True, seed=0)
typical = build_typical_pods(trace_pods)

cluster = Cluster(nodes)
sim = TimeDrivenSimulator(cluster, policy="fgd", typical_pods=typical,
                            starvation_threshold_sec=1800.0)
metrics = sim.run(trace_pods)
print(metrics.summary())
```

or compare policies directly:

```bash
python3 simulation_demo.py 150 1
```

## What's left as future work

- Wiring utilization-over-time into `k8s_sim/plotting.py` as a proper chart
  (the series is already collected in `summary()["cluster_utilization"]["series"]`)
- A real tenant/user model, if/when the trace or a different dataset provides one
- Calibrating interference intensity against actual measured slowdown data
  (would need a trace with achieved-throughput or completion-time-vs-isolated
  comparisons, which this trace doesn't have)
- Preemption/eviction (currently a scheduled job runs to its full recorded
  duration; no policy can preempt a running job to admit a higher-priority one)
