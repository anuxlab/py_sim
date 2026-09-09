# py_sim / k8s_sim

A GPU-cluster scheduling-policy simulator, in two modes:

* **static / snapshot** — schedule a fixed bag of pods against a fixed
  node set, no notion of time. Good for pure packing-quality comparisons.
* **time-driven / event-based** *(new)* — schedule a trace with real
  arrival times and durations over simulated time, with queueing and
  release-on-completion. This is what actually lets arrival-dynamics
  differences (bursty vs. smooth, flash crowds, diurnal cycles) show up in
  the results at all — see "Why two modes" below.

Built to plug directly into [`gputrace`](../gputrace)'s synthetic
workload scenarios, via `k8s_sim/gputrace_bridge.py`.

## Install

```bash
pip install -e .
# or, to use load_gputrace_dataframe() directly on a pandas DataFrame:
pip install -e ".[gputrace]"
```

Requires Python 3.10+, numpy (pandas only if using the DataFrame bridge).

## Why two modes (read this first)

The original design here was snapshot-only: `Cluster.schedule_pods()`
places a list of pods in order and never releases anything. That's fine
for asking "does policy X pack tighter than policy Y on this pod mix" —
but it cannot express *when* jobs arrive or *when* they finish, so it
cannot distinguish a workload with smooth arrivals from one with the exact
same jobs arriving in violent bursts. Every gputrace scenario that's
about arrival dynamics rather than job-size mix — `bursty_arrivals`,
`diurnal_pattern`, `flash_crowd`, `cold_start_storm`, `spot_preemption_churn`
— would collapse into indistinguishable input under snapshot scheduling.

`event_runtime.EventDrivenRunner` fixes this: a real discrete-event loop
over `(submit_time, duration)`, with a FIFO pending queue and
release-on-completion. Proof it matters — from this repo's own test suite:

```python
# 40 identical jobs, same span, same average rate.
# "smooth": one every ~1 time unit.
# "bursty": all 40 arrive in two waves of 20.
# Same cluster, same policy — only the arrival *pattern* differs.
assert r_bursty.rejection_rate > r_smooth.rejection_rate
```

and empirically, running actual gputrace scenarios through both:

```
bursty_arrivals  fgd   reject=0.165  mean_wait=911.1   <- real congestion, visible
baseline         fgd   reject=0.000  mean_wait=2719.7   <- same avg rate, no bursts
```

Static mode literally cannot produce this distinction — it has no
concept of "before" or "after."

## Quickstart

```python
from k8s_sim import (
    Cluster, RunConfig, EventDrivenRunner, build_typical_pods,
    load_gputrace_export,
)

nodes, events = load_gputrace_export("traces/bursty_arrivals/pods.csv",
                                      "traces/bursty_arrivals/nodes.csv")
cluster = Cluster(nodes)
typical = build_typical_pods([e.pod for e in events])

runner = EventDrivenRunner(cluster, RunConfig(policy="fgd", max_queue_time=None))
result = runner.run(events, typical_pods=typical)

print(result.rejection_rate, result.mean_wait_time, result.p95_wait_time, result.makespan)
```

Static/snapshot mode, if that's genuinely what you want to measure:

```python
from k8s_sim import Cluster, load_nodes_csv, load_pods_csv, build_typical_pods

nodes = load_nodes_csv("nodes.csv")
pods = load_pods_csv("pods.csv")
typical = build_typical_pods(pods)
cluster = Cluster(nodes)
results = cluster.schedule_pods(pods, policy="fgd", typical_pods=typical)
print(cluster.fragmentation_score(typical))
```

### End-to-end with gputrace, in one process

```python
import gputrace as gt
from k8s_sim import Cluster, RunConfig, EventDrivenRunner, build_typical_pods
from k8s_sim.gputrace_bridge import load_gputrace_dataframe

fit = gt.analyze_file("alibaba2020", "pai_task_table_sample.csv")
df = gt.generate("bursty_arrivals", fit, n_jobs=20_000, seed=1)

nodes, events = load_gputrace_dataframe(df, n_nodes=30, gpus_per_node=8)
cluster = Cluster(nodes)
result = EventDrivenRunner(cluster, RunConfig(policy="fgd")).run(
    events, typical_pods=build_typical_pods([e.pod for e in events])
)
```

## The benchmark sweep

```bash
# generate + export every scenario first (from the gputrace side):
gputrace generate-all --fit fit.json --n-jobs 20000 --seed 1 --out-dir /tmp/raw/
for f in /tmp/raw/*.csv; do
  name=$(basename "$f" .csv)
  gputrace export --input "$f" --format k8s_sim --out-dir "traces/$name" --n-nodes 20 --gpus-per-node 8
done

# then sweep every policy x scenario x seed x scale:
python -m k8s_sim.experiment --traces-dir traces/ \
    --policies fgd,best_fit,worst_fit,first_fit,random,gpu_packing,least_requested,round_robin \
    --seeds 1,2,3,4,5 --scale 0.5,1.0,2.0 --mode time-driven --out results.csv
```

`results.csv` is tidy — one row per (scenario, policy, scale, seed) — with
`rejection_rate`, `mean_wait_time`, `p95_wait_time`, `makespan`,
`final_fragmentation_score`, `final_gpu_utilization`. `--scale` subsamples
(< 1.0) or duplicates-with-jitter (> 1.0) the event list to sweep
contention level without regenerating traces.

Add `--mode static` to instead run the original snapshot-only comparison
(no queueing/release modeled) — useful as a contrast, not a replacement.

## Scheduling policies

`k8s_sim.list_policies()`: `random`, `first_fit`, `best_fit`, `worst_fit`,
`round_robin`, `gpu_packing`, `least_requested`, `fgd`
(fragmentation-gradient-descent — greedily minimizes the *increase* in
fragmentation score, not just raw leftover capacity; see
`policies.py`/`fragmentation.py` docstrings). Add a new one by writing a
function matching the signature documented in `policies.py` and adding it
to the `POLICIES` dict.

## Fragmentation scoring

Rather than scoring leftover capacity in the abstract, `fragmentation.py`
scores it by how many *representative pod shapes actually present in this
workload* (`build_typical_pods`, derived from quantile-binning the real
pod list) could still fit in what's left. Leftover resources that can't
fit any typical shape are fragmented in the sense that matters, regardless
of how large the raw numbers look.

## The gputrace bridge

`k8s_sim/gputrace_bridge.py` is the concrete integration point:

* `load_gputrace_export(pods_csv, nodes_csv)` — reads files written by
  `gputrace export --format k8s_sim`
* `load_gputrace_dataframe(df, n_nodes=0, gpus_per_node=8)` — same
  conversion directly from an in-memory unified-schema DataFrame, no CSV
  round trip

Both preserve `submit_time`/`duration` as `TimedPodEvent`s sorted for
`EventDrivenRunner`. Field mapping: `num_cpu × 1000 → milli_cpu`,
`num_gpu × gpu_milli_per_device → milli_gpu` (using the trace's
`gpu_milli` column when present — see gputrace's schema docs — else
assuming whole-device requests), `num_gpu (rounded) → gpu_number`,
`gpu_type` passthrough. Node topology (`nodes.csv`) is synthesized from
aggregate demand at a target utilization if not fixed — see
`gputrace/exporters/k8s_sim.py::_synthesize_nodes` for exactly how and why
that's a tunable starting point, not a validated cluster spec.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

31 tests: resource add/remove correctness, every policy's basic placement
behavior, best/worst-fit leftover-size sanity checks, fragmentation
scoring, an explicit fgd-vs-best-fit disagreement case, event-runtime
queueing/release/timeout/oversized-pod-rejection behavior, seed
reproducibility, and — the one that matters most — the bursty-vs-smooth
rejection-rate comparison that static scheduling structurally cannot
produce.

## Advanced experiments — next steps

1. **Preemption/eviction.** `spot_preemption_churn` currently only encodes
   preemption as early-terminated jobs (status=`killed`, shortened
   duration) in the *trace*; the scheduler itself has no eviction policy
   of its own yet. Next: let a policy actively evict a lower-priority
   running pod to admit a higher-priority arrival, via `NodeResource.remove()`
   + re-queueing the evicted pod, and report an eviction-count metric.
2. **SLO-aware admission.** `max_queue_time` today is a single global
   config; make it per-job (from a `priority`/`deadline` column gputrace
   could add) and report SLO-violation rate as a first-class metric
   alongside rejection rate.
3. **Autoscaling.** Let `Cluster` add/remove nodes in response to queue
   depth crossing a threshold, and measure time-to-scale vs. rejection
   rate tradeoffs under each scenario — directly answers "how many spare
   nodes do we actually need to absorb a flash crowd."
4. **Cost model.** Attach a $/node-hour to node types and report
   cost-per-admitted-job alongside performance metrics, so scheduler
   comparisons aren't purely throughput/latency — turns the sweep into a
   genuine Pareto-frontier (cost vs. rejection vs. p95 wait) exploration
   across policies.
5. **Multi-tenant fairness.** `moe_expert_load_skew`'s Zipf-skewed users
   are in the trace already; add a fairness metric (e.g. Jain's index over
   per-user wait time or admitted share) to the sweep output so a policy
   that looks good in aggregate but starves the long tail of users is
   visible.
6. **Statistical rigor on the sweep.** `experiment.py` already runs
   multiple seeds; add confidence intervals / paired significance tests
   (e.g. Wilcoxon signed-rank across matched seeds) to the output so
   "policy A beats B" claims from the sweep are backed by more than a
   single point estimate.
