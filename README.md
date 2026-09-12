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

## Running every algorithm against every gputrace scenario

`experiments/full_scenario_matrix.py` runs all 8 node-granularity policies
*and* all 3 H-TAFM variants (11 algorithms total) against every scenario a
`gputrace generate-all` + `gputrace export --format k8s_sim` run produces,
in static (batch) mode -- the only mode H-TAFM has, so it's the only mode
that can compare all 11 on equal footing. See that script's module
docstring for the full comparability and memory-assumption caveats before
trusting absolute numbers (short version: H-TAFM needs a memory dimension
gputrace's export schema doesn't carry, so one is assumed at a fixed
MiB-per-milli-cpu ratio; static mode also can't see any scenario whose
only distinguishing feature is *timing*, not resource size -- several
gputrace scenarios are timing-only, see the caveat below).

```bash
gputrace generate-all --n-jobs 300 --seed 42 --out-dir /tmp/gt_traces
mkdir -p traces
for f in /tmp/gt_traces/*.csv; do
  name=$(basename "$f" .csv)
  gputrace export --input "$f" --format k8s_sim --out-dir "traces/$name"
done
python3 experiments/full_scenario_matrix.py --traces-dir traces --out full_matrix_results.csv

# for the 8 node-granularity policies' realistic, time-driven comparison
# (rejection rate under actual arrival timing, wait times) -- H-TAFM can't
# participate, it has no time-driven mode:
python3 -m k8s_sim.experiment --traces-dir traces \
    --policies fgd,best_fit,random,worst_fit,least_requested,round_robin,gpu_packing,first_fit \
    --seeds 1,2,3 --mode time-driven --out timedriven_results.csv
```

**A genuine finding from running this**: at the time this was written, 5 of
gputrace's 14 scenarios (`checkpoint_io_burst`, `heavy_tail_demand`,
`long_context_kv_pressure`, `moe_expert_load_skew`, `spot_preemption_churn`)
produced pod-level `num_gpu`/`num_cpu`/`duration`/`gpu_type` values
byte-identical to `baseline` at the seed tested -- only `submit_time`
differed. That's expected/correct for the scenarios whose whole design is
about arrival *timing* (checkpoint bursts, spot churn), and their effect
does show up clearly in time-driven wait times. But it's a real, worth-
checking discrepancy for `heavy_tail_demand` specifically, whose own name
and gputrace's documentation describe a *resource-size* change (a fatter
GPU/CPU demand tail) that wasn't observed in this run -- worth verifying
against gputrace's current `generators/stress_scenarios.py` if you're
relying on that scenario's resource-size behavior specifically.

## H-TAFM (Hypergraph-based Topology-Aware Fragmentation Metric)

`k8s_sim/topology.py` + `k8s_sim/htafm.py` implement a second, independent
placement engine operating at NUMA-vertex granularity (one level below the
whole-node granularity everything else in this repo uses), scoring
placements against a weighted Socket/Server/Rack hypergraph rather than
flat leftover capacity. It's a from-the-methodology implementation with
citations and documented ambiguity resolutions in `docs/HTAFM.md` — worth
reading before using it, since a couple of interpretation calls were
required to turn the source methodology into working code.

```bash
python3 htafm_demo.py 300   # compares h-tafm-{cut,entropy,hier} against
                             # random/best_fit/fgd on the real bundled trace
```

Because it's a different placement granularity, not a different scoring
function, it isn't registered in `k8s_sim.policies.POLICIES` — it has its
own `Topology`/`HTAFMScheduler`, built via `load_topology_from_csv()` on
the same real `data/csv/openb_node_list_*.csv` data everything else here
uses (NUMA/socket/rack structure is synthesized on top, since the trace
itself has no sub-node topology — see `topology.py`'s module docstring for
exactly what's real vs. synthesized).

## Loading the real bundled trace data

Separately from the gputrace bridge, `k8s_sim/trace.py` also reads the
real production trace shipped in `data/csv/` (1523 nodes / 8152 pods from
a live heterogeneous GPU cluster — see `data/ORIGINAL_DATA_README.md`),
via `load_real_trace_nodes()` / `load_real_trace_pods()`. These are
separate functions from `load_nodes_csv()`/`load_pods_csv()`, which read
the *gputrace-export* CSV schema — the two schemas (raw Alibaba columns
like `sn`/`cpu_milli`/`gpu`/`model` vs. gputrace's `node_id`/
`milli_cpu_capacity`/`gpu_count`/`gpu_type`) are different, so both loader
pairs are kept, rather than one silently guessing which schema a given
CSV is in.

```bash
python3 trace_demo.py openb_pod_list_gpushare100.csv 500
```

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

## Using this together with gputrace (full local walkthrough)

Every command below is exactly what CI's `gputrace-integration` job runs
(`.github/workflows/ci.yml`) — this is copy-pasteable, not illustrative.

```bash
# 0. Get both repos side by side and install both
git clone https://github.com/anuxlab/simulated_data_generation_GPU_Scheduling.git gputrace
git clone https://github.com/anuxlab/py_sim.git
cd py_sim
pip install -e ".[gputrace]"       # numpy + pandas (for the DataFrame bridge)
pip install -e ../gputrace         # or: pip install "git+https://github.com/anuxlab/simulated_data_generation_GPU_Scheduling.git"

# 1. Fit distributions to a real trace (optional -- every gputrace command
#    below works with no --fit at all, using its built-in reference fit,
#    if you just want to try the pipeline without a real trace on hand)
gputrace analyze --loader alibaba2020 --input pai_task_table_sample.csv --out fit.json

# 2. Generate every stress scenario and export each to k8s_sim's native format
gputrace generate-all --fit fit.json --n-jobs 20000 --seed 1 --out-dir /tmp/gt_traces
mkdir -p traces
for f in /tmp/gt_traces/*.csv; do
  name=$(basename "$f" .csv)
  gputrace export --input "$f" --format k8s_sim --out-dir "traces/$name" --n-nodes 20 --gpus-per-node 8
done

# 3. Sweep every scheduling policy against every scenario, in time-driven mode
python -m k8s_sim.experiment --traces-dir traces \
    --policies fgd,best_fit,random,worst_fit,least_requested,round_robin,gpu_packing,first_fit \
    --seeds 1,2,3 --mode time-driven --out results.csv

# 4. ...or the equivalent, already wired up, in one Python process:
python3 experiments/full_pipeline_example.py
```

`results.csv` is one row per (scenario, policy, scale, seed) with
`rejection_rate`, `mean_wait_time`, `p95_wait_time`, `makespan`,
`final_fragmentation_score`, `final_gpu_utilization` — see "The benchmark
sweep" above for column details and `--scale` sweeping.

## Recent CI fixes (what changed and why)

The CI workflow was failing because a prior refactor commit (rewriting
`resource.py`/`cluster.py`/`trace.py` around the new gputrace integration)
also deleted `k8s_sim/topology.py`, `htafm.py`, `simulation.py`,
`metrics.py`, `k8s_sim/plotting.py`, `experiments/run_benchmark.py`, and
two test files — while `.github/workflows/ci.yml` and four top-level demo
scripts (`demo.py`, `trace_demo.py`, `htafm_demo.py`, `simulation_demo.py`)
still referenced the old files and the pre-refactor API (old field names
like `NodeResource(name=..., milli_cpu_left=...)` instead of the current
`NodeResource(node_id=..., milli_cpu_capacity=...)`).

Resolution, file by file:

* **`topology.py` + `htafm.py`** — restored from git history. Both are
  self-contained (no dependency on the rewritten core), still documented
  in `docs/HTAFM.md`, and the new README never mentioned dropping them —
  this looked like collateral damage from the refactor, not an
  intentional removal, so they're back. `htafm_demo.py` was rewritten
  against the current `trace.py`/`resource.py` API; `load_real_trace_nodes()`/
  `load_real_trace_pods()` were added to `trace.py` (additive only — the
  gputrace-facing `load_nodes_csv()`/`load_pods_csv()` are untouched) so
  both `topology.py` and `trace_demo.py` can read the real bundled data
  again.
* **`demo.py` / `trace_demo.py`** — rewritten against the current
  `PodResource`/`NodeResource`/`Cluster`/`fragmentation` API. No missing
  functionality, just stale field names and return types from before the
  refactor.
* **`simulation.py` / `metrics.py` / `simulation_demo.py`** — **not**
  restored. `simulation.py` depended on types that were themselves removed
  in the same rewrite (`fragmentation.TargetPod`, `trace.TracePod`), and
  the README already describes `event_runtime.EventDrivenRunner` as its
  intentional successor — reviving code built on already-gone internals
  looked like the wrong call here. `simulation_demo.py` is deleted;
  `experiments/full_pipeline_example.py` is the current equivalent.
* **`experiments/run_benchmark.py` + `k8s_sim/plotting.py`** — **not**
  restored; they produced a different (node-count-sweep, static-mode-only)
  CSV schema that predates the gputrace integration. `k8s_sim.experiment`
  is the current, actively-used benchmark driver and already produces a
  plain tabular CSV — see the "gputrace-integration" and "benchmark" CI
  jobs. No plotting step was re-added; that's a reasonable follow-up if
  you want charts, not something re-created speculatively here.
* **CI workflow** — `new-policy-check` now points at the consolidated
  `tests/test_k8s_sim.py -k "every_policy"` (the old `test_cluster.py` /
  `test_policy_contract.py` it referenced are gone, superseded by that
  file). The `test` job's benchmark+plotting step was replaced with a
  self-contained static-mode smoke test (no gputrace dependency, fast). A
  new `gputrace-integration` job installs gputrace from GitHub and runs
  the real cross-repo pipeline above end to end, uploading its results as
  an artifact. The `benchmark` job now does the same at full scale instead
  of calling the deleted `run_benchmark.py`.
* Added a `.gitignore` — there wasn't one, which is part of how build
  artifacts/pycache made it easy to lose track of what was intentionally
  deleted vs. accidentally dropped during the refactor.

### Round 2: after the first fix, `new-policy-check` failed and `benchmark` ran 3h44m

Two real problems turned up in the first actual Actions run of the fixes above:

* **`new-policy-check` failed in ~10s with pytest exit code 2** ("Interrupted:
  1 error during collection"). Root cause: that job's install step was
  `pip install pytest` — it never installed `k8s_sim` itself (or its one
  dependency, `numpy`), so `tests/test_k8s_sim.py`'s `import k8s_sim` failed
  before any test could run. Reproduced locally by mirroring the exact broken
  step; fixed by installing `-e ".[dev]"` instead of bare `pytest`.

* **`benchmark` ran for 3h44m and had to be manually canceled.** This wasn't a
  hang — profiling `k8s_sim/event_runtime.py` at increasing job counts (200
  through 4000 jobs) showed clearly *superlinear* growth (2x the jobs → ~4x
  the time), not the roughly-linear growth you'd want from a discrete-event
  simulator. The cause: on every pod-release event, the pending-queue drain
  re-scanned the *entire* remaining pending list in a `while
  pending_progress` loop that repeated the full scan again for every
  successful placement within that same release event. Since placing a
  pending pod only *consumes* cluster capacity — it never frees more — a
  second pass over the same list can never place anything a first pass
  didn't already catch, so that repeated re-scanning was pure waste. Removed
  it (single pass per release event now); verified the output is
  byte-for-byte identical to the old version on the same inputs before
  shipping the change, and it's ~1.5-1.6x faster as a direct result. The
  *default* `benchmark` scale was also reduced from 20,000 jobs x 3 seeds x
  all 8 policies (336 runs, which — even after the speedup — is genuinely
  multiple hours, not a hang, at that size) to 3,000 jobs x 2 seeds x all 8
  policies (224 runs, measured at ~15-20 minutes total). The full-scale sweep
  is still available on demand via `workflow_dispatch` inputs
  (`benchmark_n_jobs`, `benchmark_seeds`) if you deliberately want it — see
  the comments in `.github/workflows/ci.yml`'s `benchmark` job.

  **This remaining superlinear-ish scaling is architectural, not a bug I
  patched around**: `event_runtime.py`'s pending-queue drain is still a
  linear rescan of the queue on every release event, so total cost is
  roughly O(n × average queue depth), which approaches O(n²) whenever a
  scenario's backlog grows proportionally with job count (e.g. an
  under-provisioned cluster relative to demand). The single-pass fix removes
  a large *constant-factor* waste, not the underlying complexity class. If
  you want to run very large (50k+ job) time-driven sweeps routinely, a
  proper fix would replace the linear pending-list scan with an indexed
  structure (e.g. bucketing pending pods by resource-shape so a release only
  triggers a lookup against pods that could plausibly now fit, not a scan of
  every pod regardless of shape) — a bigger change than this fix, and one
  that touches scheduling-semantics code, so it's flagged here rather than
  made unilaterally.

  Timeouts (`timeout-minutes`) were also added to every job as a safety net,
  so a future regression fails within a bounded window instead of quietly
  running for hours again.

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
