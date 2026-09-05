# k8s_sim — standalone Python port of `kubernetes-scheduler-simulator`

This is a dependency-free Python re-implementation of the **core scheduling
logic** from [hkust-adsl/kubernetes-scheduler-simulator](https://github.com/hkust-adsl/kubernetes-scheduler-simulator)
("Simon"), the simulator behind the USENIX ATC'23 paper *"Beware of
Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient
Descent"*, **plus 5 additional benchmark baselines from the scheduling
literature**, a CI-enforced compatibility contract for adding new policies,
the original's real production trace data, and a benchmark/analysis/plotting
pipeline.

The original is a Go program that drives the **real Kubernetes scheduler
framework** against a fake API server (built on Alibaba's open-simulator).
That's the right design for validating against real k8s manifests and
traces, but it means you need Go, `k8s.io/kubernetes` internals, and a vendor
tree just to experiment with the scheduling math. This port strips all of
that away and keeps just the parts that actually decide *where a pod goes*:
the resource model, the fragmentation math, and the scoring policies — so
they can run anywhere Python runs, be unit-tested trivially, or be embedded
in a notebook.

## Files

| File | Ported from | Contents |
|---|---|---|
| `k8s_sim/resource.py` | `pkg/type/resource.go` | `PodResource` / `NodeResource`, fit-checking, `Sub`/`Add` allocation |
| `k8s_sim/fragmentation.py` | `pkg/utils/frag.go` | The Q1–Q4/XL/XR/NoAccess fragmentation classification and amount calculation used by FGD |
| `k8s_sim/policies.py` | `pkg/simulator/plugin/*.go` + literature | The original's 6 scoring plugins (Random, Best-Fit, Dot-Product/Tetris, GPU-Packing, GPU-Clustering, FGD) plus 5 new baselines (First-Fit, Worst-Fit, Round-Robin, DRF, Least-Requested) — see "Benchmark algorithms" below |
| `k8s_sim/cluster.py` | `pkg/simulator/simulator.go` (partial) | The filter → score → bind scheduling loop, with a per-policy persistent scratchpad for stateful policies like Round-Robin |
| `k8s_sim/trace.py` | `data/pod_csv_to_yaml.py`, `pkg/utils/frag.go`'s `GetTypicalPods` | Loader for the real production CSV traces in `data/csv/`, plus typical-pod-distribution derivation |
| `k8s_sim/experiment.py` | `scripts/analysis.py`, `experiments/analysis/*.py` | Benchmark-suite driver: sweeps policies x workloads x scale x seed, writes tidy CSVs |
| `k8s_sim/plotting.py` | `experiments/plot/*.py` | Matplotlib charts (frag ratio, frag breakdown, alloc-by-category, frag-vs-scale) from those CSVs |
| `data/csv/*.csv` | `data/csv/*.csv` (verbatim) | The original repo's real trace data: 1523 nodes / 8152 pods from a production GPU cluster, plus 20+ resampled workload-mix variants |
| `demo.py` | `example/` + `experiments/` | Compare all registered policies on a synthetic GPU-sharing workload |
| `trace_demo.py` | `experiments/` | Compare all registered policies on the **real** production trace data |
| `experiments/` | `experiments/` | Results storage, deep analysis, and plots — see `experiments/README.md` |
| `k8s_sim/simulation.py`, `k8s_sim/metrics.py` | — (new) | Time-driven simulation + 7 performance metrics — see `docs/METRICS.md` |
| `simulation_demo.py` | — (new) | Compare policies on utilization/throughput/waiting/fairness/starvation/latency/interference |
| `k8s_sim/topology.py`, `k8s_sim/htafm.py` | — (new, research prototype) | H-TAFM: topology-aware, multi-resource fragmentation metric + scheduler — see `docs/HTAFM.md` |
| `htafm_demo.py` | — (new) | Compare H-TAFM's 3 variants against FGD/Best-Fit/Random |
| `tests/` | — (new) | Conformance + integration test suite; see "CI / adding a new policy" below |
| `.github/workflows/ci.yml` | — (new) | GitHub Actions pipeline that runs the test suite on every push/PR |

## Quickstart

```bash
python3 demo.py
```

```python
from k8s_sim import Cluster, NodeResource, PodResource

cluster = Cluster([
    NodeResource(name="node-0", milli_cpu_left=32000, milli_cpu_capacity=32000,
                 milli_gpu_left_list=[1000, 1000, 1000, 1000], gpu_type="V100"),
])
pod = PodResource(milli_cpu=2000, milli_gpu=500, gpu_number=1, gpu_type="V100")
node_name = cluster.schedule_pod(pod, policy="fgd", typical_pods=[...])
```

### Using the real production trace data

```bash
python3 trace_demo.py openb_pod_list_gpushare100.csv 400
```

```python
from k8s_sim import Cluster
from k8s_sim.trace import load_nodes_csv, load_pods_csv, build_typical_pods

nodes = load_nodes_csv()                                       # all 1523 nodes
trace_pods = load_pods_csv(limit=1000, sample=True, seed=0)     # random 1000-pod subset
typical = build_typical_pods(trace_pods)                        # weighted typical-pod distribution

cluster = Cluster(nodes)
result = cluster.schedule_pods([tp.pod for tp in trace_pods], "fgd", typical_pods=typical)
print(len(result.scheduled), "scheduled,", len(result.unscheduled), "unscheduled")
```

`data/csv/` has 20+ pre-sampled trace variants (`openb_pod_list_cpu*.csv`,
`*_gpushare*.csv`, `*_multigpu*.csv`, `*_gpuspec*.csv`) emphasizing different
workload mixes — see `data/ORIGINAL_DATA_README.md` (copied verbatim from the
original repo's `data/README.md`) for the column definitions and what each
variant emphasizes. Use `load_pods_csv(path=..., sample=True)` to point at
any of them; `sample=True` draws a uniform random subset across the whole
file rather than a chronological prefix, since GPU-requesting pods are
concentrated later in some of the trace files.

## Benchmark algorithms

`k8s_sim.policies.POLICIES` currently registers 11 scoring policies:

| Policy | Family | Reference |
|---|---|---|
| `random` | spreading | — (original repo's naive baseline) |
| `first-fit` | packing | Johnson, "Fast Algorithms for Bin Packing," *J. Comput. Syst. Sci.* 8(3), 1974 |
| `worst-fit` | spreading | Coffman, Garey, Johnson, "Approximation Algorithms for Bin Packing: A Survey," 1996 |
| `round-robin` | spreading | classic scheduling baseline; used e.g. in Grandl et al., "Tetris," SIGCOMM 2014 |
| `best-fit` | packing | original repo (`pkg/simulator/plugin/best_fit_score.go`) |
| `dot-product` | packing | original repo, Tetris-style alignment scoring (`dot_product_score.go`) |
| `drf` | spreading | Ghodsi, Zaharia, Hindman, Konwinski, Shenker, Stoica, "Dominant Resource Fairness," NSDI 2011 |
| `least-requested` | spreading | Kubernetes `NodeResourcesFit` LeastAllocated strategy (kube-scheduler docs) |
| `gpu-packing` | packing | original repo, consolidates onto already-shared GPUs (`gpu_packing_score.go`) |
| `gpu-clustering` | packing | original repo, affinity-tag co-location (`gpu_clustering_score.go`) |
| `fgd` | packing | Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent," USENIX ATC 2023 |

The 5 new baselines (first-fit, worst-fit, round-robin, drf,
least-requested) exist so FGD and the original's other packing-aware
policies have a wider, better-known set of reference points to benchmark
against — classic bin-packing heuristics (First-Fit / Worst-Fit), a
fairness-theoretic baseline (DRF), a production-scheduler baseline
(Kubernetes' own default strategy), and a structural baseline (Round-Robin).
Across every workload category in the benchmark suite, the packing-family
policies consistently produce lower GPU fragmentation than the
spreading-family ones, and `fgd` comes out lowest of all — see
`experiments/plots/expected_results/` for example charts.

Each policy's docstring in `k8s_sim/policies.py` carries its citation.
`round-robin` needed one addition to the architecture: it's the first
policy that needs state to *persist across* pod placements (which node got
the last pod), so `Cluster` now keeps a small per-policy scratchpad
(`Cluster._policy_state`) that survives across `schedule_pod` calls within
the same `Cluster` instance.

## Experiments: results, deep analysis, and plots

See **`experiments/README.md`** for the full guide. In short:

```bash
pip install -r requirements-analysis.txt
python3 experiments/run_benchmark.py --preset fast
```

sweeps all 11 policies across 5 curated workload categories, writes a tidy
CSV to `experiments/results/`, and generates 3 plots to `experiments/plots/`
(fragmentation ratio by policy, the Q1-Q4/XL/XR/NoAccess breakdown, and
allocation ratio by workload category). Add `--preset scale` to also sweep
pod count and get a fragmentation-vs-load line chart. This is the Python
equivalent of the original's `experiments/run_scripts` → `scripts/analysis.py`
→ `experiments/analysis/merge_*.py` → `experiments/plot/plot_*.py` pipeline,
collapsed into one script since there's no Go binary or log-scraping step.

## Performance metrics system: utilization, throughput, waiting time, fairness, starvation, latency, interference

`k8s_sim/simulation.py` + `k8s_sim/metrics.py` add a **time-driven**
simulation mode (discrete-event: arrivals, a FIFO pending queue, departures
that release resources) on top of the static one-shot batch mode everything
else uses — this is what makes cluster utilization *over time*, job
throughput, waiting time, fairness, starvation, scheduling latency, and
interference intensity computable at all. **See `docs/METRICS.md`** for the
full writeup, including a correctness fix this required (tracking exactly
which GPU devices a pod used so departures release the right capacity) and
honest caveats on the two metrics that needed something invented (fairness
needs synthetic tenants — the trace has no real user IDs; interference
intensity is an uncalibrated co-tenancy proxy, not a validated slowdown
model). Run `python3 simulation_demo.py 150 1` for a demo with visible
contention.

## H-TAFM: a topology-aware, multi-resource extension (research prototype)

`k8s_sim/topology.py` + `k8s_sim/htafm.py` implement H-TAFM (Hypergraph-based
Topology-Aware Fragmentation Metric) — a proposed extension of FGD to
CPU+Memory+GPU and a synthetic NUMA/Socket/Server/Rack hierarchy, scored via
a weighted hypergraph and a gradient-descent scheduler analogous to FGD's.
It operates at NUMA-vertex granularity (finer than the rest of this repo's
whole-node placement), so it's its own module rather than another
`policies.POLICIES` entry. **See `docs/HTAFM.md`** for the full
methodology-to-code mapping, documented simplifications, and two concrete
findings from testing it against the real trace data: TAFI-Entropy violates
its own claimed monotonicity property (TAFI-Cut and TAFI-Hier don't), and
NUMA-granularity placement can reject large multi-GPU jobs that
node-granularity placement accepts. Run `python3 htafm_demo.py 300` to
compare all 3 variants against FGD/Best-Fit/Random on real trace data.

## What's ported faithfully

- **Resource units**: milli-cpu and per-device milli-gpu (0–1000), identical
  to the original's GPU-sharing model.
- **Fragmentation classification** (`get_node_pod_frag`): the exact
  Q1/Q2/Q3/Q4/XL/XR/NoAccess bucket logic from `frag.go`, including the Q3
  special-case where only the *undersized-device* portion of "satisfied"
  idle GPU memory counts as waste.
- **All 6 scoring formulas**: Best-Fit's weighted normalized-leftover score,
  GPU-Packing's 3-tier consolidation scoring, GPU-Clustering's affinity
  tiers, and FGD's sigmoid-of-fragmentation-delta — all translated line by
  line from the corresponding `*_score.go` file.
- **GPU packing within a node** (`NodeResource.sub`): ascending
  least-sufficient-device-first, matching `simontype.NodeResource.Sub`.

## Simplifications relative to the original

These were dropped because they're either Kubernetes-specific plumbing with
no bearing on the scheduling *decision*, or configuration knobs whose
defaults were inlined:

- **No live k8s API / scheduler framework.** The original registers these
  as `framework.ScorePlugin`s and runs them inside a fake API server driven
  by real `client-go` informers; here `Cluster.schedule_pod` just directly
  filters + scores + mutates node state.
- **No node affinity / taints / tolerations / topology spread** (`pkg/algo/*.go`).
  Only GPU-type accessibility and CPU/GPU capacity are checked as filters.
- **Dot-Product** always uses the "merge GPU dimensions" + `NormByPod` tanh
  normalization; the original supports 3 dimension-extension methods
  (`MergeGpuDim`, `SeparateGpuDim*`, `ExtGpuDim`) and 3 normalization modes
  as CLI config.
- **GPU-Clustering** keys "affinity" off `pod.gpu_type` (or an explicit
  `affinity_key`) rather than a GPU-vendor/model annotation string parsed
  off a live pod object — same mechanic, simpler input.
- **`GetTypicalPods`** (deriving a typical-pod distribution + popularity
  threshold from a real trace) is replaced by
  `fragmentation.build_typical_pods_uniform` / a hand-specified list in
  `demo.py`. Wire in real trace data by constructing `TargetPod` objects
  yourself from parsed pod specs.
- **No descheduling / eviction / multi-scheduler-cycle replay**
  (`pkg/simulator/deschedule*.go`, `analysis.go`, `export.go`); this port
  is single-pass bin-packing, which is what you need to compare scoring
  policies.
- **No skyline / Bellman-equation fragmentation variant** — the original has
  a more expensive DP-based fragmentation estimator in `frag.go` that's
  commented out / deprecated upstream in favor of `NodeGpuShareFragAmount`,
  which is what's ported here.

## CI / adding a new policy

```bash
pip install -r requirements-dev.txt
make ci        # lint + full test suite + both demos, same as GitHub Actions runs
```

Every policy registered in `k8s_sim.policies.POLICIES` (currently 11, see
"Benchmark algorithms" above) is automatically checked by
`tests/test_policy_contract.py` and `tests/test_cluster.py` against a
compatibility contract (score range, determinism, no side effects, resource
conservation, correct filtering) — no per-policy test code required.
`tests/test_experiment.py` additionally smoke-tests the benchmark/plotting
pipeline itself. `.github/workflows/ci.yml` runs all of this on every
push/PR across Python 3.9–3.12, plus a benchmark+plotting smoke step. See
**CONTRIBUTING.md** for the exact contract, how to add a new policy, and how
to read a failure (assertion messages name the policy, the contract clause,
and the offending node/pod).

## Extending

- To score with a **real trace**: parse your pod specs into `PodResource`
  objects, build a frequency-weighted `TargetPod` list (port
  `GetTypicalPods` if you want the exact popularity-threshold chopping
  logic), and call `cluster.schedule_pods(pods, policy, typical_pods=...)`.
- To add a **new policy**: write a function
  `score(node, pod, ctx) -> float` and register it in `policies.POLICIES`.
- To model **pod completion / eviction**: call `NodeResource.add(pod)` to
  release resources back onto a node (mirrors `Add` in the original).
