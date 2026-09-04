# Experiments: results storage, deep analysis, and plots

This mirrors the original repo's three-stage `experiments/` pipeline
(`run_scripts/` → `analysis/` → `plot/`), but as one Python entry point
instead of Go binary runs + shell-scripted log merging.

```
experiments/
├── run_benchmark.py       # entry point: sweep policies x workloads, write CSV, generate plots
├── results/                # CSV output lands here (gitignored except .gitkeep)
│   └── latest.csv           # always overwritten with the most recent run, for quick plotting
└── plots/                  # PNG output lands here
    └── expected_results/    # sample plots from a `--preset fast` run, checked in for reference
```

`k8s_sim/experiment.py` (`run_experiment`, `run_benchmark_suite`) does the
running and CSV-writing; `k8s_sim/plotting.py` does the charting. Both are
importable directly if you want a custom sweep or a notebook workflow
instead of the CLI.

## Getting results from CI

`.github/workflows/ci.yml` has a `benchmark` job (runs on push to `main` and
on manual dispatch from the Actions tab) that runs the `fast` preset and
uploads `experiments/results/*.csv` + `experiments/plots/*.png` as a
downloadable Actions artifact named `benchmark-results` -- go to your repo's
**Actions tab → the workflow run → Artifacts** at the bottom of the run
summary. The `test` job's own benchmark step is a separate, much smaller
smoke check that writes to `/tmp` and is thrown away; it only proves the
pipeline doesn't crash, it isn't meant to produce anything to look at.

## Quickstart (running it yourself)

```bash
pip install -r requirements-analysis.txt   # pandas + matplotlib, on top of requirements-dev.txt
python3 experiments/run_benchmark.py --preset fast
```

Three presets:

| Preset | Nodes | Pod counts | Seeds | Runtime | Use for |
|---|---|---|---|---|---|
| `fast`  | random 150-node subsample | 150 | 2 | ~1-2 min | quick iteration, sanity-checking a new policy |
| `full`  | all 1523 nodes | 300 | 3 | tens of minutes (FGD dominates the cost) | paper-scale numbers |
| `scale` | random 150-node subsample | 50/100/200/400 | 1 | a few min | fragmentation-vs-load line plot |

Every preset sweeps the 5 curated workload categories in
`k8s_sim.experiment.WORKLOAD_CATEGORIES` (default / cpu-heavy /
gpu-share-heavy / multi-gpu-heavy / gpu-type-constrained — one representative
trace file per category, same categories the original's alloc-bar plots use)
across every policy registered in `k8s_sim.policies.POLICIES`.

## What gets written

`results/benchmark_<timestamp>.csv` (and a `results/latest.csv` copy), one
row per `(policy, workload, n_pods, seed)` combination, with:

- `scheduled` / `unscheduled` counts
- `alloc_ratio_cpu`, `alloc_ratio_gpu` — cluster-wide utilization after scheduling
- `frag_ratio` — the headline FGD metric: share of idle GPU memory that's fragmented
- `frag_pct_*` — the full Q1/Q2/Q3/Q4/XL/XR/NoAccess breakdown, as percentages

`plots/`:

- `frag_ratio_by_policy.png` — mean fragmentation ratio per policy, colored by family (FGD highlighted, packing-style in blue, spreading-style in grey). This is the headline chart, analogous to `experiments/plot/plot_openb_frag_ratio.py` in the original.
- `frag_breakdown_stacked.png` — the same, broken into the Q1/Q2/Q4/XL/XR/NoAccess stack, analogous to `plot_openb_frag_amount.py`.
- `alloc_ratio_by_category.png` — grouped bars, GPU allocation ratio per policy per workload category, analogous to the `plot_openb_*_alloc_bar.py` scripts.
- `frag_ratio_vs_scale.png` — only generated when the sweep covers multiple `n_pods` values (e.g. the `scale` preset): fragmentation ratio as load grows, one line per policy.

## Running a custom sweep

```python
from k8s_sim.experiment import run_benchmark_suite
from k8s_sim import plotting

results = run_benchmark_suite(
    policies=["fgd", "best-fit", "my-new-policy"],
    workloads=["openb_pod_list_gpuspec33.csv"],
    pod_counts=(500,),
    seeds=(0, 1, 2, 3, 4),
    node_sample_size=300,
)
df = plotting.load_results()  # reads results/latest.csv, written automatically
plotting.plot_frag_ratio_by_policy(df, workload="openb_pod_list_gpuspec33.csv")
```
