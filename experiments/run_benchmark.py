"""
Run the full benchmark suite (all registered policies x a curated set of
workload categories) and generate the standard analysis plots.

This is the Python-only equivalent of the original repo's three-stage
pipeline (experiments/run_scripts -> scripts/analysis.py ->
experiments/analysis/merge_*.py -> experiments/plot/plot_*.py): one script,
results in experiments/results/, plots in experiments/plots/.

Usage:
    python3 experiments/run_benchmark.py --preset fast     # ~1-2 min, small node/pod sample
    python3 experiments/run_benchmark.py --preset full      # full 1523-node cluster, slower
    python3 experiments/run_benchmark.py --preset scale     # sweep pod count for a frag-vs-scale plot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from k8s_sim.experiment import run_benchmark_suite, WORKLOAD_CATEGORIES  # noqa: E402


PRESETS = {
    "fast": dict(
        pod_counts=(150,),
        seeds=(0, 1),
        node_sample_size=150,
    ),
    "full": dict(
        pod_counts=(300,),
        seeds=(0, 1, 2),
        node_sample_size=None,  # all 1523 nodes
    ),
    "scale": dict(
        pod_counts=(50, 100, 200, 400),
        seeds=(0,),
        node_sample_size=150,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS.keys(), default="fast")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation (e.g. no pandas/matplotlib)")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    print(f"Running benchmark suite with preset={args.preset!r}: {preset}\n")

    results = run_benchmark_suite(
        workloads=list(WORKLOAD_CATEGORIES.values()),
        **preset,
    )

    print(f"\n{len(results)} runs complete.")

    if not args.no_plots:
        try:
            from k8s_sim import plotting
        except ImportError:
            print("\nmatplotlib/pandas not installed -- skipping plots. "
                  "Install with: pip install -r requirements-analysis.txt")
            return
        df = plotting.load_results()
        paths = plotting.generate_all_plots(df)
        print("\nWrote plots:")
        for p in paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
