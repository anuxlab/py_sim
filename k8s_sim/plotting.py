"""
Plotting for benchmark results, standing in for experiments/plot/*.py in the
original repo (plot_openb_frag_ratio.py, plot_openb_frag_amount.py,
plot_openb_*_alloc_bar.py). Reads the tidy CSV that
k8s_sim.experiment.run_benchmark_suite writes and produces PNGs under
experiments/plots/.

Requires matplotlib + pandas (see requirements-analysis.txt); not imported
by the core k8s_sim package so the base package stays dependency-free.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", "plots")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "experiments", "results")

# Rough policy families, used consistently for plot coloring/legends so
# packing-style vs. spreading-style policies are visually distinguishable.
PACKING_POLICIES = {"first-fit", "best-fit", "dot-product", "gpu-packing", "gpu-clustering", "fgd"}
SPREADING_POLICIES = {"random", "worst-fit", "round-robin", "drf", "least-requested"}


def load_results(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(RESULTS_DIR, "latest.csv")
    return pd.read_csv(path)


def _bar_color(policy: str) -> str:
    if policy == "fgd":
        return "#d62728"  # highlight the paper's headline algorithm
    if policy in PACKING_POLICIES:
        return "#1f77b4"
    return "#7f7f7f"


def plot_frag_ratio_by_policy(df: pd.DataFrame, workload: Optional[str] = None,
                               out_name: str = "frag_ratio_by_policy.png") -> str:
    """Mean fragmentation ratio per policy (lower is better), analogous to
    experiments/plot/plot_openb_frag_ratio.py's headline chart."""
    sub = df if workload is None else df[df.workload == workload]
    agg = sub.groupby("policy")["frag_ratio"].mean().sort_values() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [_bar_color(p) for p in agg.index]
    ax.bar(agg.index, agg.values, color=colors)
    ax.set_ylabel("Fragmentation ratio (%)")
    ax.set_title("Fragmentation ratio by policy" + (f" ({workload})" if workload else " (all workloads)"))
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_frag_breakdown_stacked(df: pd.DataFrame, workload: Optional[str] = None,
                                 out_name: str = "frag_breakdown_stacked.png") -> str:
    """Stacked bar of the Q1/Q2/Q4/XL/XR/NoAccess breakdown per policy
    (Q3-satisfied is the "good" bucket and is intentionally excluded from
    the stack, matching frag_ratio's definition), analogous to
    plot_openb_frag_amount.py."""
    sub = df if workload is None else df[df.workload == workload]
    frag_cols = [c for c in sub.columns if c.startswith("frag_pct_") and c != "frag_pct_q3_satisfied"]
    agg = sub.groupby("policy")[frag_cols].mean()
    agg = agg.loc[agg.sum(axis=1).sort_values().index]  # order by total fragmentation

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = None
    for col in frag_cols:
        label = col.replace("frag_pct_", "")
        ax.bar(agg.index, agg[col], bottom=bottom, label=label)
        bottom = agg[col] if bottom is None else bottom + agg[col]
    ax.set_ylabel("Share of idle GPU memory (%)")
    ax.set_title("Fragmentation breakdown by policy" + (f" ({workload})" if workload else ""))
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_alloc_ratio_by_category(df: pd.DataFrame, category_to_workload: dict,
                                  out_name: str = "alloc_ratio_by_category.png") -> str:
    """Grouped bar chart: GPU allocation ratio per policy, one group of bars
    per workload category, analogous to
    plot_openb_{gpushare,multigpu,nongpu,gpuspec}_alloc_bar.py."""
    categories = list(category_to_workload.keys())
    policies = sorted(df["policy"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / max(len(categories), 1)
    x = range(len(policies))
    for i, cat in enumerate(categories):
        wl = category_to_workload[cat]
        sub = df[df.workload == wl]
        means = [sub[sub.policy == p]["alloc_ratio_gpu"].mean() * 100 for p in policies]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, means, width=width, label=cat)

    ax.set_xticks([xi + width * (len(categories) - 1) / 2 for xi in x])
    ax.set_xticklabels(policies, rotation=40, ha="right")
    ax.set_ylabel("GPU allocation ratio (%)")
    ax.set_title("Allocation ratio by policy and workload category")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_frag_ratio_vs_scale(df: pd.DataFrame, workload: Optional[str] = None,
                              out_name: str = "frag_ratio_vs_scale.png") -> str:
    """Line plot of fragmentation ratio as pod count (workload scale) grows,
    one line per policy -- useful when run_benchmark_suite was swept over
    multiple pod_counts."""
    sub = df if workload is None else df[df.workload == workload]
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, g in sub.groupby("policy"):
        agg = g.groupby("n_pods")["frag_ratio"].mean().sort_index() * 100
        ax.plot(agg.index, agg.values, marker="o",
                label=policy, color=_bar_color(policy),
                linewidth=3 if policy == "fgd" else 1.5)
    ax.set_xlabel("Number of pods scheduled")
    ax.set_ylabel("Fragmentation ratio (%)")
    ax.set_title("Fragmentation ratio vs. workload scale" + (f" ({workload})" if workload else ""))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_plots(df: Optional[pd.DataFrame] = None,
                        category_to_workload: Optional[dict] = None) -> Sequence[str]:
    """Convenience entry point: generate the standard set of plots from a
    results dataframe (or experiments/results/latest.csv) and return their
    paths."""
    if df is None:
        df = load_results()
    from .experiment import WORKLOAD_CATEGORIES
    category_to_workload = category_to_workload or WORKLOAD_CATEGORIES

    paths = [
        plot_frag_ratio_by_policy(df),
        plot_frag_breakdown_stacked(df),
    ]
    available_cats = {k: v for k, v in category_to_workload.items() if v in set(df["workload"])}
    if available_cats:
        paths.append(plot_alloc_ratio_by_category(df, available_cats))
    if df["n_pods"].nunique() > 1:
        paths.append(plot_frag_ratio_vs_scale(df))
    return paths
