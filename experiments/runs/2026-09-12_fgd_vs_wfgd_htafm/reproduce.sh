#!/usr/bin/env bash
# Reproduces every result in results/ from the ALREADY-ARCHIVED gputrace_raw/ +
# gputrace_export/ data in this directory (does not re-invoke gputrace at all,
# so this reproduces byte-identically regardless of any later changes to
# gputrace's scenario-generation logic). See manifest.json for the exact
# commands that originally produced gputrace_raw/gputrace_export, if you want
# to regenerate those from gputrace directly instead.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="../../.."  # repo root, from experiments/runs/<this run>/

PY=python3
NODE_POLICIES="random,first_fit,best_fit,worst_fit,round_robin,gpu_packing,least_requested,fgd,w_fgd,w_fgd_balanced"

echo "== static matrix (all node policies + H-TAFM) =="
$PY "$ROOT/experiments/full_scenario_matrix.py" \
    --traces-dir gputrace_export --out results/static_matrix_results.csv

echo "== time-driven sweep (node policies only) =="
$PY -m k8s_sim.experiment --traces-dir gputrace_export \
    --policies "$NODE_POLICIES" --seeds 1,2,3 --mode time-driven \
    --out results/timedriven_results.csv

echo "== broader metrics (fairness, tail latency, SLA, GPU-hours) =="
$PY "$ROOT/experiments/broader_metrics.py" --traces-dir gputrace_export \
    --policies "$NODE_POLICIES" --seeds 1,2,3 \
    --out results/broader_metrics_timedriven.csv \
    --out-static results/broader_metrics_static.csv

echo "done -- see results/*.csv"
