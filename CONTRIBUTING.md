# Contributing a new scheduling policy / algo

This project is set up so adding a new scoring policy is a two-step process,
and CI tells you immediately whether it's compatible with the rest of the
framework.

## 1. Write the policy

Add a function to `k8s_sim/policies.py` with this exact shape:

```python
def my_policy_score(node: NodeResource, pod: PodResource, ctx: dict) -> float:
    """One-line description of the strategy."""
    ...
    return score  # must be in [MIN_NODE_SCORE, MAX_NODE_SCORE] == [0, 100]
```

Then register it:

```python
POLICIES["my-policy"] = my_policy_score
```

If your policy needs a one-time per-pod setup step before nodes are scored
(like `random`'s node pre-pick), add a matching entry to `PREPARE_HOOKS`
instead of doing it inside the score function itself — that keeps scoring
pure and testable (see contract clause C3 below).

## 2. Run the compatibility checks locally

```bash
pip install -r requirements-dev.txt
pytest -v
```

Two files matter most for a new policy:

- **`tests/test_policy_contract.py`** — runs automatically against every
  entry in `POLICIES`, no per-policy test code needed. It checks:
  - **C1** — no exceptions on any feasible `(node, pod)` pair
  - **C2** — score is within `[0, 100]`
  - **C3** — scoring is deterministic given the same `(node, pod, ctx)`
  - **C4** — scoring does not mutate `node` or `pod`
  - **C5** — the policy is registered under a unique key and callable
- **`tests/test_cluster.py`** — runs your policy through the full
  `Cluster.schedule_pods` loop and checks resource conservation (no node
  ends up over-allocated or negative), consistent bookkeeping, and correct
  rejection of infeasible / GPU-type-mismatched pods.

If a check fails, the assertion message tells you exactly which contract
clause broke and on which `node`/`pod` combination, e.g.:

```
AssertionError: [my-policy] score 150.0 out of range [0, 100] for
node='node-test' pod='<CPU: 0.50, GPU: 1 x {250}m (V100)>'.
Check your normalization/scaling step.
```

```
AssertionError: [my-policy] mutated node.milli_cpu_left as a side effect of scoring
```

That's the "where it breaks" signal — fix the named issue, re-run `pytest`,
and once green your policy is confirmed compatible and integrated.

## 3. CI

`.github/workflows/ci.yml` runs on every push/PR:

1. **lint** — flake8 syntax/undefined-name check (hard fail) + style report
2. **test** — the full suite (`pytest --cov=k8s_sim`) on Python 3.9–3.12,
   plus a smoke run of `demo.py` and `trace_demo.py`
3. **new-policy-check** — re-runs just the conformance + integration suites
   as a separately-named job, so a red ❌ here unambiguously means "a
   registered policy failed the compatibility contract," not some unrelated
   test.

A PR that adds a new policy is safe to merge once all three jobs are green.

## Optional: also test against real trace data

`tests/test_trace.py` exercises the CSV loader against the real production
traces in `data/csv/` (skipped automatically if that directory isn't
present in your checkout). If you want your new policy validated against
real workload shapes rather than just the synthetic fixtures, add a
parametrized case there or extend `trace_demo.py`.
