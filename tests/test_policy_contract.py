"""
Conformance tests for scheduling policies registered in k8s_sim.policies.POLICIES.

THIS FILE IS THE COMPATIBILITY GATE for new plugin/algo development. If you
add a new scoring function and register it in `policies.POLICIES`, these
tests run against it automatically (no per-policy test code needed) and
tell you exactly which contract clause and which (node, pod) combination it
broke on. See CONTRIBUTING.md for the full checklist and how to read a
failure.

The contract, mirroring what framework.ScorePlugin implicitly guarantees in
the original Go project:

  C1. `score(node, pod, ctx)` returns a plain number (int/float), never None,
      NaN, or an exception, for any node that `node.fits(pod)`.
  C2. The returned score is within [MIN_NODE_SCORE, MAX_NODE_SCORE] (0-100).
  C3. Scoring is a pure function of (node, pod, ctx): calling it twice with
      identical (unmutated) inputs gives the identical result.
  C4. Scoring must NOT mutate the `node` or `pod` objects it was given.
  C5. The policy must be registered under a unique string key and be usable
      end-to-end through `Cluster.schedule_pod` without raising, and without
      violating resource conservation (see test_cluster.py for the deeper
      integration checks).
"""

import copy
import math

import pytest

from k8s_sim import policies as policy_mod
from k8s_sim.resource import MIN_NODE_SCORE, MAX_NODE_SCORE

ALL_POLICIES = sorted(policy_mod.POLICIES.keys())


def _make_ctx(policy, node, pod, typical_pods):
    ctx = {"typical_pods": typical_pods}
    prepare = policy_mod.PREPARE_HOOKS.get(policy)
    if prepare:
        prepare([node], pod, ctx)
    return ctx


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_policy_is_registered_and_callable(policy):
    """C5 (partial): sanity check the registry itself."""
    fn = policy_mod.POLICIES[policy]
    assert callable(fn), f"policies.POLICIES['{policy}'] must map to a callable"


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_policy_scores_feasible_pairs_without_error(policy, cluster_nodes, sample_pods, typical_pods):
    """C1: no exceptions on any feasible (node, pod) pair."""
    score_fn = policy_mod.POLICIES[policy]
    checked_any = False
    for node in cluster_nodes:
        for pod in sample_pods:
            if not node.fits(pod):
                continue
            checked_any = True
            ctx = _make_ctx(policy, node, pod, typical_pods)
            try:
                score = score_fn(node, pod, ctx)
            except Exception as e:  # noqa: BLE001 - re-raise with context
                pytest.fail(
                    f"[{policy}] raised {type(e).__name__}({e}) scoring "
                    f"node={node.repr()!r} pod={pod.repr()!r}. "
                    f"This is where your policy breaks -- add a guard for "
                    f"this (node, pod) shape."
                )
            assert score is not None, f"[{policy}] returned None for node={node.name}, pod={pod.name}"
            assert not (isinstance(score, float) and math.isnan(score)), \
                f"[{policy}] returned NaN for node={node.name}, pod={pod.name}"
    assert checked_any, "test fixtures produced no feasible (node, pod) pairs -- fix the fixtures, not the policy"


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_policy_score_in_range(policy, cluster_nodes, sample_pods, typical_pods):
    """C2: score must land in [MIN_NODE_SCORE, MAX_NODE_SCORE]."""
    score_fn = policy_mod.POLICIES[policy]
    for node in cluster_nodes:
        for pod in sample_pods:
            if not node.fits(pod):
                continue
            ctx = _make_ctx(policy, node, pod, typical_pods)
            score = score_fn(node, pod, ctx)
            assert MIN_NODE_SCORE <= score <= MAX_NODE_SCORE, (
                f"[{policy}] score {score} out of range "
                f"[{MIN_NODE_SCORE}, {MAX_NODE_SCORE}] for node={node.name!r} "
                f"pod={pod.repr()!r}. Check your normalization/scaling step."
            )


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_policy_is_deterministic(policy, cluster_nodes, sample_pods, typical_pods):
    """C3: same inputs -> same output. (Randomized *placement* is fine --
    that's driven by ctx, e.g. random_prepare -- but scoring given a fixed
    ctx must be deterministic, or Cluster.schedule_pod's max() tie-breaking
    becomes nondeterministic in a way that breaks reproducibility.)"""
    score_fn = policy_mod.POLICIES[policy]
    for node in cluster_nodes:
        for pod in sample_pods:
            if not node.fits(pod):
                continue
            ctx = _make_ctx(policy, node, pod, typical_pods)
            s1 = score_fn(node, pod, ctx)
            s2 = score_fn(node, pod, ctx)
            assert s1 == s2, (
                f"[{policy}] nondeterministic: got {s1} then {s2} for the "
                f"same node={node.name!r}, pod={pod.repr()!r}, ctx. "
                f"Scoring functions must not depend on hidden global state "
                f"(e.g. unseeded randomness) outside of a documented "
                f"PREPARE_HOOKS step."
            )


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_policy_does_not_mutate_inputs(policy, cluster_nodes, sample_pods, typical_pods):
    """C4: scoring is read-only w.r.t. node/pod."""
    score_fn = policy_mod.POLICIES[policy]
    for node in cluster_nodes:
        for pod in sample_pods:
            if not node.fits(pod):
                continue
            node_before = copy.deepcopy(node)
            pod_before = copy.deepcopy(pod)
            ctx = _make_ctx(policy, node, pod, typical_pods)
            score_fn(node, pod, ctx)
            assert node.milli_cpu_left == node_before.milli_cpu_left, \
                f"[{policy}] mutated node.milli_cpu_left as a side effect of scoring"
            assert node.milli_gpu_left_list == node_before.milli_gpu_left_list, \
                f"[{policy}] mutated node.milli_gpu_left_list as a side effect of scoring " \
                f"(likely called node.sub()/node.milli_gpu_left_list[i] -= ... in place -- " \
                f"use node.copy() before mutating, as fgd_score does)"
            assert pod == pod_before, f"[{policy}] mutated the pod object as a side effect of scoring"
