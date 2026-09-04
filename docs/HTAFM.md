# H-TAFM: implementation notes

This documents how the Hypergraph-based Topology-Aware Fragmentation Metric
methodology was decoded into code (`k8s_sim/topology.py`, `k8s_sim/htafm.py`),
what's implemented faithfully, what had to be synthesized or resolved due to
gaps/ambiguity in the source methodology, and two substantive findings from
testing it against real trace data.

## Why it's a separate module, not another `policies.POLICIES` entry

Everything else in `k8s_sim` (`resource.py`, `cluster.py`, `policies.py`)
places a pod on a whole **node** — one physical machine is one bin. H-TAFM's
system model (Sec. 1.1) places a VM on a **NUMA-node vertex**, one level
*below* the physical server, and scores the placement by its effect on a
weighted hypergraph of Socket/Server/Rack membership. That's a different
placement granularity, not just a different scoring function for the same
granularity — so it gets its own `Topology`/`HTAFMScheduler` rather than a
`score(node, pod, ctx)` function plugged into the existing registry.

## Mapping methodology → code

| Methodology | Code |
|---|---|
| Sec. 1.1 system model: NUMA/Socket/Server/Rack levels, capacity/demand vectors | `topology.py`: `TopoVertex`, `TopoDemand` |
| Sec. 2 hypergraph construction: vertices + weighted hyperedges | `topology.py`: `HyperEdge`, `Topology.from_nodes` |
| Sec. 3 Variant 1, TAFI-Cut | `htafm.py`: `phi_cut` |
| Sec. 3 Variant 2, TAFI-Entropy | `htafm.py`: `phi_entropy` |
| Sec. 3 Variant 3, TAFI-Hier | `htafm.py`: `phi_hier` |
| Sec. 4 gradient, ΔTAFI | `htafm.py`: `delta_tafi` (incremental, per Sec. 5's optimization note) |
| Sec. 5 Algorithm (H-TAFM Scheduler) | `htafm.py`: `HTAFMScheduler.schedule_one` / `.schedule` |
| Sec. 6 simulation environment, real trace | `topology.py`: `load_topology_from_csv`, reading the same `data/csv/openb_node_list_*.csv` used everywhere else in this repo |
| Sec. 7 evaluation metrics | `HTAFMScheduler.unallocated_gpus`; `htafm_demo.py` reports acceptance rate |
| Sec. 8 theoretical properties | `tests/test_htafm.py`'s monotonicity tests — see **Finding 1** below |

## What's synthesized (not in the real data)

The Alibaba trace (`data/csv/openb_node_list_*.csv`) gives per-node
`cpu_milli`, `memory_mib`, `gpu`, `model` — there's no NUMA/socket/rack
topology in it (real production traces of this kind generally don't publish
sub-node hardware topology). So:

- **NUMA/socket split**: each node's CPU and memory capacity is divided
  evenly across `numa_per_socket x sockets_per_node` vertices (defaults:
  1 x 2 = 2 vertices/node); its GPUs are dealt round-robin across them so
  vertex GPU counts stay integral. This is a documented synthetic layer —
  the underlying CPU/memory/GPU **capacities** are real trace data, only the
  sub-node partitioning is synthetic.
- **Rack assignment**: nodes are grouped into racks of `nodes_per_rack`
  (default 40) in file order — an arbitrary but stable assignment, since the
  trace has no real rack labels.
- **Memory demand**: `k8s_sim.resource.PodResource` doesn't track memory
  (the rest of the framework doesn't need it), so `htafm_demo.py` re-reads
  `memory_mib` directly from the pod CSV rather than threading a new field
  through the core resource model.

## Resolved ambiguities

1. **The NUMA-level hyperedge is a degenerate singleton.** Since vertices
   *are* NUMA nodes (Sec. 1.1's Level 0), a "hyperedge connecting vertices
   within the same NUMA node" (Sec. 2) would always contain exactly one
   vertex. A 1-vertex hyperedge can never be "cut" (φ_cut), has entropy 0
   (φ_entropy), and its φ_hier is degenerate 0-or-1 with no aggregation
   signal. It's correctly omitted rather than approximated — the Socket,
   Server, and Rack hyperedges are where the real signal is.

2. **TAFI-Hier's formula double-counts across the outer sum.** Sec. 3
   defines `φ_hier(e, Π) = Σ_level λ_level · frag_level(e, Π)` — summing
   over *all four levels inside a single hyperedge's φ* — while Sec. 3 also
   defines `TAFI = Σ_{e∈E} w(e) · φ(e, Π)`, which already iterates over
   every hyperedge at every level. Taken literally, a Rack-level hyperedge's
   φ would separately compute NUMA/Socket/Server/Rack `frag_level` terms
   that have nothing to do with a rack's own membership. We resolve this by
   having `φ_hier(e)` use only `e`'s own level's threshold/λ (i.e. a Server
   hyperedge's φ uses `λ_server`, not a sum over all four λ's) — the outer
   `Σ_{e∈E}` then performs the cross-level aggregation the inner formula
   also describes, without double-counting. See `htafm.py`'s module
   docstring for the same note in code.

3. **"Activate a new physical server" (Sec. 5, step 9) isn't modeled.**
   This is an offline, capacity-constrained simulation over a fixed,
   already-loaded cluster — consistent with how the rest of `k8s_sim` and
   the original Go simulator work (neither adds hardware mid-run). When no
   vertex fits, the demand is reported unscheduled instead, matching
   `k8s_sim.cluster.Cluster`'s convention.

4. **Affinity/anti-affinity hyperedges (Sec. 2) aren't implemented.** The
   methodology names them as a hyperedge type but gives no concrete formula
   for how they factor into any of the three TAFI variants — there's
   nothing to decode here without inventing a formula the source document
   doesn't specify. Flagged as a gap rather than guessed at.

## Finding 1: TAFI-Entropy is not actually monotonic

Sec. 8 claims: *"Adding a VM to any vertex cannot decrease the TAFI value.
The fragmentation index is non-decreasing."* This holds for TAFI-Cut and
TAFI-Hier (verified empirically — `tests/test_htafm.py::test_cut_variant_is_monotonic`
and `::test_hier_variant_is_monotonic`, 0 violations over 20-30 sequential
placements each), but **not** for TAFI-Entropy.

Why: Shannon entropy of the free-resource distribution across a hyperedge's
vertices is maximized when that distribution is perfectly uniform. Removing
capacity from one vertex moves the distribution *toward* uniform if that
vertex currently holds an above-average share (entropy goes up) but *away*
from uniform if it holds a below-average share (entropy goes down). Both
cases occur in practice — running 30 sequential placements against real
trace data (`docs` verification, also reproduced in
`tests/test_htafm.py::test_entropy_variant_violates_claimed_monotonicity`)
produced a monotonicity violation on **every single placement**, not an edge
case:

```
cut     : monotonicity violations over 30 placements = 0
hier    : monotonicity violations over 30 placements = 0
entropy : monotonicity violations over 30 placements = 30
```

This means Sec. 8's monotonicity and (by extension) schedulability-
correlation proofs need a variant-specific caveat, or a different argument
for TAFI-Entropy specifically — as stated, they don't hold for it. This
doesn't make TAFI-Entropy useless as a *scoring* heuristic (the gradient
descent still greedily minimizes it at each step, which is well-defined
regardless), but the paper shouldn't claim the general monotonicity/
boundedness/schedulability-correlation properties apply uniformly across
all three variants without this caveat.

## Finding 2: NUMA-granularity placement can reject jobs node-granularity placement accepts

Running `htafm_demo.py` against 200 real trace pods (`nodes_per_rack=40`,
default 2 NUMA vertices/node) schedules 199/200, while the equivalent
node-granularity baselines (FGD, Best-Fit, Random) schedule 200/200 on the
same demands:

```
h-tafm-cut       199/200 scheduled  (99.5%)
fgd              200/200 scheduled  (100.0%)
```

The one rejected demand needs more GPUs than fit on any single NUMA vertex
after the node's GPUs were split across `numa_per_node` vertices (e.g. a
4-GPU job on an 8-GPU node that got split into 2 vertices of 4 GPUs each is
fine; the same job on a node split into 4 vertices of 2 GPUs each is not,
even though the whole node has plenty of capacity). This is a real, expected
consequence of enforcing single-vertex placement (H-TAFM never splits one
VM across vertices) at finer granularity — not a bug. It means
`numa_per_socket`/`sockets_per_node` need to be chosen with your workload's
largest expected multi-GPU job size in mind; splitting too finely
systematically excludes large jobs regardless of how good the fragmentation
metric is.

## Running it

```python
from k8s_sim.topology import load_topology_from_csv, TopoDemand
from k8s_sim.htafm import HTAFMScheduler, HTAFMConfig

topo = load_topology_from_csv(numa_per_socket=1, sockets_per_node=2, nodes_per_rack=40)
cfg = HTAFMConfig(variant="entropy")   # or "cut" (needs pending_demands) or "hier"
sched = HTAFMScheduler(topo, cfg)

demand = TopoDemand(milli_cpu=2000, memory_mib=4096, milli_gpu=500, gpu_number=1, gpu_type="V100")
vertex_id = sched.schedule_one(demand)
```

or compare all 3 variants against the existing baselines:

```bash
python3 htafm_demo.py 300
```

## What's left as future work

- Affinity/anti-affinity hyperedges (no formula given in the source doc, see above)
- The carbon-intensity multi-objective extension mentioned in Sec. 6
- TOPSIS-FGD as an explicit comparison baseline (mentioned as a target
  comparison in the source doc's Sec. 3 "Baseline Policies," but no formula
  for TOPSIS-FGD itself was included in the provided methodology to decode)
- Wiring H-TAFM into `k8s_sim/experiment.py`/`plotting.py`'s CSV+plot
  pipeline for head-to-head charts against the node-granularity policies
  (currently `htafm_demo.py` only prints a table)
