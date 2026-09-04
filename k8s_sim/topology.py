"""
Hypergraph topology model for H-TAFM (Hypergraph-based Topology-Aware
Fragmentation Metric).

IMPORTANT SCOPE NOTE: this operates at a finer granularity than the rest of
k8s_sim. `k8s_sim.resource`/`cluster`/`policies` place pods on whole NODES
(one physical machine = one bin, GPUs are per-device but CPU/memory are
node-aggregate). H-TAFM's system model (Section 1.1 of the methodology)
places VMs on NUMA-NODE vertices, one level *below* the physical server, and
scores placements by their effect on a weighted hypergraph of Socket/Server/
Rack membership. That's a different bin-packing granularity, not just a
different scoring function, so it's implemented as its own module
(`topology.py` + `htafm.py`) with its own Topology/scheduler, rather than
registered as another entry in `policies.POLICIES` (which assumes
node-granularity placement). See docs/HTAFM.md for the full writeup,
including where this deviates from or resolves ambiguity in the source
methodology.

WHAT'S SYNTHESIZED VS. REAL: the Alibaba trace shipped in data/csv/ gives
per-node capacity (cpu_milli, memory_mib, gpu, model) but has no NUMA/
socket/rack topology at all -- real-world clusters don't publish that in
this kind of trace. So NUMA/socket/rack structure here is synthetically
generated from each node's total capacity (evenly split across NUMA
vertices, GPUs round-robinned across them) using configurable fan-out
parameters. CPU/memory/GPU capacities themselves are real trace data;
the sub-node hierarchy is a documented synthetic layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from collections import defaultdict

MILLI = 1000  # matches k8s_sim.resource.MILLI: 1000 milli-gpu == 1 whole GPU

DEFAULT_LEVEL_WEIGHTS = {"socket": 8.0, "server": 4.0, "rack": 1.0}
# w(NUMA) > w(Socket) > w(Server) > w(Rack) per the methodology (Sec. 2).
# Our vertices ARE NUMA-granularity (Sec. 1.1's Level 0), so the "NUMA
# hyperedge" of the general formulation is a singleton (one vertex) and
# contributes nothing to any of the three TAFI variants (a 1-vertex edge
# can never be "cut," has entropy 0, and its frag_level is 0-or-1 with no
# useful signal) -- it's correctly omitted here rather than approximated.


@dataclass
class TopoDemand:
    """A VM/pod's multi-resource demand: Section 1.1's d(v) = (CPU, Memory, GPU, Bandwidth).
    Bandwidth is not modeled (not present in the trace data or in any of the
    three TAFI variants' formulas)."""

    milli_cpu: int
    memory_mib: int = 0
    milli_gpu: int = 0     # per-device milli-gpu request
    gpu_number: int = 0
    gpu_type: str = ""
    name: str = ""

    def total_milli_gpu(self) -> int:
        return self.milli_gpu * self.gpu_number


@dataclass
class TopoVertex:
    """A Level-0 (NUMA node) vertex: the smallest schedulable unit."""

    id: str
    node_name: str     # originating physical node (== "server" grouping key)
    socket_id: str
    rack_id: str
    milli_cpu_capacity: int
    milli_cpu_left: int
    memory_mib_capacity: int
    memory_mib_left: int
    milli_gpu_left_list: List[int] = field(default_factory=list)
    gpu_type: str = ""

    def free_vector(self) -> "tuple[float, float, float]":
        return (float(self.milli_cpu_left), float(self.memory_mib_left),
                float(sum(self.milli_gpu_left_list)))

    def capacity_vector(self) -> "tuple[float, float, float]":
        return (float(self.milli_cpu_capacity), float(self.memory_mib_capacity),
                float(len(self.milli_gpu_left_list)) * MILLI)

    def fits(self, d: TopoDemand) -> bool:
        if self.milli_cpu_left < d.milli_cpu:
            return False
        if self.memory_mib_left < d.memory_mib:
            return False
        if d.gpu_type and self.gpu_type and d.gpu_type not in self.gpu_type.split("|"):
            return False
        if d.gpu_type and not self.gpu_type:
            return False
        if d.gpu_number > 0:
            need = d.gpu_number
            for left in self.milli_gpu_left_list:
                if left >= d.milli_gpu:
                    need -= 1
                    if need <= 0:
                        break
            if need > 0:
                return False
        return True

    def copy(self) -> "TopoVertex":
        return TopoVertex(
            id=self.id, node_name=self.node_name, socket_id=self.socket_id, rack_id=self.rack_id,
            milli_cpu_capacity=self.milli_cpu_capacity, milli_cpu_left=self.milli_cpu_left,
            memory_mib_capacity=self.memory_mib_capacity, memory_mib_left=self.memory_mib_left,
            milli_gpu_left_list=list(self.milli_gpu_left_list), gpu_type=self.gpu_type,
        )

    def place(self, d: TopoDemand) -> None:
        """Mutating in-place allocation. Raises if infeasible."""
        if not self.fits(d):
            raise ValueError(f"vertex {self.id} cannot host demand {d}")
        self.milli_cpu_left -= d.milli_cpu
        self.memory_mib_left -= d.memory_mib
        need = d.gpu_number
        for i in sorted(range(len(self.milli_gpu_left_list)), key=lambda i: self.milli_gpu_left_list[i]):
            if need <= 0:
                break
            if self.milli_gpu_left_list[i] >= d.milli_gpu:
                self.milli_gpu_left_list[i] -= d.milli_gpu
                need -= 1


@dataclass
class HyperEdge:
    id: str
    level: str  # "socket" | "server" | "rack"
    weight: float
    vertex_ids: List[str]


class Topology:
    """The hypergraph H = (V, E). Vertices are NUMA nodes; hyperedges group
    them by socket, server, and rack membership."""

    def __init__(self, vertices: Dict[str, TopoVertex], edges: Dict[str, HyperEdge]):
        self.vertices = vertices
        self.edges = edges
        self._vertex_edges: Dict[str, List[str]] = defaultdict(list)
        for e in edges.values():
            for vid in e.vertex_ids:
                self._vertex_edges[vid].append(e.id)

    def edges_touching(self, vertex_id: str) -> List[HyperEdge]:
        return [self.edges[eid] for eid in self._vertex_edges.get(vertex_id, [])]

    def copy(self) -> "Topology":
        return Topology(
            vertices={vid: v.copy() for vid, v in self.vertices.items()},
            edges=dict(self.edges),  # edges are structurally immutable (membership doesn't change)
        )

    @classmethod
    def from_nodes(cls, nodes: Sequence, numa_per_socket: int = 1, sockets_per_node: int = 2,
                    nodes_per_rack: int = 40,
                    level_weights: Optional[Dict[str, float]] = None) -> "Topology":
        """Build a synthetic Socket/Server/Rack hierarchy on top of real
        per-node capacities. `nodes` is a sequence of k8s_sim.resource.NodeResource
        (or anything with .name, .milli_cpu_capacity, .milli_gpu_left_list, .gpu_type
        and, if present, .memory_mib_capacity -- falls back to a heuristic
        memory estimate if not, see _memory_capacity_of below).

        numa_per_socket x sockets_per_node = NUMA vertices per physical node.
        Node capacity is split evenly across those vertices; GPUs are dealt
        round-robin so vertex GPU counts stay integral (a node with 8 GPUs
        and 4 NUMA vertices gets 2 GPUs each; a node with 3 GPUs and 4
        vertices gets [1,1,1,0]).
        """
        level_weights = level_weights or DEFAULT_LEVEL_WEIGHTS
        numa_per_node = numa_per_socket * sockets_per_node

        vertices: Dict[str, TopoVertex] = {}
        server_edges: Dict[str, HyperEdge] = {}
        socket_edges: Dict[str, HyperEdge] = {}
        rack_members: Dict[str, List[str]] = defaultdict(list)

        for node_idx, node in enumerate(nodes):
            rack_id = f"rack-{node_idx // nodes_per_rack:04d}"
            cpu_cap = node.milli_cpu_capacity
            mem_cap = _memory_capacity_of(node)
            gpu_list = list(node.milli_gpu_left_list)  # freshly-loaded nodes: left == capacity
            gpu_type = node.gpu_type

            cpu_share = cpu_cap // numa_per_node
            mem_share = mem_cap // numa_per_node
            # deal GPUs round-robin across the numa_per_node vertices
            gpu_buckets: List[List[int]] = [[] for _ in range(numa_per_node)]
            for gi, g in enumerate(gpu_list):
                gpu_buckets[gi % numa_per_node].append(g)

            server_vertex_ids = []
            for s in range(sockets_per_node):
                socket_id = f"{node.name}-sock{s}"
                socket_vertex_ids = []
                for n in range(numa_per_socket):
                    vidx = s * numa_per_socket + n
                    vid = f"{node.name}-numa{vidx}"
                    vertices[vid] = TopoVertex(
                        id=vid, node_name=node.name, socket_id=socket_id, rack_id=rack_id,
                        milli_cpu_capacity=cpu_share, milli_cpu_left=cpu_share,
                        memory_mib_capacity=mem_share, memory_mib_left=mem_share,
                        milli_gpu_left_list=gpu_buckets[vidx], gpu_type=gpu_type,
                    )
                    socket_vertex_ids.append(vid)
                    server_vertex_ids.append(vid)
                if numa_per_socket > 1:  # only a meaningful (non-degenerate) edge if >1 vertex
                    socket_edges[socket_id] = HyperEdge(
                        id=socket_id, level="socket", weight=level_weights["socket"],
                        vertex_ids=socket_vertex_ids,
                    )
                rack_members[rack_id].extend(socket_vertex_ids)

            if numa_per_node > 1:
                server_edges[node.name] = HyperEdge(
                    id=node.name, level="server", weight=level_weights["server"],
                    vertex_ids=server_vertex_ids,
                )

        rack_edges = {
            rid: HyperEdge(id=rid, level="rack", weight=level_weights["rack"], vertex_ids=vids)
            for rid, vids in rack_members.items() if len(vids) > 1
        }

        all_edges: Dict[str, HyperEdge] = {}
        all_edges.update(socket_edges)
        all_edges.update(server_edges)
        all_edges.update(rack_edges)
        return cls(vertices, all_edges)


def _memory_capacity_of(node) -> int:
    """NodeResource (k8s_sim.resource) doesn't track memory today -- pull it
    from the trace-loaded object if present (see topology_from_trace below,
    which attaches it), else fall back to a fixed ratio of CPU capacity as a
    documented placeholder (8 MiB per milli-cpu, i.e. 8 GiB/core)."""
    mem = getattr(node, "memory_mib_capacity", None)
    if mem is not None:
        return mem
    return node.milli_cpu_capacity * 8


# ---------------------------------------------------------------------------
# Loading real trace data, WITH the memory_mib column the core k8s_sim.resource
# module doesn't track. Deliberately independent of k8s_sim.trace.load_nodes_csv
# so H-TAFM's extra CPU+Memory+GPU dimension doesn't require touching the
# well-tested core resource model.
# ---------------------------------------------------------------------------

@dataclass
class _TraceNode:
    """Minimal stand-in for NodeResource that also carries memory_mib_capacity,
    used only as input to Topology.from_nodes."""
    name: str
    milli_cpu_capacity: int
    memory_mib_capacity: int
    milli_gpu_left_list: List[int]
    gpu_type: str


def load_topology_from_csv(node_csv_path: Optional[str] = None, **topology_kwargs) -> Topology:
    """Load real node capacities (cpu_milli, memory_mib, gpu, model) from an
    openb_node_list_*.csv and build a synthetic Socket/Server/Rack Topology
    on top of them. `**topology_kwargs` forwards to Topology.from_nodes
    (numa_per_socket, sockets_per_node, nodes_per_rack, level_weights)."""
    import csv
    import os
    from .trace import DATA_DIR, DEFAULT_NODE_TRACE

    path = node_csv_path or os.path.join(DATA_DIR, DEFAULT_NODE_TRACE)
    nodes: List[_TraceNode] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gpu_num = int(row["gpu"])
            gpu_type = (row.get("model") or "").strip()
            nodes.append(_TraceNode(
                name=row["sn"],
                milli_cpu_capacity=int(row["cpu_milli"]),
                memory_mib_capacity=int(row["memory_mib"]),
                milli_gpu_left_list=[MILLI] * gpu_num,
                gpu_type=gpu_type,
            ))
    return Topology.from_nodes(nodes, **topology_kwargs)
