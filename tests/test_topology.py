import pytest

from k8s_sim.topology import Topology, TopoVertex, TopoDemand, MILLI


def _node(name, cpu=32000, mem=131072, gpu=4, gpu_type="V100"):
    from k8s_sim.topology import _TraceNode
    return _TraceNode(name=name, milli_cpu_capacity=cpu, memory_mib_capacity=mem,
                       milli_gpu_left_list=[MILLI] * gpu, gpu_type=gpu_type)


def test_from_nodes_vertex_count():
    nodes = [_node(f"n{i}") for i in range(4)]
    topo = Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=2, nodes_per_rack=100)
    assert len(topo.vertices) == 8  # 4 nodes x 2 numa vertices each


def test_from_nodes_capacity_split_evenly():
    nodes = [_node("n0", cpu=32000, mem=131072, gpu=4)]
    topo = Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=2)
    verts = [v for v in topo.vertices.values() if v.node_name == "n0"]
    assert len(verts) == 2
    assert sum(v.milli_cpu_capacity for v in verts) == 32000
    assert sum(v.memory_mib_capacity for v in verts) == 131072
    assert sum(len(v.milli_gpu_left_list) for v in verts) == 4


def test_server_edge_groups_vertices_of_same_node():
    nodes = [_node("n0"), _node("n1")]
    topo = Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=2, nodes_per_rack=100)
    assert "n0" in topo.edges
    edge = topo.edges["n0"]
    assert edge.level == "server"
    assert set(edge.vertex_ids) == {v.id for v in topo.vertices.values() if v.node_name == "n0"}


def test_rack_edge_groups_vertices_across_nodes():
    nodes = [_node(f"n{i}") for i in range(6)]
    topo = Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=1, nodes_per_rack=3)
    rack_edges = [e for e in topo.edges.values() if e.level == "rack"]
    assert len(rack_edges) == 2  # 6 nodes / 3 per rack
    for e in rack_edges:
        assert len(e.vertex_ids) == 3


def test_single_numa_per_node_omits_server_edge():
    # numa_per_node == 1 (1 socket x 1 numa) -> server hyperedge would be a
    # singleton and is correctly omitted
    nodes = [_node("n0")]
    topo = Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=1, nodes_per_rack=100)
    assert "n0" not in topo.edges
    assert len(topo.vertices) == 1


def test_edges_touching_returns_all_levels():
    nodes = [_node(f"n{i}") for i in range(3)]
    topo = Topology.from_nodes(nodes, numa_per_socket=1, sockets_per_node=2, nodes_per_rack=3)
    vid = next(v.id for v in topo.vertices.values() if v.node_name == "n0")
    touched = topo.edges_touching(vid)
    levels = {e.level for e in touched}
    assert "server" in levels
    assert "rack" in levels


def test_vertex_fits_and_place():
    v = TopoVertex(id="v", node_name="n", socket_id="s", rack_id="r",
                    milli_cpu_capacity=8000, milli_cpu_left=8000,
                    memory_mib_capacity=16384, memory_mib_left=16384,
                    milli_gpu_left_list=[1000, 1000], gpu_type="V100")
    d = TopoDemand(milli_cpu=2000, memory_mib=4096, milli_gpu=500, gpu_number=1, gpu_type="V100")
    assert v.fits(d)
    v.place(d)
    assert v.milli_cpu_left == 6000
    assert v.memory_mib_left == 12288
    assert sum(v.milli_gpu_left_list) == 1500


def test_vertex_fits_rejects_gpu_type_mismatch():
    v = TopoVertex(id="v", node_name="n", socket_id="s", rack_id="r",
                    milli_cpu_capacity=8000, milli_cpu_left=8000,
                    memory_mib_capacity=16384, memory_mib_left=16384,
                    milli_gpu_left_list=[1000], gpu_type="V100")
    d = TopoDemand(milli_cpu=1000, memory_mib=1024, milli_gpu=500, gpu_number=1, gpu_type="A100")
    assert not v.fits(d)


def test_place_raises_when_infeasible():
    v = TopoVertex(id="v", node_name="n", socket_id="s", rack_id="r",
                    milli_cpu_capacity=1000, milli_cpu_left=1000,
                    memory_mib_capacity=1024, memory_mib_left=1024,
                    milli_gpu_left_list=[], gpu_type="")
    d = TopoDemand(milli_cpu=99999, memory_mib=0)
    with pytest.raises(ValueError):
        v.place(d)


def test_load_topology_from_csv_smoke():
    from k8s_sim.topology import load_topology_from_csv
    import os
    from k8s_sim.trace import DATA_DIR
    if not os.path.isdir(DATA_DIR):
        pytest.skip("data/csv/ not present in this checkout")
    topo = load_topology_from_csv(nodes_per_rack=50)
    assert len(topo.vertices) > 0
    assert len(topo.edges) > 0
    server_edges = [e for e in topo.edges.values() if e.level == "server"]
    rack_edges = [e for e in topo.edges.values() if e.level == "rack"]
    assert server_edges and rack_edges
