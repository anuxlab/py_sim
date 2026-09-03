import pytest

from k8s_sim import NodeResource, PodResource
from k8s_sim.fragmentation import build_typical_pods_uniform


@pytest.fixture
def empty_node():
    """A fresh, fully-idle 4-GPU node with plenty of CPU."""
    return NodeResource(
        name="node-test",
        milli_cpu_left=32000,
        milli_cpu_capacity=32000,
        milli_gpu_left_list=[1000, 1000, 1000, 1000],
        gpu_type="V100",
    )


@pytest.fixture
def partially_used_node():
    """A node with mixed GPU occupancy: one fully free, one lightly used,
    one heavily used, one fully used."""
    return NodeResource(
        name="node-partial",
        milli_cpu_left=16000,
        milli_cpu_capacity=32000,
        milli_gpu_left_list=[1000, 750, 100, 0],
        gpu_type="V100",
    )


@pytest.fixture
def cpu_only_node():
    return NodeResource(
        name="node-cpu",
        milli_cpu_left=16000,
        milli_cpu_capacity=16000,
        milli_gpu_left_list=[],
        gpu_type="",
    )


@pytest.fixture
def cluster_nodes(empty_node, partially_used_node, cpu_only_node):
    return [empty_node, partially_used_node, cpu_only_node]


@pytest.fixture
def typical_pods():
    shapes = [
        PodResource(milli_cpu=500, milli_gpu=250, gpu_number=1, gpu_type="V100"),
        PodResource(milli_cpu=2000, milli_gpu=1000, gpu_number=1, gpu_type="V100"),
        PodResource(milli_cpu=1000, milli_gpu=0, gpu_number=0, gpu_type=""),
    ]
    return build_typical_pods_uniform(shapes)


@pytest.fixture
def sample_pods():
    """A battery of representative pod shapes used to stress-test policies:
    fractional-GPU share, full single-GPU, multi-GPU exclusive, CPU-only,
    and a zero-cpu edge case."""
    return [
        PodResource(milli_cpu=500, milli_gpu=250, gpu_number=1, gpu_type="V100", name="share-small"),
        PodResource(milli_cpu=1000, milli_gpu=500, gpu_number=1, gpu_type="V100", name="share-half"),
        PodResource(milli_cpu=2000, milli_gpu=1000, gpu_number=1, gpu_type="V100", name="full-1gpu"),
        PodResource(milli_cpu=4000, milli_gpu=1000, gpu_number=2, gpu_type="V100", name="full-2gpu"),
        PodResource(milli_cpu=1000, milli_gpu=0, gpu_number=0, gpu_type="", name="cpu-only"),
        PodResource(milli_cpu=0, milli_gpu=0, gpu_number=0, gpu_type="", name="zero-request"),
    ]
