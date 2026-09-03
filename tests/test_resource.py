import pytest

from k8s_sim import PodResource


def test_pod_total_milli_gpu():
    p = PodResource(milli_cpu=1000, milli_gpu=500, gpu_number=2)
    assert p.total_milli_gpu() == 1000


def test_pod_is_gpu_share():
    assert PodResource(milli_cpu=1000, milli_gpu=500, gpu_number=1).is_gpu_share()
    assert not PodResource(milli_cpu=1000, milli_gpu=1000, gpu_number=1).is_gpu_share()
    assert not PodResource(milli_cpu=1000, milli_gpu=1000, gpu_number=2).is_gpu_share()


def test_node_totals(empty_node):
    assert empty_node.total_milli_gpu_left() == 4000
    assert empty_node.fully_free_gpu_num() == 4


def test_node_fits_cpu_only(cpu_only_node):
    fits = PodResource(milli_cpu=8000, milli_gpu=0, gpu_number=0)
    too_big = PodResource(milli_cpu=20000, milli_gpu=0, gpu_number=0)
    assert cpu_only_node.fits(fits)
    assert not cpu_only_node.fits(too_big)


def test_node_fits_gpu_type_mismatch(empty_node):
    pod = PodResource(milli_cpu=500, milli_gpu=500, gpu_number=1, gpu_type="A100")
    assert not empty_node.fits(pod)  # node is V100 only


def test_node_fits_gpu_type_union(empty_node):
    pod = PodResource(milli_cpu=500, milli_gpu=500, gpu_number=1, gpu_type="A100|V100")
    assert empty_node.fits(pod)


def test_sub_reduces_capacity(empty_node):
    pod = PodResource(milli_cpu=2000, milli_gpu=500, gpu_number=1)
    out = empty_node.sub(pod)
    assert out.milli_cpu_left == empty_node.milli_cpu_left - 2000
    assert sum(out.milli_gpu_left_list) == sum(empty_node.milli_gpu_left_list) - 500
    # original untouched (sub returns a new object)
    assert empty_node.milli_cpu_left == 32000


def test_sub_packs_least_sufficient_device_first(partially_used_node):
    # milli_gpu_left_list = [1000, 750, 100, 0]; a 100m pod should land on
    # the smallest device that still fits it (index 2, which has exactly 100).
    pod = PodResource(milli_cpu=1000, milli_gpu=100, gpu_number=1)
    out = partially_used_node.sub(pod)
    assert out.milli_gpu_left_list == [1000, 750, 0, 0]


def test_sub_raises_when_infeasible(empty_node):
    pod = PodResource(milli_cpu=999999, milli_gpu=0, gpu_number=0)
    with pytest.raises(ValueError):
        empty_node.sub(pod)


def test_sub_raises_when_gpu_infeasible(partially_used_node):
    # needs 2 GPUs with >=800m each; only one device (1000m) qualifies
    pod = PodResource(milli_cpu=1000, milli_gpu=800, gpu_number=2)
    with pytest.raises(ValueError):
        partially_used_node.sub(pod)


def test_add_reverses_sub(empty_node):
    pod = PodResource(milli_cpu=2000, milli_gpu=500, gpu_number=1)
    reduced = empty_node.sub(pod)
    restored = reduced.add(pod)
    assert restored.milli_cpu_left == empty_node.milli_cpu_left
    assert sorted(restored.milli_gpu_left_list) == sorted(empty_node.milli_gpu_left_list)


def test_add_raises_over_capacity(empty_node):
    pod = PodResource(milli_cpu=1000, milli_gpu=0, gpu_number=0)
    with pytest.raises(ValueError):
        empty_node.add(pod)  # already at capacity, adding back would overflow
