from k8s_sim import NodeResource, PodResource, TargetPod  # noqa: F401
from k8s_sim.fragmentation import (  # noqa: F401
    get_node_pod_frag, node_gpu_share_frag_amount, node_gpu_share_frag_amount_score,
    Q1_LACK_BOTH, Q2_LACK_GPU, Q3_SATISFIED, Q4_LACK_CPU, XL_SATISFIED, XR_LACK_CPU, NO_ACCESS,
)


def test_frag_xl_satisfied(cpu_only_node):
    pod = PodResource(milli_cpu=1000, milli_gpu=0, gpu_number=0)
    assert get_node_pod_frag(cpu_only_node, pod) == XL_SATISFIED


def test_frag_xr_lack_cpu(cpu_only_node):
    pod = PodResource(milli_cpu=999999, milli_gpu=0, gpu_number=0)
    assert get_node_pod_frag(cpu_only_node, pod) == XR_LACK_CPU


def test_frag_no_access(empty_node):
    pod = PodResource(milli_cpu=500, milli_gpu=500, gpu_number=1, gpu_type="A100")
    assert get_node_pod_frag(empty_node, pod) == NO_ACCESS


def test_frag_q3_satisfied(empty_node):
    pod = PodResource(milli_cpu=500, milli_gpu=500, gpu_number=1, gpu_type="V100")
    assert get_node_pod_frag(empty_node, pod) == Q3_SATISFIED


def test_frag_q4_lack_cpu(empty_node):
    pod = PodResource(milli_cpu=999999, milli_gpu=500, gpu_number=1, gpu_type="V100")
    assert get_node_pod_frag(empty_node, pod) == Q4_LACK_CPU


def test_frag_q2_lack_gpu(partially_used_node):
    # needs 2 GPUs >= 800m each; node has only one device with 1000m free
    pod = PodResource(milli_cpu=1000, milli_gpu=800, gpu_number=2, gpu_type="V100")
    assert get_node_pod_frag(partially_used_node, pod) == Q2_LACK_GPU


def test_frag_q1_lack_both(partially_used_node):
    pod = PodResource(milli_cpu=999999, milli_gpu=800, gpu_number=2, gpu_type="V100")
    assert get_node_pod_frag(partially_used_node, pod) == Q1_LACK_BOTH


def test_frag_amount_fully_idle_node_is_all_q3(empty_node, typical_pods):
    # every typical pod fits comfortably -> everything should land in Q3,
    # meaning zero *fragmented* (non-Q3) amount.
    small_gpu_typical = [TargetPod(pod=PodResource(milli_cpu=500, milli_gpu=250, gpu_number=1, gpu_type="V100"),
                                    percentage=1.0)]
    score = node_gpu_share_frag_amount_score(empty_node, small_gpu_typical)
    assert score == 0.0


def test_frag_amount_score_nonnegative(cluster_nodes, typical_pods):
    for node in cluster_nodes:
        score = node_gpu_share_frag_amount_score(node, typical_pods)
        assert score >= 0.0


def test_frag_amount_empty_typical_pods_is_zero(empty_node):
    assert node_gpu_share_frag_amount_score(empty_node, []) == 0.0


def test_frag_amount_no_gpu_left_is_zero(partially_used_node, typical_pods):
    fully_used = partially_used_node.copy()
    fully_used.milli_gpu_left_list = [0, 0, 0, 0]
    score = node_gpu_share_frag_amount_score(fully_used, typical_pods)
    assert score == 0.0
