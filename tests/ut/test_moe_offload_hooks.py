from unittest.mock import MagicMock, patch

import torch

from vllm_ascend._moe_offload_null import get_moe_offload_runtime, get_moe_pipeline_profiler
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, MoECommMethod
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEOffloadParams,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)


def test_null_moe_offload_hooks_keep_optional_plugin_disabled():
    runtime = get_moe_offload_runtime()
    topk_ids = object()
    topk_weights = object()

    traced_ids, traced_weights = runtime.trace_routing(
        layer_id=0,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_experts=8,
    )

    assert traced_ids is topk_ids
    assert traced_weights is topk_weights
    assert runtime.config.enabled is False
    assert runtime.should_use_fixed_slots is False
    assert runtime.should_use_fixed_slot_plan_for_layer(0) is False
    assert get_moe_pipeline_profiler().enabled is False


def test_moe_offload_plan_preserves_current_runtime_fields():
    original_w1 = torch.randn(3, 8, 16)
    original_w2 = torch.randn(3, 16, 8)
    slot_w1 = torch.randn(2, 8, 16)
    slot_w2 = torch.randn(2, 16, 8)
    slot_log2phy = torch.tensor([0, 1, 0], dtype=torch.int32)
    fused_experts_input = MoEFusedExpertsInput(
        hidden_states=torch.randn(2, 8),
        topk_weights=torch.randn(2, 2),
        topk_ids=torch.tensor([[0, 1], [2, 1]], dtype=torch.int32),
        weights=MoEWeights(w1=original_w1, w2=original_w2),
        routing=MoERoutingParams(
            expert_map=None,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(),
        swiglu_limit=7.0,
        offload=MoEOffloadParams(
            enabled=True,
            layer_id=6,
            num_logical_experts=3,
            expected_device_type="cpu",
            step_id=11,
        ),
    )
    prepared = MagicMock(
        w1=slot_w1,
        w2=slot_w2,
        log2phy=slot_log2phy,
        physical_expert_count=2,
    )
    runtime = MagicMock(should_use_layered_runtime=False)
    runtime.prepare_fixed_slot_plan.return_value = prepared
    comm_method = object.__new__(AllGatherCommImpl)

    with patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_offload_runtime",
        return_value=runtime,
    ):
        result = MoECommMethod._maybe_apply_moe_offload_plan(comm_method, fused_experts_input)

    assert result.weights.w1 is slot_w1
    assert result.weights.w2 is slot_w2
    assert result.routing.log2phy is slot_log2phy
    assert result.routing.physical_expert_count == 2
    assert result.swiglu_limit == 7.0
    assert result.offload is not None
    assert result.offload.step_id == 11
    prepared.validate_backend_ready.assert_called_once_with(expected_device_type="cpu")
    runtime.prepare_fixed_slot_plan.assert_called_once_with(
        layer_id=6,
        step_id=11,
        active_experts=(0, 1, 2),
        num_logical_experts=3,
        device=fused_experts_input.topk_ids.device,
    )
