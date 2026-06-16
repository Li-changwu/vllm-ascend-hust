# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.


from dataclasses import dataclass

import torch
import torch_npu
from torch.nn.functional import pad
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.mxfp_compat import (
    ensure_mxfp8_moe_available,
)
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.utils import (
    dispose_tensor,
    enable_custom_op,
    get_weight_prefetch_method,
)


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
    return fusion and dynamic_eplb and enable_custom_op()


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    w2_scale: list[torch.Tensor] | torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    dynamic_eplb: bool = False,
    use_mxfp_quant: bool = False,
    act_quant_type: torch.dtype = torch.float8_e4m3fn,
    weight_quant_type: torch.dtype | None = None,
    scale_type: torch.dtype | None = None,
    per_token_scale_type: torch.dtype | None = None,
    use_bf16: bool = True,
) -> torch.Tensor:
    input_hidden_dtype = hidden_states.dtype
    use_gmm_swiglu_quant_fusion = use_mxfp_quant or (fusion and not dynamic_eplb)

    if use_mxfp_quant:
        ensure_mxfp8_moe_available("MXFP MoE MLP path")

        if w1_scale_bias is not None or w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")
        if w1_offset is not None or w2_offset is not None:
            raise NotImplementedError("MXFP path does not support antiquant offset yet.")

    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = (
            DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale) if use_mxfp_quant else dynamic_scale
        )
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(hidden_states)
    is_mc2 = _EXTRA_CTX.moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
            )
        elif use_gmm_swiglu_quant_fusion:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=None,
                use_mxfp_quant=use_mxfp_quant,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=None,
            fallback_output_dtype=w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype,
        )
    elif w1_offset is not None:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
        dispose_tensor(unquantized_hidden_states)
        # act_fn: swiglu
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                bias=bias1,
            )
        elif use_gmm_swiglu_quant_fusion:
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=bias1,
                use_mxfp_quant=use_mxfp_quant,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                scale=w1_scale,
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            if HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states, group_list=group_list, group_list_type=group_list_type
                )
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=bias2,
            fallback_output_dtype=_output_dtype,
        )
    return hidden_states


# --- GMM1 in-situ probe (env-gated, default OFF, no behavior change) ---
# Set VLLM_ASCEND_GMM1_PROBE_BACKEND=custom to route GMM1 (gate/up) through the
# custom NZ grouped-matmul kernel; set VLLM_ASCEND_GMM1_PROBE_PATH to a file to
# append per-call synchronized GMM1 latency (ms). Both unset => original path.
import os as _gmm1_os

_GMM1_PROBE_BACKEND = _gmm1_os.environ.get("VLLM_ASCEND_GMM1_PROBE_BACKEND", "")
_GMM1_PROBE_PATH = _gmm1_os.environ.get("VLLM_ASCEND_GMM1_PROBE_PATH", "")


def _gmm1_custom_nz(hidden_states, w1, group_list, group_list_type):
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    if group_list_type == 0:  # cumsum -> counts
        gl = group_list.to(torch.int64)
        counts = torch.cat([gl[:1], gl[1:] - gl[:-1]])
    else:
        counts = group_list.to(torch.int64)
    w_nz = torch_npu.npu_format_cast(w1.contiguous(), ACL_FORMAT_FRACTAL_NZ)
    active_ids = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
    active_counts = counts.index_select(0, active_ids)
    gl_pairs = torch.stack([active_ids, active_counts], dim=1).to(torch.int64)
    return torch.ops._C_ascend.moe_grouped_matmul(
        hidden_states, w_nz, gl_pairs, 2, 0, 2,
    )[0]


def _gmm1_run(hidden_states, w1, w1_bias, group_list, group_list_type):
    use_custom = _GMM1_PROBE_BACKEND == "custom"
    if not _GMM1_PROBE_PATH:
        if use_custom:
            return _gmm1_custom_nz(hidden_states, w1, group_list, group_list_type)
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states], weight=[w1],
            bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
            split_item=2, group_list_type=group_list_type,
            group_type=0, group_list=group_list,
        )[0]
    # timed path
    from time import perf_counter
    torch_npu.npu.synchronize()
    _t0 = perf_counter()
    if use_custom:
        out = _gmm1_custom_nz(hidden_states, w1, group_list, group_list_type)
    else:
        out = torch_npu.npu_grouped_matmul(
            x=[hidden_states], weight=[w1],
            bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
            split_item=2, group_list_type=group_list_type,
            group_type=0, group_list=group_list,
        )[0]
    torch_npu.npu.synchronize()
    _dt_ms = (perf_counter() - _t0) * 1000.0
    try:
        with open(_GMM1_PROBE_PATH, "a") as _f:
            _f.write(f"{_dt_ms}\n")
    except OSError:
        pass
    return out


def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
) -> torch.Tensor:
    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    act_name = getattr(activation, "value", activation)

    gate_up_out = _gmm1_run(hidden_states, w1, w1_bias, group_list, group_list_type)

    if act_name == "swigluoai":
        num_experts, _, hidden_size = w1.shape
        gate_up_out = AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out.view(-1, hidden_size))
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    """
    Unified MoE MLP entry.
    Quant path is dispatched by DeviceOperator with explicit typed kernel flags.
    """
    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    dynamic_scale = mlp_compute_input.dynamic_scale
    topk_scales = mlp_compute_input.topk_scales
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_bias = mlp_compute_input.weights.w1_bias
    w2_bias = mlp_compute_input.weights.w2_bias
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    w1_scale_bias = mlp_compute_input.weights.w1_scale_bias
    w2_scale_bias = mlp_compute_input.weights.w2_scale_bias
    w1_offset = mlp_compute_input.weights.w1_offset
    w2_offset = mlp_compute_input.weights.w2_offset
    activation = mlp_compute_input.activation
    need_trans = mlp_compute_input.need_trans
    dynamic_eplb = mlp_compute_input.dynamic_eplb
    fusion = mlp_compute_input.fusion

    if not mlp_compute_input.quant.is_quant:
        return unquant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            activation=activation,
            group_list=group_list,
            group_list_type=group_list_type,
            topk_scales=topk_scales,
            need_trans=need_trans,
        )

    assert w1_scale is not None and w2_scale is not None
    act_quant_type = torch.float8_e4m3fn
    weight_quant_type = torch.float8_e4m3fn
    scale_type = None
    per_token_scale_type = None
    use_bf16 = hidden_states.dtype == torch.bfloat16
    use_mxfp_quant = mlp_compute_input.quant.is_mxfp

    if use_mxfp_quant:
        mxfp = mlp_compute_input.quant.mxfp
        assert mxfp is not None, "mlp_compute_input.quant.mxfp is required for MXFP quant types."
        act_quant_type = mxfp.act_quant_type or act_quant_type
        weight_quant_type = mxfp.weight_quant_type or weight_quant_type
        scale_type = mxfp.scale_dtype
        per_token_scale_type = mxfp.per_token_scale_dtype
        use_bf16 = mxfp.use_bf16

    return quant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group_list=group_list,
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        w1_offset=w1_offset,
        w2_offset=w2_offset,
        fusion=fusion,
        dynamic_eplb=dynamic_eplb,
        use_mxfp_quant=use_mxfp_quant,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_bf16=use_bf16,
    )


# ── P5: env-var controlled custom GMM1 kernel toggle ─────────────────────────
# Set VLLM_ASCEND_USE_CUSTOM_GMM=1 to replace torch_npu GMM1 with _C_ascend.
# This runs at import time so it takes effect inside the vLLM API server process.
import os as _os
if _os.environ.get("VLLM_ASCEND_USE_CUSTOM_GMM", "") == "1":
    import vllm_ascend.vllm_ascend_C as _custom_ops

    _orig_unquant = unquant_apply_mlp

    def _custom_unquant_apply_mlp(
        hidden_states, w1, w2, group_list,
        w1_bias=None, w2_bias=None, activation=None,
        group_list_type=1, topk_scales=None, need_trans=True,
    ):
        if need_trans:
            w1 = w1.transpose(1, 2)
            w2 = w2.transpose(1, 2)
        # Custom kernel expects ND format weight (transposed, contiguous)
        # torch_binding.cpp dispatches to aclnnMoeGroupedMatmul (ND path)
        w1_nd = w1.contiguous()
        act_name = getattr(activation, "value", activation)
        gl = group_list.to(torch.int64) if group_list_type == 1 else group_list
        if group_list_type == 0:
            diffs = gl[1:] - gl[:-1]
            gl = torch.cat([gl[:1], diffs])
        active_ids = torch.nonzero(gl > 0).reshape(-1).to(torch.int64)
        active_counts = gl.index_select(0, active_ids.long())
        gl2 = torch.stack([active_ids, active_counts], dim=1)
        gate_up_out = torch.ops._C_ascend.moe_grouped_matmul(
            hidden_states, w1_nd, gl2, 2, 0, 2)[0]
        if act_name == "swigluoai":
            gate_up_out = AscendSwigluOAIAndMul.swiglu_oai_forward(
                gate_up_out.view(-1, w1.shape[2]))
        else:
            gate_up_out = torch_npu.npu_swiglu(gate_up_out)
        if topk_scales is not None:
            gate_up_out = gate_up_out * topk_scales
        w2_bias_arg = None
        if w2_bias is not None:
            w2_bias_arg = [w2_bias.to(dtype=torch.float32)]
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[gate_up_out], weight=[w2],
            bias=w2_bias_arg,
            split_item=2, group_list_type=group_list_type,
            group_type=0, group_list=group_list)[0]
        return hidden_states

    unquant_apply_mlp = _custom_unquant_apply_mlp
