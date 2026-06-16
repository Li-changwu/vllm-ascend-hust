/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
using namespace AscendC;


// P4: Enable doMTE2Preload=2 (N-dimension weight preload) to overlap
// MTE weight loading of tile[n+1] with Cube computation of tile[n].
// Qwen3-30B-A3B GMM1 has N=768 >> M (1–16 tokens), so N-preload
// overlaps the dominant weight-loading MTE traffic with Cube compute,
// reducing effective HBM latency regardless of L2 cache state.
constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false, false, false, false, 2, 0, 0,
                                         0, 0, 0, 0, true};
struct GMMConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint64_t wBaseOffset = 0;
    uint64_t nAxisBaseOffset = 0;
    uint64_t mAxisBaseOffset = 0;
    uint64_t xBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wOutOffset = 0;
};


template <typename T, typename T2, CubeFormat formatWeight, bool transWeight, bool FuseSwiGLU = false>
class KernelMoeGMMNoQuant {

protected:
  using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
  using weightType = MatmulType<AscendC::TPosition::GM, formatWeight, T, transWeight>;
  using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
  using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
  using mmT = matmul::MatmulImpl<xType, weightType, yType, biasType, matmulCFGUnitFlag>;
  mmT mm;

  // P5: second matmul instance with float (int32_t) output for SwiGLU fusion.
  // AscendC matmul accumulator is fp32; writing to int32_t preserves full
  // precision, enabling float Vector ops (Silu/Mul) which are well-supported,
  // unlike bf16/half TensorTrait which is unavailable in CANN 8.5.1.
  using yTypeFloat = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
  using mmFloatT = matmul::MatmulImpl<xType, weightType, yTypeFloat, biasType, matmulCFGUnitFlag>;
  mmFloatT mm_float;

  MoeGroupedMatmulTilingData tiling_;
  AscendC::TPipe *pipe_ = nullptr;

  GlobalTensor<T> x_gm_;
  GlobalTensor<T> weight_gm_;
  GlobalTensor<T> y_gm_;
  GlobalTensor<T2> group_list_gm_;
  // P5: workspace for fp32 intermediate gate_up output (matmul → SwiGLU)
  GlobalTensor<float> workspace_gm_;
  ListTensorDesc x_list_;
  ListTensorDesc weight_list_;
  ListTensorDesc y_list_;

  // P1 optimization: cached tiling fields (avoid repeated GM tiling reads)
  uint32_t k_ = 0;
  uint32_t n_ = 0;
  uint32_t base_m_ = 0;
  uint32_t base_n_ = 0;

  uint32_t core_idx;
  uint32_t used_core_num;
  constexpr static bool transposeW = transWeight;
  constexpr static uint32_t UB_BLOCK_UNIT_SIZE = 32;

public:
  __aicore__ inline KernelMoeGMMNoQuant(AscendC::TPipe *pipe) {pipe_ = pipe;}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y, \
      GM_ADDR workspace, const MoeGroupedMatmulTilingData *tiling) {
      core_idx = GetBlockIdx();
      tiling_ = *tiling;
      used_core_num = GetBlockNum();
      group_list_gm_.SetGlobalBuffer((__gm__ T2*)group_list);
      x_list_.Init((__gm__ void*)x);
      weight_list_.Init((__gm__ void*)weight);
      y_list_.Init((__gm__ void*)y);
      GM_ADDR x_first_addr = (__gm__ uint8_t*)x_list_.GetDataPtr<__gm__ uint8_t>(0);
      GM_ADDR weight_first_addr = (__gm__ uint8_t*)weight_list_.GetDataPtr<__gm__ uint8_t>(0);
      GM_ADDR y_first_addr = (__gm__ uint8_t*)y_list_.GetDataPtr<__gm__ uint8_t>(0);
      x_gm_.SetGlobalBuffer((__gm__ T*)x_first_addr);
      weight_gm_.SetGlobalBuffer((__gm__ T*)weight_first_addr);
      y_gm_.SetGlobalBuffer((__gm__ T*)y_first_addr);

      // P1: cache tiling fields in registers (eliminates repeated tiling_ GM reads)
      k_ = tiling_.k;
      n_ = tiling_.n;
      base_m_ = tiling_.single_m;
      base_n_ = tiling_.single_n;

      mm.Init(&tiling_.mm_tiling, pipe_);
      if constexpr (FuseSwiGLU) {
        mm_float.Init(&tiling_.mm_tiling_float, pipe_);
      }
      workspace_gm_.SetGlobalBuffer((__gm__ float*)workspace);
  }

  __aicore__ inline void Process() {
      constexpr uint32_t group_list_inner_shape = 2u;
      uint32_t group_list_shape_size = tiling_.group_num * group_list_inner_shape;
      GMMConfig mn_config;

      for (uint32_t loop = 0, count = 0; loop < group_list_shape_size; loop += group_list_inner_shape) {
        int32_t split_value = static_cast<int32_t>(group_list_gm_.GetValue(loop + 1));
        if (split_value <= 0) {
          break;
        }
        uint32_t group_idx = static_cast<uint32_t>(group_list_gm_.GetValue(loop));

        // P1: use cached tiling fields (k_, n_) instead of tiling_. reads
        mn_config.mAxisBaseOffset += mn_config.m;
        mn_config.xBaseOffset += mn_config.m * k_;
        mn_config.yBaseOffset += mn_config.m * n_;

        this->SetMNConfigCached(static_cast<uint32_t>(split_value), mn_config);
        mn_config.nAxisBaseOffset = group_idx * n_;
        if constexpr (formatWeight == CubeFormat::NZ) {
          mn_config.wBaseOffset = AlignUp(k_, 16) * AlignUp(mn_config.nAxisBaseOffset, 16);
        } else {
          mn_config.wBaseOffset = k_ * mn_config.nAxisBaseOffset;
        }

        mn_config.blockDimM = Ceil(mn_config.m, mn_config.singleM);
        mn_config.blockDimN = Ceil(n_, base_n_);
        uint32_t cur_count = count + mn_config.blockDimM * mn_config.blockDimN;
        uint32_t cur_block = this->core_idx >= count ? this->core_idx : this->core_idx + used_core_num;

        while (cur_block < cur_count) {
            mn_config.mIdx = (cur_block - count) / mn_config.blockDimN;
            mn_config.nIdx = (cur_block - count) % mn_config.blockDimN;
            this->MMCompute(group_idx, mn_config, this->core_idx);
            cur_block += used_core_num;
        }
        count = cur_count % used_core_num;
      }
  }

protected:
  __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t base) {
      return (a + base - 1) / base * base;
  }

  __aicore__ inline uint32_t Ceil(uint32_t a, uint32_t base) {
      if (base == 0) {
        return a;
      }
      return (a + base - 1) / base;
  }

  // P1: Use cached tiling fields (k_, n_, base_m_, base_n_) instead of tiling_ GM reads
  __aicore__ inline void SetMNConfigCached(const uint32_t m, GMMConfig & mn_config) {
      mn_config.m = m;
      mn_config.k = k_;
      mn_config.n = n_;
      mn_config.baseM = base_m_;
      mn_config.baseN = base_n_;
      mn_config.singleM = base_m_;
      mn_config.singleN = base_n_;
  }

  __aicore__ inline void MMCompute(uint32_t group_idx, GMMConfig& mn_config, uint32_t core_idx) {
      uint32_t tail_n = mn_config.nIdx * mn_config.singleN;
      uint32_t cur_single_n = mn_config.nIdx < mn_config.blockDimN - 1 ? mn_config.singleN : mn_config.n - tail_n;
      uint32_t cur_single_m = mn_config.mIdx < mn_config.blockDimM - 1 ? mn_config.singleM
                                                                    : mn_config.m - mn_config.mIdx * mn_config.singleM;
      uint64_t x_offset = mn_config.mIdx * mn_config.singleM * mn_config.k;
      uint64_t out_offset = mn_config.mIdx * mn_config.singleM * mn_config.n + tail_n;
      GlobalTensor<T> weight_gm_local = GetGlobalBufferW(group_idx, tail_n, mn_config);

      GlobalTensor<T> x_local = x_gm_[mn_config.xBaseOffset + x_offset];
      x_local.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);

      mm.SetOrgShape(mn_config.m, mn_config.n, mn_config.k);
      mm.SetSingleShape(cur_single_m, cur_single_n, mn_config.k);
      mm.SetTensorA(x_local, false);
      mm.SetTensorB(weight_gm_local, transposeW);

      if constexpr (FuseSwiGLU) {
          // P5: GMM1+SwiGLU fusion via fp32 intermediate.
          // CANN 8.5.1 does not support bf16/half TensorTrait for LocalTensor,
          // but float and int32_t are well-supported.
          // Strategy: matmul writes fp32 to workspace (preserving accumulator
          // precision), Vector ops apply SwiGLU in float, then DataCopy
          // writes bf16 result to output (automatic fp32→bf16 conversion).
          mm_float.SetOrgShape(mn_config.m, mn_config.n, mn_config.k);
          mm_float.SetSingleShape(cur_single_m, cur_single_n, mn_config.k);
          mm_float.SetTensorA(x_local, false);
          mm_float.SetTensorB(weight_gm_local, transposeW);
          mm_float.template IterateAll<false>(workspace_gm_[mn_config.yBaseOffset + out_offset], 0);

          // SwiGLU on float in UB: gate * silu(up)
          uint32_t half_n = cur_single_n / 2;
          for (uint32_t m = 0; m < cur_single_m; ++m) {
              uint64_t row_base = mn_config.yBaseOffset + out_offset + static_cast<uint64_t>(m) * mn_config.n;

              TBuf<TPosition::VECCALC> up_buf, gate_buf;
              LocalTensor<float> up_f32   = up_buf.template Get<float>(half_n);
              LocalTensor<float> gate_f32 = gate_buf.template Get<float>(half_n);

              DataCopyParams cp;
              cp.blockCount = 1;
              cp.blockLen   = half_n * sizeof(float);
              DataCopy(up_f32,   workspace_gm_[row_base + half_n], cp);
              DataCopy(gate_f32, workspace_gm_[row_base], cp);

              Silu(up_f32, up_f32, half_n);
              Mul(gate_f32, gate_f32, up_f32, half_n);

              // Write activated float → bf16 output (automatic conversion)
              DataCopy(y_gm_[row_base], gate_f32, cp);
          }
      } else {
          mm.template IterateAll<false>(y_gm_[mn_config.yBaseOffset + out_offset], 0);
      }
  }

  __aicore__ inline GlobalTensor<T> GetGlobalBufferW(uint32_t group_idx, uint32_t tail_n, GMMConfig& mn_config) {
      uint64_t w_offset = SetWOffset(tail_n, mn_config.k);
      GlobalTensor<T> weight_gm_local;
      weight_gm_local = weight_gm_[mn_config.wBaseOffset + w_offset];
      // P2: Remove unconditional CACHE_MODE_DISABLE for blockDimM==1.
      // In MoE, multiple cores access the same expert's weight simultaneously
      // for different M-dimension rows. L2 caching enables cross-core weight
      // reuse, reducing HBM MTE2 traffic (the dominant bottleneck at 91.8%).
      // Only disable L2 when the expert weight is so large (>=64K per block)
      // that caching would thrash L2 for other cores' data.
      if (mn_config.blockDimM == 1 && mn_config.k * mn_config.n >= (64UL * 1024UL)) {
        weight_gm_local.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
      }
      return weight_gm_local;
  }

  __aicore__ inline uint64_t SetWOffset(uint32_t tail_n, uint32_t k) {
    uint64_t w_offset = 0;
    if constexpr (formatWeight == CubeFormat::NZ && transposeW) {
        w_offset = tail_n * (UB_BLOCK_UNIT_SIZE / sizeof(T));  // 32: quant is 32, float16 is 16
    } else if constexpr (formatWeight == CubeFormat::NZ) {
        w_offset = tail_n * AlignUp(k, 16);  // 16: nz format last two dim size
    } else if constexpr (transposeW) {
        w_offset = tail_n * k;
    } else {
        w_offset = tail_n;
    }
    return w_offset;
  }
};



