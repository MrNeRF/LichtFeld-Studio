/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace fast_lfs::optimizer::kernels::adam {

    // Vectorized Adam kernel using float4 for better memory throughput
    __global__ void adam_step_vectorized_cu(
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        const float* param_grad,
        const int n_elements,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Early exit if beyond range
        if (idx * 4 >= n_elements)
            return;

        const float beta1_comp = 1.0f - beta1;
        const float beta2_comp = 1.0f - beta2;
        const float step_size = lr * bias_correction1_rcp;

        const int base_idx = idx * 4;
        const int remaining = n_elements - base_idx;

        // Process up to 4 elements per thread
        if (remaining >= 4) {
            // Vectorized path: load/store 4 elements at once (128-bit transactions)
            float4 grad4 = *reinterpret_cast<const float4*>(param_grad + base_idx);
            float4 m1_4 = *reinterpret_cast<float4*>(exp_avg + base_idx);
            float4 m2_4 = *reinterpret_cast<float4*>(exp_avg_sq + base_idx);
            float4 p4 = *reinterpret_cast<float4*>(param + base_idx);

#pragma unroll
            for (int i = 0; i < 4; i++) {
                float grad = reinterpret_cast<float*>(&grad4)[i];
                float m1 = reinterpret_cast<float*>(&m1_4)[i];
                float m2 = reinterpret_cast<float*>(&m2_4)[i];
                float p = reinterpret_cast<float*>(&p4)[i];

                m1 = beta1 * m1 + beta1_comp * grad;
                m2 = beta2 * m2 + beta2_comp * grad * grad;
                p -= step_size * m1 / (sqrtf(m2) * bias_correction2_sqrt_rcp + eps);

                reinterpret_cast<float*>(&m1_4)[i] = m1;
                reinterpret_cast<float*>(&m2_4)[i] = m2;
                reinterpret_cast<float*>(&p4)[i] = p;
            }

            *reinterpret_cast<float4*>(exp_avg + base_idx) = m1_4;
            *reinterpret_cast<float4*>(exp_avg_sq + base_idx) = m2_4;
            *reinterpret_cast<float4*>(param + base_idx) = p4;
        } else {
// Scalar path for tail elements (1-3 remaining elements)
#pragma unroll
            for (int i = 0; i < remaining; i++) {
                const int elem_idx = base_idx + i;
                const float grad = param_grad[elem_idx];
                const float m1 = beta1 * exp_avg[elem_idx] + beta1_comp * grad;
                const float m2 = beta2 * exp_avg_sq[elem_idx] + beta2_comp * grad * grad;
                param[elem_idx] -= step_size * m1 / (sqrtf(m2) * bias_correction2_sqrt_rcp + eps);
                exp_avg[elem_idx] = m1;
                exp_avg_sq[elem_idx] = m2;
            }
        }
    }

    // Original scalar kernel (kept for compatibility)
    // based on https://github.com/pytorch/pytorch/blob/9d32aa9789fc0ef0cad01a788157ecc2121db810/torch/csrc/api/src/optim/adam.cpp#L72-L142
    __global__ void adam_step_cu(
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        const float* param_grad,
        const int n_elements,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp) {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= n_elements)
            return;
        const float grad = param_grad[idx];
        const float moment1 = beta1 * exp_avg[idx] + (1.0f - beta1) * grad;
        const float moment2 = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * grad * grad;
        const float denom = sqrtf(moment2) * bias_correction2_sqrt_rcp + eps;
        const float step_size = lr * bias_correction1_rcp;
        param[idx] -= step_size * moment1 / denom;
        exp_avg[idx] = moment1;
        exp_avg_sq[idx] = moment2;
    }

    // Batched kernel to zero out specific rows (for MCMC relocation)
    // Much faster than element-by-element indexing on CPU
    __global__ void zero_rows_cu(
        float* tensor,
        const int64_t* indices,
        const int n_indices,
        const int row_size) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_indices)
            return;

        const int64_t row_idx = indices[idx];
        const int64_t row_start = row_idx * static_cast<int64_t>(row_size);

// Zero out the entire row
#pragma unroll 4
        for (int i = 0; i < row_size; i++) {
            tensor[row_start + i] = 0.0f;
        }
    }

} // namespace fast_lfs::optimizer::kernels::adam
