/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_lfs::optimizer::kernels::adam {

    constexpr uint8_t kSignedMomentZeroPoint = 128;

    __device__ inline float dequant_signed_moment(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : (static_cast<int>(q) - static_cast<int>(kSignedMomentZeroPoint)) * scale;
    }

    __device__ inline float dequant_unsigned_moment(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : static_cast<float>(q) * scale;
    }

    __device__ inline uint8_t quantize_signed_moment(const float value, const float scale) {
        if (scale == 0.0f)
            return kSignedMomentZeroPoint;
        const int q = static_cast<int>(roundf(value / scale)) + static_cast<int>(kSignedMomentZeroPoint);
        return static_cast<uint8_t>(min(255, max(0, q)));
    }

    __device__ inline uint8_t quantize_unsigned_moment(const float value, const float scale) {
        if (scale == 0.0f)
            return 0;
        const int q = static_cast<int>(roundf(fmaxf(value, 0.0f) / scale));
        if (value > 0.0f && q == 0)
            return 1;
        return static_cast<uint8_t>(min(255, max(0, q)));
    }

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

    __global__ void adam_step_quantized_cu(
        float* param,
        uint8_t* exp_avg_q,
        float* exp_avg_scale,
        uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const float* param_grad,
        const int n_rows,
        const int row_size,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp) {
        const int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= n_rows || row_size <= 0)
            return;

        const int base = row * row_size;
        const float old_m_scale = exp_avg_scale[row];
        const float old_v_scale = exp_avg_sq_scale[row];
        float max_abs_m = 0.0f;
        float max_v = 0.0f;

        for (int i = 0; i < row_size; ++i) {
            const int idx = base + i;
            const float grad = param_grad[idx];
            const float old_m = dequant_signed_moment(exp_avg_q[idx], old_m_scale);
            const float old_v = dequant_unsigned_moment(exp_avg_sq_q[idx], old_v_scale);
            const float m = beta1 * old_m + (1.0f - beta1) * grad;
            const float v = beta2 * old_v + (1.0f - beta2) * grad * grad;
            max_abs_m = fmaxf(max_abs_m, fabsf(m));
            max_v = fmaxf(max_v, v);
        }

        const float new_m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float new_v_scale = max_v > 0.0f ? max_v / 255.0f : 0.0f;
        const float step_size = lr * bias_correction1_rcp;

        for (int i = 0; i < row_size; ++i) {
            const int idx = base + i;
            const float grad = param_grad[idx];
            const float old_m = dequant_signed_moment(exp_avg_q[idx], old_m_scale);
            const float old_v = dequant_unsigned_moment(exp_avg_sq_q[idx], old_v_scale);
            const float m = beta1 * old_m + (1.0f - beta1) * grad;
            const float v = beta2 * old_v + (1.0f - beta2) * grad * grad;
            const float denom = sqrtf(v) * bias_correction2_sqrt_rcp + eps;
            param[idx] -= step_size * m / denom;
            exp_avg_q[idx] = quantize_signed_moment(m, new_m_scale);
            exp_avg_sq_q[idx] = quantize_unsigned_moment(v, new_v_scale);
        }

        exp_avg_scale[row] = new_m_scale;
        exp_avg_sq_scale[row] = new_v_scale;
    }

    __global__ void quantize_adam_moments_cu(
        const float* exp_avg,
        const float* exp_avg_sq,
        uint8_t* exp_avg_q,
        float* exp_avg_scale,
        uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const int n_rows,
        const int row_size) {
        const int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= n_rows || row_size <= 0)
            return;

        const int base = row * row_size;
        float max_abs_m = 0.0f;
        float max_v = 0.0f;
        for (int i = 0; i < row_size; ++i) {
            max_abs_m = fmaxf(max_abs_m, fabsf(exp_avg[base + i]));
            max_v = fmaxf(max_v, fmaxf(exp_avg_sq[base + i], 0.0f));
        }

        const float m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float v_scale = max_v > 0.0f ? max_v / 255.0f : 0.0f;
        for (int i = 0; i < row_size; ++i) {
            exp_avg_q[base + i] = quantize_signed_moment(exp_avg[base + i], m_scale);
            exp_avg_sq_q[base + i] = quantize_unsigned_moment(exp_avg_sq[base + i], v_scale);
        }
        exp_avg_scale[row] = m_scale;
        exp_avg_sq_scale[row] = v_scale;
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

    __global__ void zero_quantized_rows_cu(
        uint8_t* tensor_q,
        float* scales,
        const int64_t* indices,
        const int n_indices,
        const int row_size,
        const uint8_t zero_point) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_indices)
            return;

        const int64_t row_idx = indices[idx];
        const int64_t row_start = row_idx * static_cast<int64_t>(row_size);
        for (int i = 0; i < row_size; i++) {
            tensor_q[row_start + i] = zero_point;
        }
        scales[row_idx] = 0.0f;
    }

} // namespace fast_lfs::optimizer::kernels::adam
