/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_lfs::optimizer::kernels::adam {

    // Quantized Adam moment helpers.
    //   m  (first moment, signed):  symmetric int8 around zero-point 128, scale = max|m| / 127.
    //   v  (second moment, >=0):    stored as quantized sqrt(v), scale = sqrt(max v) / 255.
    // sqrt-domain quantisation of v spends the 8-bit grid where 1/(sqrt(v)+eps) is sensitive
    // (the small-v tail), ~16x finer there than quantising v linearly.
    constexpr uint8_t kMomentSignedZeroPoint = 128;

    __device__ inline float dequant_m(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : (static_cast<int>(q) - static_cast<int>(kMomentSignedZeroPoint)) * scale;
    }

    __device__ inline uint8_t quantize_m(const float value, const float scale) {
        if (scale == 0.0f)
            return kMomentSignedZeroPoint;
        const int q = static_cast<int>(roundf(value / scale)) + static_cast<int>(kMomentSignedZeroPoint);
        return static_cast<uint8_t>(min(255, max(0, q)));
    }

    // Returns sqrt(v) from the quantised second moment.
    __device__ inline float dequant_sqrt_v(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : static_cast<float>(q) * scale;
    }

    // Takes v, stores quantised sqrt(v). `scale` is the sqrt-domain scale (sqrt(max v) / 255).
    __device__ inline uint8_t quantize_sqrt_v(const float v, const float scale) {
        if (scale == 0.0f)
            return 0;
        const float s = sqrtf(fmaxf(v, 0.0f));
        const int q = static_cast<int>(roundf(s / scale));
        if (v > 0.0f && q == 0)
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

    // Quantized Adam step over contiguous [n_rows, row_size] moments with per-row scales.
    // Two-pass: pass 1 derives the new per-row scales from max|m| / max(v); pass 2 writes
    // the param update and requantises. Used by the standalone (non-fused) fallback path
    // for all non-shN params (shN uses the swizzle-aware variant below).
    __global__ void adam_step_quantized_cu(
        float* param,
        uint8_t* exp_avg_q,
        float* exp_avg_scale,
        uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const float* param_grad,
        const bool* frozen_mask,
        const int frozen_mask_size,
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
        if (frozen_mask != nullptr && row < frozen_mask_size && frozen_mask[row])
            return;

        const int base = row * row_size;
        const float old_m_scale = exp_avg_scale[row];
        const float old_v_scale = exp_avg_sq_scale[row];
        const float beta1_comp = 1.0f - beta1;
        const float beta2_comp = 1.0f - beta2;
        const float step_size = lr * bias_correction1_rcp;

        float max_abs_m = 0.0f;
        float max_v = 0.0f;
        for (int i = 0; i < row_size; ++i) {
            const float grad = param_grad[base + i];
            const float old_m = dequant_m(exp_avg_q[base + i], old_m_scale);
            const float sqrt_old_v = dequant_sqrt_v(exp_avg_sq_q[base + i], old_v_scale);
            const float old_v = sqrt_old_v * sqrt_old_v;
            const float m = beta1 * old_m + beta1_comp * grad;
            const float v = beta2 * old_v + beta2_comp * grad * grad;
            max_abs_m = fmaxf(max_abs_m, fabsf(m));
            max_v = fmaxf(max_v, v);
        }

        const float new_m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float new_v_scale = max_v > 0.0f ? sqrtf(max_v) / 255.0f : 0.0f;

        for (int i = 0; i < row_size; ++i) {
            const float grad = param_grad[base + i];
            const float old_m = dequant_m(exp_avg_q[base + i], old_m_scale);
            const float sqrt_old_v = dequant_sqrt_v(exp_avg_sq_q[base + i], old_v_scale);
            const float old_v = sqrt_old_v * sqrt_old_v;
            const float m = beta1 * old_m + beta1_comp * grad;
            const float v = beta2 * old_v + beta2_comp * grad * grad;
            const float denom = sqrtf(v) * bias_correction2_sqrt_rcp + eps;
            param[base + i] -= step_size * m / denom;
            exp_avg_q[base + i] = quantize_m(m, new_m_scale);
            exp_avg_sq_q[base + i] = quantize_sqrt_v(v, new_v_scale);
        }

        exp_avg_scale[row] = new_m_scale;
        exp_avg_sq_scale[row] = new_v_scale;
    }

    // Quantise existing float moments (exp_avg, exp_avg_sq) into the uint8 representation.
    // Used on legacy (v1) checkpoint load and set_state. Contiguous [n_rows, row_size].
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
        const float v_scale = max_v > 0.0f ? sqrtf(max_v) / 255.0f : 0.0f;
        for (int i = 0; i < row_size; ++i) {
            exp_avg_q[base + i] = quantize_m(exp_avg[base + i], m_scale);
            exp_avg_sq_q[base + i] = quantize_sqrt_v(exp_avg_sq[base + i], v_scale);
        }
        exp_avg_scale[row] = m_scale;
        exp_avg_sq_scale[row] = v_scale;
    }

    // Swizzled SH layout index (mirror of lfs::core::sh_swizzled_index / rasterization shAt).
    // Returns the float4 slot index; multiply by 4 for the float offset.
    __device__ __forceinline__ unsigned int sh_swizzled_slot(
        const unsigned int primitive_idx,
        const unsigned int float4_slot,
        const unsigned int slots_per_primitive) {
        constexpr unsigned int R = 32u;
        const unsigned int block = primitive_idx / R;
        const unsigned int lane = primitive_idx % R;
        return block * (slots_per_primitive * R) + float4_slot * R + lane;
    }

    // Quantized Adam step over swizzled shN moments (1 thread per primitive, two-pass).
    // param/grad are float4 swizzled; moments are uchar4 swizzled at the same slots; scales
    // are per-primitive. Mirrors adam_step_quantized_cu but walks the strided swizzle slots.
    __global__ void adam_step_quantized_swizzled_cu(
        float* param,
        uint8_t* exp_avg_q,
        float* exp_avg_scale,
        uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const float* param_grad,
        const bool* frozen_mask,
        const int frozen_mask_size,
        const int n_primitives,
        const int slots_per_primitive,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp) {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= n_primitives || slots_per_primitive <= 0)
            return;
        if (frozen_mask != nullptr && p < frozen_mask_size && frozen_mask[p])
            return;

        float4* param4 = reinterpret_cast<float4*>(param);
        const float4* grad4 = reinterpret_cast<const float4*>(param_grad);
        uchar4* m4 = reinterpret_cast<uchar4*>(exp_avg_q);
        uchar4* v4 = reinterpret_cast<uchar4*>(exp_avg_sq_q);

        const float old_m_scale = exp_avg_scale[p];
        const float old_v_scale = exp_avg_sq_scale[p];
        const float beta1_comp = 1.0f - beta1;
        const float beta2_comp = 1.0f - beta2;
        const float step_size = lr * bias_correction1_rcp;

        float max_abs_m = 0.0f;
        float max_v = 0.0f;
        for (int k = 0; k < slots_per_primitive; ++k) {
            const unsigned int slot = sh_swizzled_slot(p, k, slots_per_primitive);
            const float4 g = grad4[slot];
            const uchar4 qm = m4[slot];
            const uchar4 qv = v4[slot];
            const float gc[4] = {g.x, g.y, g.z, g.w};
            const uint8_t mc[4] = {qm.x, qm.y, qm.z, qm.w};
            const uint8_t vc[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float sqrt_old_v = dequant_sqrt_v(vc[c], old_v_scale);
                const float old_v = sqrt_old_v * sqrt_old_v;
                const float m = beta1 * dequant_m(mc[c], old_m_scale) + beta1_comp * gc[c];
                const float v = beta2 * old_v + beta2_comp * gc[c] * gc[c];
                max_abs_m = fmaxf(max_abs_m, fabsf(m));
                max_v = fmaxf(max_v, v);
            }
        }

        const float new_m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float new_v_scale = max_v > 0.0f ? sqrtf(max_v) / 255.0f : 0.0f;

        for (int k = 0; k < slots_per_primitive; ++k) {
            const unsigned int slot = sh_swizzled_slot(p, k, slots_per_primitive);
            const float4 g = grad4[slot];
            const uchar4 qm = m4[slot];
            const uchar4 qv = v4[slot];
            const float gc[4] = {g.x, g.y, g.z, g.w};
            const uint8_t mc[4] = {qm.x, qm.y, qm.z, qm.w};
            const uint8_t vc[4] = {qv.x, qv.y, qv.z, qv.w};
            float4 p4 = param4[slot];
            float pc[4] = {p4.x, p4.y, p4.z, p4.w};
            uint8_t nm[4];
            uint8_t nv[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float sqrt_old_v = dequant_sqrt_v(vc[c], old_v_scale);
                const float old_v = sqrt_old_v * sqrt_old_v;
                const float m = beta1 * dequant_m(mc[c], old_m_scale) + beta1_comp * gc[c];
                const float v = beta2 * old_v + beta2_comp * gc[c] * gc[c];
                const float denom = sqrtf(v) * bias_correction2_sqrt_rcp + eps;
                pc[c] -= step_size * m / denom;
                nm[c] = quantize_m(m, new_m_scale);
                nv[c] = quantize_sqrt_v(v, new_v_scale);
            }
            param4[slot] = make_float4(pc[0], pc[1], pc[2], pc[3]);
            m4[slot] = make_uchar4(nm[0], nm[1], nm[2], nm[3]);
            v4[slot] = make_uchar4(nv[0], nv[1], nv[2], nv[3]);
        }

        exp_avg_scale[p] = new_m_scale;
        exp_avg_sq_scale[p] = new_v_scale;
    }

    // Quantise existing swizzled float shN moments into the uint8 swizzled representation
    // (v1 checkpoint load / set_state for shN). 1 thread per primitive.
    __global__ void quantize_adam_moments_swizzled_cu(
        const float* exp_avg,
        const float* exp_avg_sq,
        uint8_t* exp_avg_q,
        float* exp_avg_scale,
        uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const int n_primitives,
        const int slots_per_primitive) {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= n_primitives || slots_per_primitive <= 0)
            return;

        const float4* m_src = reinterpret_cast<const float4*>(exp_avg);
        const float4* v_src = reinterpret_cast<const float4*>(exp_avg_sq);
        uchar4* m4 = reinterpret_cast<uchar4*>(exp_avg_q);
        uchar4* v4 = reinterpret_cast<uchar4*>(exp_avg_sq_q);

        float max_abs_m = 0.0f;
        float max_v = 0.0f;
        for (int k = 0; k < slots_per_primitive; ++k) {
            const unsigned int slot = sh_swizzled_slot(p, k, slots_per_primitive);
            const float4 m = m_src[slot];
            const float4 v = v_src[slot];
            max_abs_m = fmaxf(max_abs_m, fmaxf(fmaxf(fabsf(m.x), fabsf(m.y)), fmaxf(fabsf(m.z), fabsf(m.w))));
            max_v = fmaxf(max_v, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
        max_v = fmaxf(max_v, 0.0f);
        const float m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float v_scale = max_v > 0.0f ? sqrtf(max_v) / 255.0f : 0.0f;

        for (int k = 0; k < slots_per_primitive; ++k) {
            const unsigned int slot = sh_swizzled_slot(p, k, slots_per_primitive);
            const float4 m = m_src[slot];
            const float4 v = v_src[slot];
            m4[slot] = make_uchar4(quantize_m(m.x, m_scale), quantize_m(m.y, m_scale),
                                   quantize_m(m.z, m_scale), quantize_m(m.w, m_scale));
            v4[slot] = make_uchar4(quantize_sqrt_v(v.x, v_scale), quantize_sqrt_v(v.y, v_scale),
                                   quantize_sqrt_v(v.z, v_scale), quantize_sqrt_v(v.w, v_scale));
        }
        exp_avg_scale[p] = m_scale;
        exp_avg_sq_scale[p] = v_scale;
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

    // Reset quantised rows to the "zero moment" state: bytes -> zero_point, scale -> 0.
    // (m uses zero_point 128, v uses 0; both dequantise to 0 with scale 0.)
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
        for (int i = 0; i < row_size; ++i) {
            tensor_q[row_start + i] = zero_point;
        }
        scales[row_idx] = 0.0f;
    }

} // namespace fast_lfs::optimizer::kernels::adam
