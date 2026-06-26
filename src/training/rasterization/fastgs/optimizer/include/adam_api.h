/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fast_lfs::optimizer {

    // Pure CUDA interface - no torch dependencies
    void adam_step_raw(
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
        const float bias_correction2_sqrt_rcp,
        cudaStream_t stream = nullptr);

    // Quantized step over contiguous [n_rows, row_size] moments (non-shN).
    void adam_step_quantized_raw(
        float* param,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const float* param_grad,
        const bool* frozen_mask,
        int frozen_mask_size,
        const int n_rows,
        const int row_size,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp,
        cudaStream_t stream = nullptr);

    // Quantized step over swizzled shN moments (1 thread per primitive).
    void adam_step_quantized_swizzled_raw(
        float* param,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const float* param_grad,
        const bool* frozen_mask,
        int frozen_mask_size,
        const int n_primitives,
        const int slots_per_primitive,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp,
        cudaStream_t stream = nullptr);

    // Quantise existing float moments into uint8 (legacy checkpoint load / set_state).
    void quantize_adam_moments_raw(
        const float* exp_avg,
        const float* exp_avg_sq,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const int n_rows,
        const int row_size,
        cudaStream_t stream = nullptr);

    // Swizzled (shN) variant of quantize_adam_moments_raw.
    void quantize_adam_moments_swizzled_raw(
        const float* exp_avg,
        const float* exp_avg_sq,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const int n_primitives,
        const int slots_per_primitive,
        cudaStream_t stream = nullptr);

    // Batched zero operation for MCMC relocation (much faster than CPU loop)
    void zero_rows_at_indices(
        float* tensor,
        const int64_t* indices_device, // Must be on device!
        const int n_indices,
        const int row_size,
        cudaStream_t stream = nullptr);

    // Reset quantised rows: moment bytes -> zero_point, scale -> 0. Contiguous layout.
    void zero_quantized_rows_at_indices(
        std::uint8_t* tensor_q,
        float* scales,
        const int64_t* indices_device, // Must be on device!
        const int n_indices,
        const int row_size,
        const std::uint8_t zero_point,
        cudaStream_t stream = nullptr);

} // namespace fast_lfs::optimizer
