/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

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
        const float bias_correction2_sqrt_rcp);

    void adam_step_quantized_raw(
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
        const float bias_correction2_sqrt_rcp);

    void quantize_adam_moments_raw(
        const float* exp_avg,
        const float* exp_avg_sq,
        uint8_t* exp_avg_q,
        float* exp_avg_scale,
        uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const int n_rows,
        const int row_size);

    // Batched zero operation for MCMC relocation (much faster than CPU loop)
    void zero_rows_at_indices(
        float* tensor,
        const int64_t* indices_device, // Must be on device!
        const int n_indices,
        const int row_size);

    void zero_quantized_rows_at_indices(
        uint8_t* tensor_q,
        float* scales,
        const int64_t* indices_device,
        const int n_indices,
        const int row_size,
        const uint8_t zero_point);

} // namespace fast_lfs::optimizer
