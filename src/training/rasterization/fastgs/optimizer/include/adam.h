/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fast_lfs::optimizer {

    void adam_step(
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
    void adam_step_quantized(
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
    void adam_step_quantized_swizzled(
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

} // namespace fast_lfs::optimizer
