/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

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

    // Batched zero operation for MCMC relocation (much faster than CPU loop)
    void zero_rows_at_indices(
        float* tensor,
        const int64_t* indices_device, // Must be on device!
        const int n_indices,
        const int row_size);

} // namespace fast_lfs::optimizer
