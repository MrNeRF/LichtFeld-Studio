/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

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
        const float bias_correction2_sqrt_rcp);

    void adam_step_quantized(
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

}
