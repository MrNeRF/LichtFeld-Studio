/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "optimizer_config.h"
#include "utils.h"
#include <cstdint>

// Forward declare the kernels (defined in adam_api.cu's TU via adam_kernels.cuh).
namespace fast_lfs::optimizer::kernels::adam {
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
        const float bias_correction2_sqrt_rcp);

    __global__ void adam_step_quantized_cu(
        float* param,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
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
        const float bias_correction2_sqrt_rcp);

    __global__ void adam_step_quantized_swizzled_cu(
        float* param,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
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
        const float bias_correction2_sqrt_rcp);
} // namespace fast_lfs::optimizer::kernels::adam

void fast_lfs::optimizer::adam_step(
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
    cudaStream_t stream) {

    // IMPORTANT: Use the SAME kernel as legacy (adam_step_cu), NOT the vectorized version!
    // The vectorized kernel (adam_step_vectorized_cu) has different floating-point rounding
    // behavior which causes divergence from legacy implementation.
    kernels::adam::adam_step_cu<<<div_round_up(n_elements, config::block_size_adam_step), config::block_size_adam_step, 0, stream>>>(
        param,
        exp_avg,
        exp_avg_sq,
        param_grad,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp);

    LFS_FASTGS_PHASE_CHECK(config::debug, "adam step");
}

void fast_lfs::optimizer::adam_step_quantized(
    float* param,
    std::uint8_t* exp_avg_q,
    float* exp_avg_scale,
    std::uint8_t* exp_avg_sq_q,
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
    const float bias_correction2_sqrt_rcp,
    cudaStream_t stream) {

    kernels::adam::adam_step_quantized_cu<<<
        div_round_up(n_rows, config::block_size_adam_step), config::block_size_adam_step, 0, stream>>>(
        param, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale, param_grad,
        frozen_mask, frozen_mask_size, n_rows, row_size, lr, beta1, beta2, eps, bias_correction1_rcp, bias_correction2_sqrt_rcp);

    LFS_FASTGS_PHASE_CHECK(config::debug, "quantized adam step");
}

void fast_lfs::optimizer::adam_step_quantized_swizzled(
    float* param,
    std::uint8_t* exp_avg_q,
    float* exp_avg_scale,
    std::uint8_t* exp_avg_sq_q,
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
    const float bias_correction2_sqrt_rcp,
    cudaStream_t stream) {

    kernels::adam::adam_step_quantized_swizzled_cu<<<
        div_round_up(n_primitives, config::block_size_adam_step), config::block_size_adam_step, 0, stream>>>(
        param, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale, param_grad,
        frozen_mask, frozen_mask_size, n_primitives, slots_per_primitive, lr, beta1, beta2, eps, bias_correction1_rcp, bias_correction2_sqrt_rcp);

    LFS_FASTGS_PHASE_CHECK(config::debug, "quantized adam step (swizzled)");
}
