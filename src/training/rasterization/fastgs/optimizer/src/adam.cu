/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "optimizer_config.h"
#include "utils.h"

// Forward declare the kernels (defined in adam_api.cu)
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
}

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

    // Always check for errors to ensure kernel launched successfully
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("adam_step_cu kernel launch failed: ") + cudaGetErrorString(err));
    }

    CHECK_CUDA(config::debug, "adam step")
}
