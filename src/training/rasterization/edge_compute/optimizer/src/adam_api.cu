/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "adam_api.h"
#include "adam_kernels.cuh"
#include "cuda_utils.h"
#include "optimizer_config.h"
#include "utils.h"

namespace fast_lfs::optimizer {

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
        const float bias_correction2_sqrt_rcp) {

        // Validate pointers
        CHECK_CUDA_PTR(param, "param");
        CHECK_CUDA_PTR(exp_avg, "exp_avg");
        CHECK_CUDA_PTR(exp_avg_sq, "exp_avg_sq");
        CHECK_CUDA_PTR(param_grad, "param_grad");

        // Validate parameters
        if (n_elements <= 0) {
            throw std::runtime_error("n_elements must be positive");
        }

        // Call the actual implementation
        adam_step(
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
    }

    void zero_rows_at_indices(
        float* tensor,
        const int64_t* indices_device,
        const int n_indices,
        const int row_size) {

        // Validate pointers
        CHECK_CUDA_PTR(tensor, "tensor");
        CHECK_CUDA_PTR(indices_device, "indices_device");

        // Validate parameters
        if (n_indices <= 0)
            return; // Nothing to do
        if (row_size <= 0) {
            throw std::runtime_error("row_size must be positive");
        }

        // Launch kernel: one thread per index
        kernels::adam::zero_rows_cu<<<div_round_up(n_indices, config::block_size_adam_step), config::block_size_adam_step>>>(
            tensor,
            indices_device,
            n_indices,
            row_size);

        CHECK_CUDA(config::debug, "zero_rows_at_indices")
    }

} // namespace fast_lfs::optimizer
