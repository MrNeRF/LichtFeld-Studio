/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace lfs::training {

    // ============= FUSED CUDA KERNEL FOR ADMM BACKWARD PASS =============
    // Computes: grad_opacities = rho * (opa - z + u) * opa * (1 - opa) * grad_loss
    // Single kernel, zero intermediate allocations!
    // If accumulate=true: grad_opacities += result
    // If accumulate=false: grad_opacities = result (overwrites)
    __global__ void admm_backward_fused_kernel(
        float* __restrict__ grad_opacities,    // Output: gradients [N]
        const float* __restrict__ opa_sigmoid, // Input: sigmoid(opacities) [N]
        const float* __restrict__ z,           // Input: ADMM auxiliary variable [N]
        const float* __restrict__ u,           // Input: ADMM dual variable [N]
        float rho,                             // ADMM penalty parameter
        float grad_loss,                       // Gradient from upstream
        size_t n,                              // Number of elements
        bool accumulate                        // Whether to accumulate or overwrite
    ) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        // Load values
        const float opa = opa_sigmoid[idx];
        const float z_val = z[idx];
        const float u_val = u[idx];

        // Compute: diff = opa - z + u
        const float diff = opa - z_val + u_val;

        // Compute: sigmoid_grad = opa * (1 - opa)
        const float sigmoid_grad = opa * (1.0f - opa);

        // Compute: grad = rho * diff * sigmoid_grad * grad_loss
        const float grad = rho * diff * sigmoid_grad * grad_loss;

        // Write output (accumulate or overwrite)
        if (accumulate) {
            grad_opacities[idx] += grad;
        } else {
            grad_opacities[idx] = grad;
        }
    }

    // Host wrapper function to launch the kernel
    void launch_admm_backward_fused(
        float* grad_opacities,
        const float* opa_sigmoid,
        const float* z,
        const float* u,
        float rho,
        float grad_loss,
        size_t n,
        bool accumulate) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;

        admm_backward_fused_kernel<<<blocks, threads>>>(
            grad_opacities, opa_sigmoid, z, u, rho, grad_loss, n, accumulate);
        LFS_CUDA_LAUNCH_CHECK(nullptr, "training.sparsity.admm_backward_fused");
    }

} // namespace lfs::training
