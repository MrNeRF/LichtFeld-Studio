/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "adam_api.h"
#include "adam_kernels.cuh"
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
        const float bias_correction2_sqrt_rcp,
        cudaStream_t stream) {

        // Validate pointers
        LFS_VALIDATE_CUDA_DEVICE_POINTER(param, "param");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg, "exp_avg");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq, "exp_avg_sq");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(param_grad, "param_grad");

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
            bias_correction2_sqrt_rcp,
            stream);
    }

    void adam_step_quantized_raw(
        float* param,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const float* param_grad,
        const bool* frozen_mask,
        const int frozen_mask_size,
        const float frozen_lr_scale,
        const int n_rows,
        const int row_size,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp,
        cudaStream_t stream) {

        LFS_VALIDATE_CUDA_DEVICE_POINTER(param, "param");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_q, "exp_avg_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_scale, "exp_avg_scale");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_q, "exp_avg_sq_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_scale, "exp_avg_sq_scale");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(param_grad, "param_grad");
        if (n_rows <= 0 || row_size <= 0) {
            throw std::runtime_error("n_rows and row_size must be positive");
        }

        adam_step_quantized(
            param, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale,
            param_grad, frozen_mask, frozen_mask_size, frozen_lr_scale, n_rows, row_size, lr,
            beta1, beta2, eps, bias_correction1_rcp, bias_correction2_sqrt_rcp, stream);
    }

    void adam_step_quantized_swizzled_raw(
        float* param,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const float* param_grad,
        const bool* frozen_mask,
        const int frozen_mask_size,
        const float frozen_lr_scale,
        const int n_primitives,
        const int slots_per_primitive,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp,
        cudaStream_t stream) {

        LFS_VALIDATE_CUDA_DEVICE_POINTER(param, "param");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_q, "exp_avg_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_scale, "exp_avg_scale");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_q, "exp_avg_sq_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_scale, "exp_avg_sq_scale");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(param_grad, "param_grad");
        if (n_primitives <= 0 || slots_per_primitive <= 0) {
            throw std::runtime_error("n_primitives and slots_per_primitive must be positive");
        }

        adam_step_quantized_swizzled(
            param, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale,
            param_grad, frozen_mask, frozen_mask_size, frozen_lr_scale, n_primitives,
            slots_per_primitive, lr, beta1, beta2, eps, bias_correction1_rcp,
            bias_correction2_sqrt_rcp, stream);
    }

    void quantize_adam_moments_raw(
        const float* exp_avg,
        const float* exp_avg_sq,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const int n_rows,
        const int row_size,
        cudaStream_t stream) {

        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg, "exp_avg");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq, "exp_avg_sq");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_q, "exp_avg_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_scale, "exp_avg_scale");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_q, "exp_avg_sq_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_scale, "exp_avg_sq_scale");
        if (n_rows <= 0 || row_size <= 0)
            return;

        kernels::adam::quantize_adam_moments_cu<<<div_round_up(n_rows, config::block_size_adam_step), config::block_size_adam_step, 0, stream>>>(
            exp_avg, exp_avg_sq, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale, n_rows, row_size);

        LFS_CUDA_LAUNCH_CHECK(stream, "quantize_adam_moments");
    }

    void quantize_adam_moments_swizzled_raw(
        const float* exp_avg,
        const float* exp_avg_sq,
        std::uint8_t* exp_avg_q,
        float* exp_avg_scale,
        std::uint8_t* exp_avg_sq_q,
        float* exp_avg_sq_scale,
        const int n_primitives,
        const int slots_per_primitive,
        cudaStream_t stream) {

        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg, "exp_avg");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq, "exp_avg_sq");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_q, "exp_avg_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_scale, "exp_avg_scale");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_q, "exp_avg_sq_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(exp_avg_sq_scale, "exp_avg_sq_scale");
        if (n_primitives <= 0 || slots_per_primitive <= 0)
            return;

        kernels::adam::quantize_adam_moments_swizzled_cu<<<div_round_up(n_primitives, config::block_size_adam_step), config::block_size_adam_step, 0, stream>>>(
            exp_avg, exp_avg_sq, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale, n_primitives, slots_per_primitive);

        LFS_CUDA_LAUNCH_CHECK(stream, "quantize_adam_moments_swizzled");
    }

    void zero_rows_at_indices(
        float* tensor,
        const int64_t* indices_device,
        const int n_indices,
        const int row_size,
        cudaStream_t stream) {

        // Validate pointers
        LFS_VALIDATE_CUDA_DEVICE_POINTER(tensor, "tensor");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(indices_device, "indices_device");

        // Validate parameters
        if (n_indices <= 0)
            return; // Nothing to do
        if (row_size <= 0) {
            throw std::runtime_error("row_size must be positive");
        }

        // Launch kernel: one thread per index
        kernels::adam::zero_rows_cu<<<div_round_up(n_indices, config::block_size_adam_step), config::block_size_adam_step, 0, stream>>>(
            tensor,
            indices_device,
            n_indices,
            row_size);

        LFS_CUDA_LAUNCH_CHECK(stream, "zero_rows_at_indices");
    }

    void zero_quantized_rows_at_indices(
        std::uint8_t* tensor_q,
        float* scales,
        const int64_t* indices_device,
        const int n_indices,
        const int row_size,
        const std::uint8_t zero_point,
        cudaStream_t stream) {

        LFS_VALIDATE_CUDA_DEVICE_POINTER(tensor_q, "tensor_q");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(scales, "scales");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(indices_device, "indices_device");
        if (n_indices <= 0)
            return;
        if (row_size <= 0) {
            throw std::runtime_error("row_size must be positive");
        }

        kernels::adam::zero_quantized_rows_cu<<<div_round_up(n_indices, config::block_size_adam_step), config::block_size_adam_step, 0, stream>>>(
            tensor_q, scales, indices_device, n_indices, row_size, zero_point);

        LFS_CUDA_LAUNCH_CHECK(stream, "zero_quantized_rows_at_indices");
    }

} // namespace fast_lfs::optimizer
