/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_api.h"
#include "fused_adam_types.h"
#include "kernel_utils.cuh"
#include "utils.h"

#include <stdexcept>

namespace fast_lfs::optimizer {

    namespace {

        template <int ACTIVE_SH_BASES>
        __global__ void __launch_bounds__(256, 3) adam_step_shN_joint_from_grad_cu(
            fast_lfs::rasterization::FusedAdamSettings fused_adam,
            const float* __restrict__ shN_grad,
            const unsigned int sh_layout_slots) {
            const unsigned int primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
            using fast_lfs::rasterization::kernels::apply_shN_grads_packed_joint;
            using fast_lfs::rasterization::kernels::ShNGradFromSwizzle;
            apply_shN_grads_packed_joint<ACTIVE_SH_BASES>(
                fused_adam,
                primitive_idx,
                sh_layout_slots,
                ShNGradFromSwizzle{reinterpret_cast<const float4*>(shN_grad)});
        }

    } // namespace

    void adam_step_shN_joint_from_grad(
        float* param,
        std::uint8_t* packed,
        float* bounds,
        float* sh_value_bounds,
        const float* shN_grad,
        const bool* frozen_mask,
        const int frozen_mask_size,
        const float frozen_lr_scale,
        const bool* crop_damping_mask,
        const int crop_damping_mask_size,
        const float cropbox_lr_scale,
        const int n_prims,
        const int sh_layout_slots,
        const int active_sh_bases,
        const int sh_value_bits,
        const int sh_value_n_cells,
        const float step_size,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction2_sqrt_rcp,
        cudaStream_t stream) {
        LFS_VALIDATE_CUDA_DEVICE_POINTER(param, "param");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(packed, "joint_packed");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(bounds, "joint_bounds");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(shN_grad, "shN_grad");
        if (n_prims <= 0 || sh_layout_slots <= 0) {
            return;
        }
        if (active_sh_bases != 4 && active_sh_bases != 9 && active_sh_bases != 16) {
            throw std::runtime_error(
                "adam_step_shN_joint_from_grad: active_sh_bases must be 4, 9, or 16");
        }
        if (sh_value_bits != 0 && sh_value_bits != 16) {
            throw std::runtime_error(
                "adam_step_shN_joint_from_grad: sh_value_bits must be 0 or 16");
        }

        fast_lfs::rasterization::FusedAdamSettings fused{};
        fused.enabled = true;
        fused.beta1 = beta1;
        fused.beta2 = beta2;
        fused.eps = eps;
        fused.shN.param = param;
        fused.shN.joint_packed = packed;
        fused.shN.joint_bounds = bounds;
        fused.shN.joint_bits = 8;
        fused.shN.sh_value_bounds = sh_value_bounds;
        fused.shN.sh_value_bits = sh_value_bits;
        fused.shN.sh_value_n_cells = sh_value_n_cells;
        fused.shN.n_primitives = n_prims;
        fused.shN.frozen_mask = frozen_mask;
        fused.shN.frozen_mask_size = frozen_mask_size;
        fused.shN.frozen_lr_scale = frozen_lr_scale;
        fused.shN.crop_damping_mask = crop_damping_mask;
        fused.shN.crop_damping_mask_size = crop_damping_mask_size;
        fused.shN.cropbox_lr_scale = cropbox_lr_scale;
        fused.shN.step_size = step_size;
        fused.shN.bias_correction2_sqrt_rcp = bias_correction2_sqrt_rcp;
        fused.shN.enabled = true;

        constexpr int kBS = 256;
        const int n_blocks = (n_prims + kBS - 1) / kBS;
        const auto slots = static_cast<unsigned int>(sh_layout_slots);
        if (active_sh_bases == 16) {
            adam_step_shN_joint_from_grad_cu<16><<<n_blocks, kBS, 0, stream>>>(
                fused, shN_grad, slots);
            LFS_CUDA_LAUNCH_CHECK(stream, "adam_step_shN_joint_from_grad<16>");
        } else if (active_sh_bases == 9) {
            adam_step_shN_joint_from_grad_cu<9><<<n_blocks, kBS, 0, stream>>>(
                fused, shN_grad, slots);
            LFS_CUDA_LAUNCH_CHECK(stream, "adam_step_shN_joint_from_grad<9>");
        } else {
            adam_step_shN_joint_from_grad_cu<4><<<n_blocks, kBS, 0, stream>>>(
                fused, shN_grad, slots);
            LFS_CUDA_LAUNCH_CHECK(stream, "adam_step_shN_joint_from_grad<4>");
        }
    }

} // namespace fast_lfs::optimizer
