/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fast_lfs::optimizer {

    struct JointContiguousBatchEntry {
        float* param = nullptr;
        std::uint8_t* packed = nullptr;
        float* bounds = nullptr;
        const float* grad = nullptr;
        int n_prims = 0;
        int n_attr = 0;
        float lr = 0.0f;
        float bias_correction1_rcp = 1.0f;
        float bias_correction2_sqrt_rcp = 1.0f;
        // per-splat mean-step scaling applies to this entry (Means only)
        int apply_mean_step = 0;
    };

    // Pure CUDA interface - no torch dependencies

    // Non-fused joint (u,log_s) Adam step for contiguous [n_prims, n_attr] parameters.
    // Launches one 256-thread block per joint-bounds block (blockIdx.x is the bounds row).
    // bits is the joint codec cell width (8 or 16). Swizzled shN uses
    // adam_step_shN_joint_from_grad instead of this contiguous step.
    void adam_step_joint_contiguous_raw(
        float* param,
        std::uint8_t* packed,
        float* bounds,
        const float* param_grad,
        const bool* frozen_mask,
        int frozen_mask_size,
        float frozen_lr_scale,
        const bool* crop_damping_mask,
        int crop_damping_mask_size,
        float cropbox_lr_scale,
        int n_prims,
        int n_attr,
        int bits,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float bias_correction1_rcp,
        float bias_correction2_sqrt_rcp,
        cudaStream_t stream = nullptr,
        const float* mean_step_scale_raw = nullptr,
        int mean_step_scale_n = 0,
        float mean_step_median_extent = 0.0f,
        float mean_step_r_min = 1.0f,
        float mean_step_r_max = 300.0f,
        const bool* mean_step_far_mask = nullptr,
        int mean_step_far_mask_n = 0);

    // One launch over a device table of contiguous joint params (means/sh0/scale/rot/opa).
    void adam_step_joint_contiguous_batched(
        const JointContiguousBatchEntry* host_entries,
        int n_entries,
        const bool* frozen_mask,
        int frozen_mask_size,
        float frozen_lr_scale,
        const bool* crop_damping_mask,
        int crop_damping_mask_size,
        float cropbox_lr_scale,
        float beta1,
        float beta2,
        float eps,
        cudaStream_t stream = nullptr,
        const float* mean_step_scale_raw = nullptr,
        int mean_step_scale_n = 0,
        float mean_step_median_extent = 0.0f,
        float mean_step_r_min = 1.0f,
        float mean_step_r_max = 300.0f,
        const bool* mean_step_far_mask = nullptr,
        int mean_step_far_mask_n = 0);

    // Standalone joint 8-bit shN Adam over a swizzled fp32 gradient buffer.
    // Reuses apply_shN_grads_packed_joint (same device code as fused FastGS).
    // Grid is ceil(n_prims/256) blocks of 256 threads; blockIdx.x is the bounds row.
    // active_sh_bases is 4/9/16 for SH degree 1/2/3. sh_value_bits is 0 (fp32),
    // 16 with bounds (q16), or 16 without bounds (IEEE f16 swizzle).
    void adam_step_shN_joint_from_grad(
        float* param,
        std::uint8_t* packed,
        float* bounds,
        float* sh_value_bounds,
        const float* shN_grad,
        const bool* frozen_mask,
        int frozen_mask_size,
        float frozen_lr_scale,
        const bool* crop_damping_mask,
        int crop_damping_mask_size,
        float cropbox_lr_scale,
        int n_prims,
        int sh_layout_slots,
        int active_sh_bases,
        int sh_value_bits,
        int sh_value_n_cells,
        float step_size,
        float beta1,
        float beta2,
        float eps,
        float bias_correction2_sqrt_rcp,
        cudaStream_t stream = nullptr);

    // Joint (u,log_s) densify reset: encode true (m,v)=(0,0). Widens block
    // bounds to include 0 when needed and re-encodes the block.
    // Contiguous params: n_attr cells/row, bits 8 or 16. n_prims = live N.
    void joint_encode_zero_rows_at_indices(
        std::uint8_t* packed,
        float* bounds, // float4 per 256-splat block (mutable — may widen)
        const int64_t* indices_device,
        const int n_indices,
        const int n_attr,
        const int bits,
        const int n_prims,
        cudaStream_t stream = nullptr);

    // Same for swizzled shN: walks float4 slots via shAt layout.
    void joint_encode_zero_shN_at_indices(
        std::uint8_t* packed,
        float* bounds,
        const int64_t* indices_device,
        const int n_indices,
        const int slots_per_primitive,
        const int bits,
        const int n_prims,
        cudaStream_t stream = nullptr);

    // re-encode gathered joint rows under destination block bounds.
    void joint_transcode_gathered_rows_at_indices(
        std::uint8_t* packed,
        const float* bounds,
        const int64_t* indices_device,
        const int n_new,
        const int old_N,
        const int n_attr,
        const int bits,
        cudaStream_t stream = nullptr);

} // namespace fast_lfs::optimizer
