/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef> // Added for size_t
#include <cstdint>
#include <tuple>

namespace fast_lfs::rasterization {

    struct FastGSSettings {
        const float* cam_position_ptr; // Device pointer [3]
        int active_sh_bases;
        int width;
        int height;
        float focal_x;
        float focal_y;
        float center_x;
        float center_y;
        float near_plane;
        float far_plane;
    };

    struct ForwardContext {
        void* per_primitive_buffers;
        void* per_tile_buffers;
        void* per_instance_buffers;
        void* per_bucket_buffers;
        size_t per_primitive_buffers_size;
        size_t per_tile_buffers_size;
        size_t per_instance_buffers_size;
        size_t per_bucket_buffers_size;
        int n_visible_primitives;
        int n_instances;
        int n_buckets;
        int primitive_primitive_indices_selector;
        int instance_primitive_indices_selector;
        uint64_t frame_id;
        // Add helper buffer pointers to avoid re-allocation in backward
        void* grad_mean2d_helper;
        void* grad_conic_helper;
        // Error handling for OOM
        bool success;
        const char* error_message;
    };

    ForwardContext forward_raw(
        const float* means_ptr,                // Device pointer [N*3]
        const float* scales_raw_ptr,           // Device pointer [N*3]
        const float* rotations_raw_ptr,        // Device pointer [N*4]
        const float* opacities_raw_ptr,        // Device pointer [N]
        const float* sh_coefficients_0_ptr,    // Device pointer [N*3]
        const float* sh_coefficients_rest_ptr, // Device pointer [N*total_bases_sh_rest*3]
        const float* w2c_ptr,                  // Device pointer [4*4]
        const float* cam_position_ptr,         // Device pointer [3]
        float* image_ptr,                      // Device pointer [3*H*W]
        float* alpha_ptr,                      // Device pointer [H*W]
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        bool mip_filter = false);

    struct BackwardOutputs {
        // These are filled in the provided pointers, not allocated
        bool success;
        const char* error_message;
    };

    BackwardOutputs backward_raw(
        float* densification_info_ptr,         // Device pointer [2*N] or nullptr
        const float* grad_image_ptr,           // Device pointer [3*H*W]
        const float* grad_alpha_ptr,           // Device pointer [H*W]
        const float* image_ptr,                // Device pointer [3*H*W]
        const float* alpha_ptr,                // Device pointer [H*W]
        const float* means_ptr,                // Device pointer [N*3]
        const float* scales_raw_ptr,           // Device pointer [N*3]
        const float* rotations_raw_ptr,        // Device pointer [N*4]
        const float* raw_opacities_ptr,        // Device pointer [N]
        const float* sh_coefficients_rest_ptr, // Device pointer [N*total_bases_sh_rest*3]
        const float* w2c_ptr,                  // Device pointer [4*4]
        const float* cam_position_ptr,         // Device pointer [3]
        const ForwardContext& forward_ctx,
        float* grad_means_ptr,                // Device pointer [N*3] - output
        float* grad_scales_raw_ptr,           // Device pointer [N*3] - output
        float* grad_rotations_raw_ptr,        // Device pointer [N*4] - output
        float* grad_opacities_raw_ptr,        // Device pointer [N] - output
        float* grad_sh_coefficients_0_ptr,    // Device pointer [N*3] - output
        float* grad_sh_coefficients_rest_ptr, // Device pointer [N*total_bases_sh_rest*3] - output
        float* grad_w2c_ptr,                  // Device pointer [4*4] - output or nullptr
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        bool mip_filter = false);

    // Pre-compile all CUDA kernels to avoid JIT delays during rendering
    void warmup_kernels();

} // namespace fast_lfs::rasterization