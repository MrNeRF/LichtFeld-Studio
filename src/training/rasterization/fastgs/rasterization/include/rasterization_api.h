/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fused_adam_types.h"
#include "rasterization_config.h"
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
        void* per_primitive_buffers = nullptr;
        void* per_tile_buffers = nullptr;
        void* sorted_primitive_indices = nullptr;
        size_t per_primitive_buffers_size = 0;
        size_t per_tile_buffers_size = 0;
        size_t sorted_primitive_indices_size = 0;
        size_t per_instance_sort_scratch_size = 0;
        size_t per_instance_sort_total_size = 0;
        int n_instances = 0;
        uint64_t frame_id = 0;
        // Add helper buffer pointers to avoid re-allocation in backward
        void* grad_mean2d_helper = nullptr;
        void* grad_conic_helper = nullptr;
        void* grad_opacity_helper = nullptr;
        void* grad_color_helper = nullptr;
        // Error handling for OOM
        bool success = false;
        const char* error_message = nullptr;
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

    void release_forward_context(const ForwardContext& forward_ctx);

    struct BackwardOutputs {
        bool success;
        const char* error_message;
    };

    BackwardOutputs backward_raw(
        float* densification_info_ptr,            // Device pointer [2*N] or nullptr
        const float* densification_error_map_ptr, // Device pointer [H*W] or nullptr
        const float* grad_image_ptr,              // Device pointer [3*H*W]
        const float* grad_alpha_ptr,              // Device pointer [H*W]
        const float* image_ptr,                   // Device pointer [3*H*W]
        const float* alpha_ptr,                   // Device pointer [H*W]
        const float* means_ptr,                   // Device pointer [N*3]
        const float* scales_raw_ptr,              // Device pointer [N*3]
        const float* rotations_raw_ptr,           // Device pointer [N*4]
        const float* raw_opacities_ptr,           // Device pointer [N]
        const float* sh_coefficients_rest_ptr,    // Device pointer [N*total_bases_sh_rest*3]
        const float* w2c_ptr,                     // Device pointer [4*4]
        const float* cam_position_ptr,            // Device pointer [3]
        const ForwardContext& forward_ctx,
        float* grad_w2c_ptr, // Device pointer [4*4] - output or nullptr
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        bool mip_filter,
        DensificationType densification_type,
        const FusedAdamSettings* fused_adam);

    // Pre-compile all CUDA kernels to avoid JIT delays during rendering
    void warmup_kernels();

} // namespace fast_lfs::rasterization
