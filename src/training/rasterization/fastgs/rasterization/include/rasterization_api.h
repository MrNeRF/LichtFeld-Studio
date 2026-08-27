/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fused_adam_types.h"
#include "rasterization_config.h"
#include <cstddef> // Added for size_t
#include <cstdint>
#include <tuple>

// cstdint already provides std::uint64_t for preflight counters

namespace fast_lfs::rasterization {

    struct FastGSSettings {
        const float* cam_position_ptr; // Device pointer [3]
        int active_sh_bases;
        int sh_layout_bases;
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
        int sh_layout_bases = 1;
        uint64_t frame_id = 0;
        // The stream all of this context's kernels/allocations are ordered on;
        // releases (sorted indices, arena frame, helper buffers) must use it.
        cudaStream_t stream = nullptr;
        // Add helper buffer pointers to avoid re-allocation in backward
        void* grad_mean2d_helper = nullptr;
        void* grad_conic_helper = nullptr;
        void* grad_depth_helper = nullptr;
        void* grad_opacity_helper = nullptr;
        void* grad_color_helper = nullptr;
        void* primitive_normals = nullptr;
        // Error handling for OOM / pathological frames
        bool success = false;
        bool resource_exhausted = false;
        // 32-bit instance-count overflow (garbage extents). Soft
        // fail the step, do not kill the training run.
        bool instance_count_overflow = false;
        const char* error_message = nullptr;
    };

    ForwardContext forward_raw(
        const float* means_ptr,                // Device pointer [N*3]
        const float* scales_raw_ptr,           // Device pointer [N*3]
        const float* rotations_raw_ptr,        // Device pointer [N*4]
        const float* opacities_raw_ptr,        // Device pointer [N]
        const float* sh_coefficients_0_ptr,    // Device pointer [N*3]
        const float* sh_coefficients_rest_ptr, // Device pointer to swizzled shN (float or u16 bitcast)
        const float* w2c_ptr,                  // Device pointer [4*4]
        const float* cam_position_ptr,         // Device pointer [3]
        float* image_ptr,                      // Device pointer [3*H*W]
        float* alpha_ptr,                      // Device pointer [H*W]
        float* depth_ptr,                      // Device pointer [H*W]
        float* normal_ptr,                     // Device pointer [3*H*W] or nullptr — enables the normal render channel
        const float* bg_color_ptr,             // Device pointer [3] solid bg, or nullptr
        const float* bg_image_ptr,             // Device pointer [3*H*W] CHW, or nullptr
        int n_primitives,
        int active_sh_bases,
        int sh_layout_bases,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        bool mip_filter = false,
        cudaStream_t stream = nullptr,              // nullptr → getCurrentCUDAStream()
        const float* sh_value_bounds_ptr = nullptr, // float2 per 256; null = fp32/IEEE-f16 shN
        unsigned int sh_value_n_cells = 0,
        unsigned int sh_value_bits = 0); // 0=fp32, 16=q16(+bounds) or IEEE f16

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
        const float* grad_depth_ptr,              // Device pointer [H*W] or nullptr
        const float* grad_normal_ptr,             // Device pointer [3*H*W] or nullptr
        const float* image_ptr,                   // Device pointer [3*H*W]
        const float* alpha_ptr,                   // Device pointer [H*W]
        const float* means_ptr,                   // Device pointer [N*3]
        const float* scales_raw_ptr,              // Device pointer [N*3]
        const float* rotations_raw_ptr,           // Device pointer [N*4]
        const float* raw_opacities_ptr,           // Device pointer [N]
        const float* sh_coefficients_rest_ptr,    // Device pointer to swizzled shN float buffer
        const float* w2c_ptr,                     // Device pointer [4*4]
        const float* cam_position_ptr,            // Device pointer [3]
        const ForwardContext& forward_ctx,
        float* grad_w2c_ptr, // Device pointer [4*4] - output or nullptr
        int n_primitives,
        int active_sh_bases,
        int sh_layout_bases,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        bool mip_filter,
        DensificationType densification_type,
        const FusedAdamSettings* fused_adam,
        // shN representation binds, mirroring forward_raw. The backward
        // kernel decodes sh_coefficients_rest with THESE, never with the fused
        // Adam settings' copy — optimizer enablement (SH warmup) must not change
        // how the coefficient buffer is interpreted. bits==16 with bounds → q16
        // codes; bits==16 without bounds → IEEE f16 swizzle; bits==0 → fp32.
        const float* shN_value_bounds_ptr = nullptr,
        unsigned shN_value_n_cells = 0u,
        unsigned shN_value_bits = 0u,
        const bool* mean_step_far_mask = nullptr,
        int mean_step_far_mask_n = 0);

    // Pre-compile all CUDA kernels to avoid JIT delays during rendering
    void warmup_kernels();

    /// cudaPointerGetAttributes preflight call count (0 in Release/NDEBUG).
    [[nodiscard]] std::uint64_t preflight_pointer_attr_call_count() noexcept;
    void reset_preflight_pointer_attr_call_count() noexcept;

} // namespace fast_lfs::rasterization
