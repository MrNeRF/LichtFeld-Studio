/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef> // Added for size_t
#include <cstdint>
#include <tuple>

namespace edge_compute::rasterization {

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

    ForwardContext edge_forward_raw(
        const float* means_ptr,         // Device pointer [N*3]
        const float* scales_raw_ptr,    // Device pointer [N*3]
        const float* rotations_raw_ptr, // Device pointer [N*4]
        const float* opacities_raw_ptr, // Device pointer [N]
        const float* w2c_ptr,           // Device pointer [4*4]
        const float* cam_position_ptr,  // Device pointer [3]
        float* alpha_ptr,               // Device pointer [H*W]
        int n_primitives,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        const float* pixel_weights,
        float* accum_weights);

    // Pre-compile all CUDA kernels to avoid JIT delays during rendering
    void warmup_kernels();

} // namespace edge_compute::rasterization
