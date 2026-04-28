/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace edge_compute::rasterization {

    struct ForwardResult {
        uint64_t frame_id;
        bool success;
        const char* error_message;
    };

    ForwardResult edge_forward_raw(
        const float* means_ptr,         // Device pointer [N*3]
        const float* scales_raw_ptr,    // Device pointer [N*3]
        const float* rotations_raw_ptr, // Device pointer [N*4]
        const float* opacities_raw_ptr, // Device pointer [N]
        const float* w2c_ptr,           // Device pointer [4*4]
        int n_primitives,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        bool mip_filter,
        const float* pixel_weights,
        float* accum_weights);

    // Pre-compile all CUDA kernels to avoid JIT delays during rendering
    void warmup_kernels();

} // namespace edge_compute::rasterization
