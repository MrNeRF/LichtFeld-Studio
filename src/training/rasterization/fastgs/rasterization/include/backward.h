/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fused_adam_types.h"
#include "helper_math.h"
#include "rasterization_config.h"
#include <cuda_runtime.h>
#include <functional>

namespace fast_lfs::rasterization {

    void backward(
        const float* densification_error_map,
        const float* grad_image,
        const float* grad_alpha,
        const float* grad_depth,
        const float* grad_normal, // [3*H*W] or nullptr
        const float* image,
        const float* alpha,
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* raw_opacities,
        const float4* sh_coefficients_rest, // compact float4-packed swizzled layout
        const float4* w2c,
        const float3* cam_position,
        const float3* primitive_normals, // [N] or nullptr, required when grad_normal != nullptr
        char* per_primitive_buffers_blob,
        char* per_tile_buffers_blob,
        const uint* sorted_primitive_indices,
        float* grad_opacity_helper,
        float3* grad_color_helper,
        float2* grad_mean2d_helper,
        float* grad_conic_helper,
        float* grad_depth_helper,
        float3* grad_normal_helper, // [N] or nullptr, required when grad_normal != nullptr
        float4* grad_w2c,
        float* densification_info,
        const int n_primitives,
        const int n_instances,
        const int active_sh_bases,
        const int sh_layout_bases,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        bool mip_filter,
        DensificationType densification_type = DensificationType::None,
        FusedAdamSettings fused_adam = {},
        cudaStream_t stream = nullptr);

}
