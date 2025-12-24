/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include <functional>

namespace fast_lfs::rasterization {

    void backward(
        const float* grad_image,
        const float* grad_alpha,
        const float* image,
        const float* alpha,
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* raw_opacities,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        char* per_primitive_buffers_blob,
        char* per_tile_buffers_blob,
        char* per_instance_buffers_blob,
        char* per_bucket_buffers_blob,
        float3* grad_means,
        float3* grad_scales_raw,
        float4* grad_rotations_raw,
        float* grad_opacities_raw,
        float3* grad_sh_coefficients_0,
        float3* grad_sh_coefficients_rest,
        float2* grad_mean2d_helper,
        float* grad_conic_helper,
        float4* grad_w2c,
        float* densification_info,
        const int n_primitives,
        const int n_visible_primitives,
        const int n_instances,
        const int n_buckets,
        const int primitive_primitive_indices_selector,
        const int instance_primitive_indices_selector,
        const int active_sh_bases,
        const int total_bases_sh_rest,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        bool mip_filter);

}
