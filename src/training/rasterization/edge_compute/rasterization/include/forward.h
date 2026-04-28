/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include <functional>

namespace edge_compute::rasterization {

    int edge_forward(
        std::function<char*(size_t)> per_primitive_buffers_func,
        std::function<char*(size_t)> per_tile_buffers_func,
        std::function<char*(size_t)> per_instance_buffers_func,
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* opacities_raw,
        const float4* w2c,
        const int n_primitives,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near,
        const float far,
        bool mip_filter,
        const float* pixel_weights,
        float* accum_weights);

}
