/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"

#define DEF inline constexpr

enum class DensificationType : int { None = 0,
                                     MCMC = 1,
                                     MRNF = 2 };

namespace fast_lfs::rasterization::config {
    // rendering constants
    DEF float dilation = 0.3f;            // Standard dilation when mip_filter OFF
    DEF float dilation_mip_filter = 0.1f; // Smaller dilation when mip_filter ON
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float max_fragment_alpha = 0.999f;                          // 0.99f in original 3dgs
    DEF float min_cov2d_determinant = 1e-6f;
    DEF float transmittance_threshold = 1e-4f;
    DEF float max_raw_scale = 20.0f;  // exp(40) ≈ 2.35e17, safe margin before overflow
    DEF float max_blend_color = 4.0f; // SH output is typically < 2.0
    // block size constants
    DEF int block_size_preprocess = 128;
    DEF int block_size_preprocess_backward = 128;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int block_size_adam_step_invisible = 256;
    DEF int tile_width = 16;
    DEF int tile_height = 16;
    DEF int block_size_blend = tile_width * tile_height;
    DEF int block_size_blend_backward = 128;

    // SH coefficient swizzle (vksplat float4 layout): 32 primitives per block, 12 float4 slots
    // per primitive packing 15 float3 SH-rest coefficients with 3 floats of tail padding.
    DEF unsigned int sh_reorder_size = 32u;
    DEF unsigned int sh_rest_float4_per_primitive = 12u;
    DEF unsigned int sh_max_coeffs_rest = 15u;
} // namespace fast_lfs::rasterization::config

namespace config = fast_lfs::rasterization::config;

#undef DEF
