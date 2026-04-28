/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace lfs::rendering::config {
    DEF bool debug = false;
    DEF float dilation = 0.3f;                  // Standard dilation when mip_filter OFF
    DEF float dilation_mip_filter = 0.1f;       // Smaller dilation when mip_filter ON
    DEF float max_stddev = 2.8284271247461903f; // sqrt(8)
    DEF float max_power_threshold = 4.0f;       // 0.5 * max_stddev^2
    DEF float min_alpha_threshold = 0.5f / 255.0f;
    DEF float min_alpha_threshold_rcp = 1.0f / min_alpha_threshold;
    DEF float max_fragment_alpha = 0.999f;
    DEF float transmittance_threshold = 1e-4f;
    DEF float max_pixel_radius = 512.0f;
    DEF float max_raw_scale = 20.0f;  // exp(40) stays finite in float, with margin.
    DEF float max_blend_color = 4.0f; // SH output is typically below 2.0.
    DEF float clip_xy = 1.4f;
    DEF int block_size_preprocess = 128;
    DEF int block_size_preprocess_backward = 128;
    DEF int block_size_apply_depth_ordering = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int tile_width = 16;
    DEF int tile_height = 16;
    DEF int block_size_blend = tile_width * tile_height;

    // Selection group colors (0 = center marker, 1-255 = groups). Defined in forward.cu.
    constexpr int MAX_SELECTION_GROUPS = 256;

    void setSelectionGroupColor(int group_id, float3 color);
    void setSelectionPreviewColor(float3 color);
} // namespace lfs::rendering::config

namespace config = lfs::rendering::config;

#undef DEF
