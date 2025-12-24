/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace lfs::rendering::config {
    DEF bool debug = false;
    DEF float dilation = 0.3f;            // Standard dilation when mip_filter OFF
    DEF float dilation_mip_filter = 0.1f; // Smaller dilation when mip_filter ON
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp;
    DEF float max_fragment_alpha = 0.999f;
    DEF float transmittance_threshold = 1e-4f;
    DEF int block_size_preprocess = 128;
    DEF int block_size_preprocess_backward = 128;
    DEF int block_size_apply_depth_ordering = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int block_size_extract_bucket_counts = 256;
    DEF int tile_width = 16;
    DEF int tile_height = 16;
    DEF int block_size_blend = tile_width * tile_height;
    DEF int n_sequential_threshold = 4;

    // Selection group colors (0 = center marker, 1-255 = groups). Defined in forward.cu.
    constexpr int MAX_SELECTION_GROUPS = 256;

    void setSelectionGroupColor(int group_id, float3 color);
    void setSelectionPreviewColor(float3 color);
} // namespace lfs::rendering::config

namespace config = lfs::rendering::config;

#undef DEF
