/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace fast_lfs::rasterization::config {
    DEF bool debug = false;
    // rendering constants
    DEF float dilation = 0.3f;            // Standard dilation when mip_filter OFF
    DEF float dilation_mip_filter = 0.1f; // Smaller dilation when mip_filter ON
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float max_fragment_alpha = 0.999f;                          // 0.99f in original 3dgs
    DEF float transmittance_threshold = 1e-4f;
    DEF float max_raw_scale = 20.0f; // exp(40) ≈ 2.35e17, safe margin before overflow
    // block size constants
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
    // checkpoint interval for gradient computation (how often to save intermediate blending state)
    // higher values = less memory, more recomputation in backward pass
    // 32: baseline (2.6 GB), 64: 1.3 GB, 128: 650 MB, 256: 325 MB
    // NOTE: Currently only 32 is supported. Larger values require redesigning bucket/checkpoint separation.
    DEF int checkpoint_interval = 32;
} // namespace fast_lfs::rasterization::config

namespace config = fast_lfs::rasterization::config;

#undef DEF
