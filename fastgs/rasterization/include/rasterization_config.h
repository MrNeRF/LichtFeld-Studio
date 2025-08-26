#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace fast_gs::rasterization::config {
    DEF bool debug = false;
    // rendering constants
    DEF float dilation = 0.3f;
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float max_fragment_alpha = 0.999f;                          // 0.99f in original 3dgs
    DEF float transmittance_threshold = 1e-4f;
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
} // namespace fast_gs::rasterization::config

namespace config = fast_gs::rasterization::config;

#undef DEF
