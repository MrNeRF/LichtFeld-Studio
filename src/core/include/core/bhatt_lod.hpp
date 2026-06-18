/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_simplify_types.hpp"

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {

    class SplatData;

    struct BhattLodWorkset {
        // Flat SoA arrays for all splats (including interior nodes)
        // Pre-allocated to capacity = 1.5 * initial_count
        std::vector<float> center_x, center_y, center_z;
        std::vector<float> scale_x, scale_y, scale_z;
        std::vector<float> qw, qx, qy, qz;
        std::vector<float> opacity;
        std::vector<float> r, g, b;
        std::vector<float> sh1; // flat [N, 9]
        std::vector<float> sh2; // flat [N, 15]
        std::vector<float> sh3; // flat [N, 21]
        std::vector<float> feature_size;
        std::vector<float> area;

        // Cached covariance matrices (symmetric 3x3 + det) for fast similarity
        std::vector<float> cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz;
        std::vector<float> cov_det;

        // Binary tree: for each node, store up to 2 child indices
        // child_a[i] = first child, child_b[i] = second child (or -1 if none)
        std::vector<int32_t> child_a;
        std::vector<int32_t> child_b;

        // Active mask during construction
        std::vector<uint8_t> is_active;

        int max_sh_degree = 0;
        size_t initial_count = 0;
        size_t capacity = 0;
        size_t current_count = 0;

        // Reserve capacity
        void reserve(size_t initial_count, int max_sh_degree);

        // Add a new node from raw data, return its index
        size_t add_node(
            float cx, float cy, float cz,
            float sx, float sy, float sz,
            float qw_, float qx_, float qy_, float qz_,
            float op,
            float r_, float g_, float b_,
            const float* sh1_ptr, const float* sh2_ptr, const float* sh3_ptr);

        // Merge two existing nodes and return the new node index
        size_t merge_nodes(size_t a, size_t b, float filter_size);

        // Feature size for a node: 2 * max_scale * lod_opacity
        float compute_feature_size(size_t idx) const;

        // Area: ellipsoid surface area
        float compute_area(size_t idx) const;

        // Grid cell for spatial hashing
        void grid_cell(size_t idx, float step, int64_t& gx, int64_t& gy, int64_t& gz) const;

        // Bhattacharyya similarity between two nodes
        float similarity(size_t a, size_t b) const;
    };

    struct BhattLodBuildOptions {
        // Pruning aggressiveness (SparkJS default ~1.25).
        float lod_base = 1.25f;
        // Input opacity_raw already holds display-space alpha (Spark lodOpacity
        // encoding, can exceed 1.0), e.g. when re-clustering previously merged
        // LOD nodes. Skips the sigmoid activation in workset extraction.
        bool input_lod_opacity = false;
        // When set, receives one entry per output node: the visible-order input
        // splat index for leaves, or UINT32_MAX for merged interior nodes.
        std::vector<uint32_t>* leaf_input_indices = nullptr;
    };

    // Build a hierarchical LOD tree using Bhattacharyya distance.
    // Returns a SplatData with a populated SplatLodTree (binary tree).
    // lod_base: controls pruning aggressiveness (SparkJS default ~1.25)
    LFS_CORE_API std::expected<std::unique_ptr<SplatData>, std::string> build_bhatt_lod(
        const SplatData& input,
        float lod_base = 1.25f,
        SplatSimplifyProgressCallback progress = {});

    LFS_CORE_API std::expected<std::unique_ptr<SplatData>, std::string> build_bhatt_lod(
        const SplatData& input,
        const BhattLodBuildOptions& options,
        SplatSimplifyProgressCallback progress = {});

} // namespace lfs::core
