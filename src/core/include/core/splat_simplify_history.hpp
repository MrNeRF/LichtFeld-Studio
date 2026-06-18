/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_simplify.hpp"
#include "core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace lfs::core {

    struct SplatSimplifyMergeTree {
        Tensor source_means;
        Tensor source_sh0;
        Tensor source_shN;
        Tensor source_scaling;
        Tensor source_rotation;
        Tensor source_opacity;

        int source_active_sh_degree = 0;
        int source_max_sh_degree = 0;
        float source_scene_scale = 1.0f;

        int target_count = 0;
        int post_prune_count = 0;
        double requested_ratio = 0.5;
        float requested_lod_base = 2.0f;
        float requested_opacity_prune_threshold = 0.1f;

        std::vector<int32_t> final_roots;
        std::vector<int32_t> pruned_leaf_ids;
        std::vector<int32_t> merge_left;
        std::vector<int32_t> merge_right;
        std::vector<int32_t> merge_pass;

        [[nodiscard]] int leaf_count() const {
            return source_means.is_valid() ? static_cast<int>(source_means.size(0)) : 0;
        }

        [[nodiscard]] int merge_count() const {
            return static_cast<int>(merge_left.size());
        }
    };

    struct SplatSimplifyResult {
        std::unique_ptr<SplatData> splat;
        SplatSimplifyMergeTree merge_tree;
    };

    LFS_CORE_API std::expected<SplatSimplifyResult, std::string> simplify_splats_with_history(
        const SplatData& input,
        const SplatSimplifyOptions& options = {},
        SplatSimplifyProgressCallback progress = {});

} // namespace lfs::core
