/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Faithful CPU port of LichtFeld-Studio's shipped LOD method, for quality
// comparison against the lod_splat scheme on machines where the app itself
// cannot run (CUDA/Vulkan).
//
// Build: src/core/bhatt_lod.cpp's algorithm — greedy Bhattacharyya-similarity
// agglomerative clustering into a binary merge tree (level-doubling hash grid,
// smallest-feature-first, 8-random-neighbor search), area*alpha weighted
// merges with unclamped lodOpacity, then lod_base post-order pruning into a
// flat variable-arity tree. Formulas and constants are kept verbatim; only
// the SplatData/tensor I/O boundary is replaced with SplatCloud.
//
// Select: the per-node logic of lod_select_threshold.slang — pixel_scale =
// size/distance with viewport-edge foveation (gaze cone disabled: no eye
// tracking in this comparison), parent-above-limit cut test, and the
// continuous [1.0, 1.18] transition band. Their runtime has no splat budget
// (output_capacity truncates arbitrarily), so for budget-matched comparisons
// bhattLimitForBudget() binary-searches pixel_scale_limit to a target count —
// the steady state their CPU feedback loop would reach.
#pragma once

#include "camera.hpp"
#include "hierarchy.hpp" // kInvalidNode
#include "splat.hpp"

#include <cstdint>
#include <vector>

namespace lfs::lod {

    struct BhattRefTree {
        // node i renders splat i of `splats` (lodOpacity linear alpha)
        SplatCloud splats;
        std::vector<Vec3> centers;
        std::vector<float> sizes; // 2 * expansion * max_scale (Spark expansion for alpha > 1)
        std::vector<uint32_t> child_start;
        std::vector<uint16_t> child_count;
        std::vector<uint32_t> parent;
        std::vector<uint8_t> level;
        double build_ms = 0;
        size_t input_count = 0, total_merges = 0, output_count = 0;
    };

    void buildBhattRef(const SplatCloud& input, float lod_base, BhattRefTree& out);

    struct BhattSelection {
        std::vector<uint32_t> nodes;
        std::vector<float> weights;
    };

    // One pass of the selection shader logic at a given pixel_scale_limit.
    void bhattSelect(const BhattRefTree& t, const Camera& cam, float pixel_scale_limit,
                     BhattSelection& out);

    // Binary-search the limit whose selection count best matches `budget`.
    float bhattLimitForBudget(const BhattRefTree& t, const Camera& cam, uint64_t budget);

    // Gather a selection into a renderable cloud (weights premultiplied into the
    // linear lodOpacity alphas).
    void bhattGather(const BhattRefTree& t, const BhattSelection& sel, SplatCloud& out);

} // namespace lfs::lod
