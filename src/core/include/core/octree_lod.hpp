/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_simplify_types.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {

    class SplatData;

    struct OctreeLodBuildOptions {
        // Octree subdivision stops at this many splats; the group then pairs
        // up into binary sub-levels like every other cell. Clamped to [2, 64].
        uint32_t leaf_group_splats = 8;
        // The bottom-up octree pass stops at the deepest depth cut whose
        // surviving representatives fit this budget; build_bhatt_lod then
        // assembles the tree top from those representatives with globally
        // similarity-ordered merges, so the visually dominant coarse levels
        // avoid compounded cell-forced pairings. 0 keeps the pure octree
        // tree all the way to the root.
        uint32_t bhatt_top_nodes = 32768;
        // Pruning aggressiveness of the bhatt top (BhattLodBuildOptions::lod_base).
        float bhatt_lod_base = 1.25f;
        // Input opacity_raw already holds display-space alpha (Spark lodOpacity
        // encoding, can exceed 1.0). Skips the sigmoid activation.
        bool input_lod_opacity = false;
        // When set, receives one entry per output node: the visible-order input
        // splat index for leaves, or UINT32_MAX for merged interior nodes.
        std::vector<uint32_t>* leaf_input_indices = nullptr;
    };

    // Build a hierarchical LOD tree from a Morton-ordered parallel octree with
    // moment-matched interior representatives. Inside each octree cell the
    // children pair up by Bhattacharyya similarity into binary sub-levels
    // (8 -> 4 -> 2 -> 1), so the fine levels form a strict binary tree whose
    // populations step ~2x -- granularity the pixel-threshold LOD selector
    // needs. By default (bhatt_top_nodes > 0) the bottom-up pass hands its
    // surviving representatives to build_bhatt_lod, which builds the coarse
    // levels with similarity-ordered merges instead of compounding
    // cell-forced pairings; the stitched output is emitted as one tree in
    // BFS level order. Output contract matches build_bhatt_lod: node 0 is
    // the root, children are contiguous and follow their parent, every input
    // splat survives as a leaf bit-exactly, interior alpha is
    // integrated-alpha-conserving lodOpacity (may exceed 1), and interior
    // nodes carry weight-blended SH1-3. Orders of magnitude faster than the
    // agglomerative Bhattacharyya builder on multi-million-splat buckets
    // because the fine levels carry nearly all merges and parallelize.
    LFS_CORE_API std::expected<std::unique_ptr<SplatData>, std::string> build_octree_lod(
        const SplatData& input,
        const OctreeLodBuildOptions& options = {},
        SplatSimplifyProgressCallback progress = {});

} // namespace lfs::core
