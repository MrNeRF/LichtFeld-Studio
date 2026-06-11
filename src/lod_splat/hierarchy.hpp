/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Offline LOD hierarchy: an octree over Morton-ordered splats where each
// interior node stores merged representative splats sized to the node's scale.
//
// Merging is Gaussian moment matching: children are treated as a mixture with
// weights w_i = alpha_i * area_i; the merged mean/covariance are the mixture
// moments (covariance includes the spread between child means), opacity is
// chosen so the merged splat's integrated alpha (alpha * area) matches the
// cluster's total, and color is the alpha-weighted average. With octree
// branching 8 and a reduction of R=4 source splats per representative, splat
// extent roughly doubles per level (density quarters), so level selection maps
// to log2 of the pixel-footprint ratio.
#pragma once

#include "splat.hpp"
#include "util.hpp"

#include <cstdint>
#include <vector>

namespace lfs::lod {

    constexpr uint32_t kInvalidNode = 0xFFFFFFFFu;

    struct HierarchyParams {
        // Leaf size sets the cut granularity (the quantum a refine/coarsen flips).
        // Finer leaves approach the per-splat granularity of LichtFeld's binary
        // tree at no build cost (the merge work is unchanged, only node count
        // grows): on the 30M reference scene, 256 -> 64 gained +10.7 dB at a
        // budget-saturated street view. Below ~32 the selection cost grows
        // (cut size) with no measurable quality gain.
        uint32_t max_leaf_splats = 64;
        // Source splats merged per representative. For surface-like scenes the
        // octree branches ~4-way, so reduction=2 halves density per level (extent
        // grows by sqrt(2)), making level selection track log2 of the footprint
        // ratio; reduction=4 quarters density (extent doubles) and costs ~1/3 the
        // merged-level memory instead of ~1x.
        uint32_t reduction = 2;
        uint32_t max_rep_count = 65536; // safety cap per node payload (chunk size)
        int max_depth = 18;
        unsigned threads = std::thread::hardware_concurrency();
        // Ported from LichtFeld's Bhattacharyya LOD builder:
        // - similarity_pairing: choose merge partners by Bhattacharyya distance
        //   within a Morton window instead of blind run grouping (reduction=2
        //   only), so coplanar/similar splats merge before dissimilar neighbors.
        // - lod_opacity: merged splats store linear alpha that may exceed 1
        //   (Spark lodOpacity), conserving integrated alpha exactly instead of
        //   clamping at 1.
        bool similarity_pairing = true;
        bool lod_opacity = true;
        float max_lod_opacity = 4.f;
    };

    struct Node {
        Aabb bounds;
        uint32_t parent = kInvalidNode;
        uint32_t children[8];
        uint8_t child_count = 0;
        uint8_t depth = 0;
        bool is_leaf = false;
        uint32_t rep_offset = 0; // leaf: into Hierarchy::leaves; interior: into Hierarchy::merged
        uint32_t rep_count = 0;
        uint32_t subtree_splats = 0; // raw splats below this node
        float rep_extent = 0.f;      // mean 2*sigma_max of this node's representatives, world units
    };

    struct BuildStats {
        double morton_ms = 0, sort_ms = 0, reorder_ms = 0, topology_ms = 0, merge_ms = 0;
        double total_ms = 0;
        std::vector<uint32_t> nodes_per_depth;
        std::vector<uint64_t> reps_per_depth;
    };

    struct Hierarchy {
        std::vector<Node> nodes; // pre-order; nodes[0] is the root
        SplatCloud leaves;       // input splats, Morton-reordered
        SplatCloud merged;       // interior-node representatives
        Aabb scene_bounds;
        BuildStats stats;

        const SplatCloud& cloudFor(const Node& n) const { return n.is_leaf ? leaves : merged; }
    };

    // Consumes `input` (moved into Morton order inside the hierarchy).
    void buildHierarchy(SplatCloud&& input, const HierarchyParams& params, Hierarchy& out);

} // namespace lfs::lod
