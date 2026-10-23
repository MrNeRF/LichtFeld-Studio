/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "collision_tree.hpp"
#include "core/collision.hpp"

namespace lfs::core::collision_detail {
    struct ClusterSelection {
        std::vector<uint32_t> indices;
        glm::vec3 resolved_seed;
        uint64_t occupied_voxels = 0;
    };
    ClusterSelection selectCluster(std::span<const CollisionGaussian> source, const CollisionVolume& coarse,
                                   glm::vec3 seed, float min_contribution, std::stop_token stop = {},
                                   CollisionProgress progress = {});

    CollisionVolume voxelizeSparse(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                   std::stop_token stop, CollisionProgress progress = {}, unsigned parallelism = 0);
    CollisionVolume processSparse(CollisionVolume volume, std::stop_token stop, CollisionProgress progress);
    inline Tree volumeTree(const CollisionVolume& volume, std::stop_token stop = {}) {
        Tree tree(volume.dimensions, stop);
        if (!volume.tree.empty()) {
            tree.nodes = volume.tree;
        } else {
            volume.forEachSolidBox([&](auto lo, auto hi) { tree.setBox(lo, hi); });
            tree.compact();
        }
        return tree;
    }
    inline void storeTree(CollisionVolume& volume, Tree tree) {
        tree.compact();
        volume.tree = std::move(tree.nodes);
        volume.tree_size = tree.size;
        volume.solid.clear();
    }
} // namespace lfs::core::collision_detail
