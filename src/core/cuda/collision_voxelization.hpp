/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <stop_token>
#include <vector>

namespace lfs::core::cuda {
    // Plain layout shared with PreparedGaussian; uploaded once without repacking.
    struct CollisionGaussianGpu {
        float center[3], inverse[9], extent[3], opacity;
        int min[3], max[3];
    };
    struct CollisionBvhNode {
        int min[3], max[3];
        // Unused traversal slots retain float bits for max(min(extent.xyz)):
        // interior: begin; leaf: left. This adds a support bound without growing
        // the descriptor or its GPU transfer/cache footprint.
        uint32_t left, right, begin, count;
    };
    struct CollisionBrick {
        // Low 57 bits: Morton key in 4-cell units. High bits: log2(edge/4)
        // for a proven uniform solid region (mask == UINT64_MAX).
        static constexpr uint64_t key_mask = (uint64_t(1) << 57) - 1;
        uint64_t morton, mask;
    };
    std::vector<CollisionBrick> voxelizeCollisionBricks(
        std::span<const CollisionGaussianGpu> gaussians, std::span<const CollisionBvhNode> nodes,
        int root_size, std::array<int, 3> dimensions, std::array<float, 3> origin,
        float voxel_size, double cutoff, std::stop_token stop,
        const std::function<void(float)>& progress);
} // namespace lfs::core::cuda
