/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/checked_arithmetic.hpp"
#include "core/collision.hpp"
namespace lfs::core::cuda {
    struct CollisionBrick {
        static constexpr uint64_t key_mask = (uint64_t(1) << 57) - 1;
        uint64_t morton, mask;
    };
    inline size_t collisionOutputBytes(size_t count, size_t additional = 0) {
        const auto total = saturating_add(count, additional);
        if (total > std::vector<CollisionBrick>().max_size())
            throw std::length_error("Collision output exceeds the addressable record range");
        return checked_product(total, sizeof(CollisionBrick), "Collision output");
    }
    LFS_CORE_API size_t collisionSortWorkspaceBytes(size_t count);
    size_t queryCollisionSortWorkspaceBytes(size_t count);
    struct CollisionBricks {
        CollisionVolume volume;
        std::vector<CollisionBrick> bricks;
    };
    CollisionBricks voxelizeCollision(std::span<const CollisionSource>, CollisionSettings, std::stop_token, ProgressCallback, const std::function<void(float)>&);
} // namespace lfs::core::cuda
