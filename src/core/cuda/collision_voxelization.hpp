/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "collision_voxelization_kernels.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/collision.hpp"
#include <format>
namespace lfs::core::cuda {
    inline constexpr size_t max_collision_records = 64'000'000;
    inline size_t collisionOutputBytes(size_t count, size_t additional = 0) {
        const auto total = saturating_add(count, additional);
        if (total > std::vector<CollisionBrick>().max_size())
            throw std::length_error("Collision output exceeds the addressable record range");
        const auto bytes = checked_product(total, sizeof(CollisionBrick), "Collision output");
        if (total > max_collision_records)
            throw std::length_error(std::format("Collision output record limit exceeded: {} records ({} bytes), limit {}. Choose a larger cell size or Automatic resolution; no output was coarsened or cropped", total, bytes, max_collision_records));
        return bytes;
    }
    LFS_CORE_API size_t collisionSortWorkspaceBytes(size_t count);
    struct CollisionBricks {
        CollisionVolume volume;
        std::vector<CollisionBrick> bricks;
    };
    CollisionBricks voxelizeCollision(const CollisionSource&, CollisionSettings, std::stop_token, ProgressCallback, const std::function<void(float)>&, const std::function<void(uint64_t)>&);
} // namespace lfs::core::cuda
