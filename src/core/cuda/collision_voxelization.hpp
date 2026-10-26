/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "collision_voxelization_kernels.hpp"
#include "core/collision.hpp"
namespace lfs::core::cuda {
    inline constexpr size_t max_collision_records = 64'000'000;
    LFS_CORE_API size_t collisionOutputBytes(size_t count, size_t additional = 0);
    struct CollisionBricks {
        CollisionVolume volume;
        std::vector<CollisionBrick> bricks;
    };
    CollisionBricks voxelizeCollision(const CollisionSource&, CollisionSettings, std::stop_token, ProgressCallback, const std::function<void(float)>&, const std::function<void(uint64_t)>&);
} // namespace lfs::core::cuda
