/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <algorithm>
#include <memory>
#include <ranges>
namespace collision_test {
    using namespace lfs::core;
    inline auto cells(glm::ivec3 dimensions) {
        return std::views::iota(0, dimensions.x * dimensions.y * dimensions.z) |
               std::views::transform([dimensions](int i) {
                   return glm::ivec3(i % dimensions.x, (i / dimensions.x) % dimensions.y, i / (dimensions.x * dimensions.y));
               });
    }
    inline std::shared_ptr<const CollisionVolume> box(glm::vec3 origin, glm::ivec3 dimensions, float size = 1) {
        auto volume = std::make_shared<CollisionVolume>(CollisionVolume{
            .settings = {.voxel_size = size},
            .origin = origin,
            .dimensions = dimensions,
            .edge = 4,
            .occupied_count = uint64_t(dimensions.x) * dimensions.y * dimensions.z,
            .words = {1}});
        while (volume->edge < std::max({dimensions.x, dimensions.y, dimensions.z}))
            volume->edge *= 2;
        volume->validate();
        return volume;
    }
} // namespace collision_test
