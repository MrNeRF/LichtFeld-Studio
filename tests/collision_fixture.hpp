/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <algorithm>
#include <memory>
namespace collision_test {
    using namespace lfs::core;
    inline std::shared_ptr<const CollisionVolume> box(glm::vec3 origin, glm::ivec3 dimensions, float size = 1) {
        auto volume = std::make_shared<CollisionVolume>();
        volume->origin = origin;
        volume->dimensions = dimensions;
        volume->edge = 4;
        while (volume->edge < std::max({dimensions.x, dimensions.y, dimensions.z}))
            volume->edge *= 2;
        volume->settings.voxel_size = size;
        volume->words = {1};
        volume->occupied_count = uint64_t(dimensions.x) * dimensions.y * dimensions.z;
        volume->validate();
        return volume;
    }

} // namespace collision_test
