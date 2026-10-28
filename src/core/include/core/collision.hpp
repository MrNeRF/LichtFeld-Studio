/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision_volume.hpp"
#include "core/export.hpp"
#include "core/progress.hpp"
#include "core/tensor.hpp"
#include <glm/glm.hpp>
#include <vector>
namespace lfs::core {
    class SplatData;
    // Owned activated geometry; no appearance data or live scene objects.
    struct CollisionSource {
        Tensor means, scales, rotations, opacity, deleted;
        glm::mat4 transform{1};
    };
    LFS_CORE_API CollisionSource captureCollisionSource(const SplatData&, const glm::mat4& transform = glm::mat4(1));
    LFS_CORE_API CollisionVolume generateCollision(const CollisionSource&, CollisionSettings,
                                                   std::stop_token = {}, ProgressCallback = {}, std::function<void(float)> resolution = {},
                                                   std::function<void(uint64_t)> mixed_blocks = {});
} // namespace lfs::core
