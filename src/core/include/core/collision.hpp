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
    LFS_CORE_API glm::mat4 collisionInverse(const glm::mat4& transform);
    LFS_CORE_API CollisionSource captureCollisionSource(const SplatData&, const glm::mat4& transform = glm::mat4(1), size_t begin = 0, size_t count = SIZE_MAX);
    LFS_CORE_API CollisionVolume generateCollision(const CollisionSource&, CollisionSettings,
                                                   std::stop_token = {}, ProgressCallback = {}, std::function<void(float)> resolution = {},
                                                   std::function<void(uint64_t)> mixed_blocks = {});
    struct LFS_CORE_API WalkSettings {
        float height = 1.7f, radius = .25f, step = .3f, speed = 2;
        void validate() const;
    };
    struct WalkState {
        // Feet retain the tested body position; camera rounding never feeds back into collision.
        glm::vec3 eye{0}, presentation{0}, feet{0};
        bool grounded = false;
    };
    using CapsuleQuery = std::function<bool(glm::vec3 feet, glm::vec3 up, float height, float radius)>;
    LFS_CORE_API bool collisionCapsuleIntersects(const CollisionVolume&, const glm::mat4& transform,
                                                 glm::vec3 feet, glm::vec3 up, float height, float radius);
    LFS_CORE_API void settleWalk(WalkState&, const WalkSettings&, glm::vec3 up, const CapsuleQuery&);
    LFS_CORE_API void advanceWalk(WalkState&, const WalkSettings&, glm::vec3 up, glm::vec3 velocity,
                                  float dt, const CapsuleQuery&);
} // namespace lfs::core
