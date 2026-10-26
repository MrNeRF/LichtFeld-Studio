/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <functional>
namespace lfs::core {
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
