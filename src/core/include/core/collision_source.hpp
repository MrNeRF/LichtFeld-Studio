/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include "core/scene.hpp"

namespace lfs::core {
    // Consumes owned CPU snapshots only, never live model or SH/optimizer data.
    // Stable node/row order and deletion filtering match scene collision authoring.
    [[nodiscard]] LFS_CORE_API std::vector<CollisionGaussian> extractCollisionSource(
        std::span<const Scene::SplatGeometrySnapshot> snapshots, const glm::mat4& world_to_grid,
        std::stop_token stop = {}, unsigned parallelism = 8);
} // namespace lfs::core
