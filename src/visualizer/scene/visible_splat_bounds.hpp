/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "rendering/coordinate_conventions.hpp"
#include <algorithm>
#include <limits>

namespace lfs::vis {
    // Bounds of the visible splat nodes' original geometry, in the world space
    // used by Gaussian projection. Leaves `lo` and `hi` alone when no visible
    // node has bounds.
    inline bool visibleSplatBounds(const core::Scene& scene, glm::vec3& lo, glm::vec3& hi) {
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();
        bool found = false;
        for (const auto& slot : scene.getVisibleSplatNodeSlots()) {
            if (!slot.node || !scene.isNodeEffectivelyVisible(slot.node->id))
                continue;
            glm::vec3 node_lo, node_hi;
            if (!scene.getNodeRevealBounds(slot.node->id, node_lo, node_hi))
                continue;
            const auto transform = rendering::dataWorldTransformToVisualizerWorld(scene.getWorldTransform(slot.node->id));
            for (int i = 0; i < 8; ++i) {
                const glm::vec3 corner{(i & 1) ? node_hi.x : node_lo.x,
                                       (i & 2) ? node_hi.y : node_lo.y,
                                       (i & 4) ? node_hi.z : node_lo.z};
                const auto world = transform * glm::vec4(corner, 1.0f);
                min_x = std::min(min_x, world.x);
                max_x = std::max(max_x, world.x);
                min_y = std::min(min_y, world.y);
                max_y = std::max(max_y, world.y);
                min_z = std::min(min_z, world.z);
                max_z = std::max(max_z, world.z);
                found = true;
            }
        }
        if (!found)
            return false;
        lo = {min_x, min_y, min_z};
        hi = {max_x, max_y, max_z};
        return true;
    }
} // namespace lfs::vis
