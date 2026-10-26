/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "sequencer/timeline.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace lfs::vis {
    // Bounds are calculated in the same world space used by Gaussian projection.
    inline std::optional<rendering::GaussianRevealState> makeRevealRenderState(
        const core::Scene& scene, const sequencer::RevealClip& clip, const float time) {
        if (time >= clip.end)
            return std::nullopt;
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
            glm::vec3 lo, hi;
            if (!scene.getNodeRevealBounds(slot.node->id, lo, hi))
                continue;
            const auto transform = rendering::dataWorldTransformToVisualizerWorld(scene.getWorldTransform(slot.node->id));
            for (int i = 0; i < 8; ++i) {
                const glm::vec3 corner{(i & 1) ? hi.x : lo.x,
                                       (i & 2) ? hi.y : lo.y,
                                       (i & 4) ? hi.z : lo.z};
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
            return std::nullopt;
        // The spread starts at the scene's world origin, independently of the
        // positions or transforms of the visible splats.
        const float center_x = 0.0f;
        const float center_z = 0.0f;
        const float radius = glm::length(glm::vec2(std::max(std::abs(min_x), std::abs(max_x)),
                                                   std::max(std::abs(min_z), std::abs(max_z))));
        return rendering::GaussianRevealState{clip.progress(time), clip.softness,
                                              center_x, center_z, radius, static_cast<int>(clip.effect), min_y, max_y};
    }
} // namespace lfs::vis
