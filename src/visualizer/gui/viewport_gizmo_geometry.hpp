/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <array>
#include <glm/glm.hpp>
#include <limits>

namespace lfs::vis::gui {

    struct ViewportGizmoMarker {
        int encoded_axis = -1;
        glm::vec2 screen_pos{0.0f};
        float radius = 0.0f;
        bool visible = false;
    };

    constexpr float VIEWPORT_GIZMO_HIT_RADIUS_SCALE = 2.5f;

    [[nodiscard]] inline int hitTestViewportGizmoMarkers(
        const std::array<ViewportGizmoMarker, 6>& markers,
        const glm::vec2& mouse_pos) {
        int closest_axis = -1;
        float closest_distance_sq = std::numeric_limits<float>::max();
        for (const auto& marker : markers) {
            if (!marker.visible) {
                continue;
            }
            const glm::vec2 delta = mouse_pos - marker.screen_pos;
            const float radius = marker.radius * VIEWPORT_GIZMO_HIT_RADIUS_SCALE;
            const float distance_sq = glm::dot(delta, delta);
            // Expanded hit regions overlap; choose the circle nearest the pointer.
            if (distance_sq <= radius * radius && distance_sq < closest_distance_sq) {
                closest_axis = marker.encoded_axis;
                closest_distance_sq = distance_sq;
            }
        }
        return closest_axis;
    }

    [[nodiscard]] inline bool viewportGizmoFits(const float width, const float height, const float scale) {
        // Reserve room beside the two-column tool rail and below the axis widget
        // for its home/focus controls. Hidden widgets must not capture input.
        const float dpi = std::max(scale, 1.0f);
        return width >= 200.0f * dpi && height >= 150.0f * dpi;
    }

} // namespace lfs::vis::gui
