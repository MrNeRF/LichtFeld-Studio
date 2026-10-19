/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/coordinate_conventions.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <optional>

namespace lfs::vis::gui {

    inline constexpr float VIEWPORT_AXIS_WIDGET_SIZE = 104.0f;
    inline constexpr float VIEWPORT_AXIS_WIDGET_MARGIN = 10.0f;
    inline constexpr float VIEWPORT_AXIS_MARKER_RADIUS = 0.18f;
    inline constexpr float VIEWPORT_AXIS_MARKER_DISTANCE = 0.73f;
    inline constexpr float VIEWPORT_AXIS_CAMERA_DISTANCE = 2.8f;
    inline constexpr float VIEWPORT_AXIS_FOV_DEGREES = 38.0f;
    // A hovered marker is drawn this much larger, and picking accepts its drawn edge.
    inline constexpr float VIEWPORT_AXIS_MARKER_HOVER_SCALE = 1.2f;
    // A press on a marker becomes an orbit once the pointer moves this far (dp).
    inline constexpr float VIEWPORT_AXIS_DRAG_THRESHOLD = 3.0f;

    struct ViewportGizmoMarker {
        int encoded_axis = -1;
        glm::vec2 screen_pos{0.0f};
        float radius = 0.0f;
        bool visible = false;
        // Normalized device depth; smaller is nearer the viewer.
        float depth = 0.0f;

        [[nodiscard]] int axis() const { return encoded_axis % 3; }
        [[nodiscard]] bool negative() const { return encoded_axis >= 3; }
    };

    struct ViewportGizmoLayout {
        glm::vec2 top_left{0.0f};
        glm::vec2 center{0.0f};
        float size = 0.0f;
        std::array<ViewportGizmoMarker, 6> markers{};

        // The draggable sphere; every marker lies inside it.
        [[nodiscard]] float discRadius() const { return size * 0.5f; }
        [[nodiscard]] bool discContains(const glm::vec2& point) const {
            const glm::vec2 delta = point - center;
            return glm::dot(delta, delta) <= discRadius() * discRadius();
        }
    };

    // Picks the marker drawn under the pointer. Where circles overlap, the front
    // one wins, as it is drawn on top. Just outside every drawn circle, the
    // nearest one within its hovered size still counts.
    template <typename Marker, size_t N>
    [[nodiscard]] inline int hitTestViewportGizmoMarkers(
        const std::array<Marker, N>& markers,
        const glm::vec2& mouse_pos) {
        for (const bool drawn_edge : {true, false}) {
            int hit_axis = -1;
            float hit_depth = std::numeric_limits<float>::max();
            float hit_distance_sq = std::numeric_limits<float>::max();
            for (const auto& marker : markers) {
                if (!marker.visible) {
                    continue;
                }
                const glm::vec2 delta = mouse_pos - marker.screen_pos;
                const float distance_sq = glm::dot(delta, delta);
                const float radius = marker.radius * (drawn_edge ? 1.0f : VIEWPORT_AXIS_MARKER_HOVER_SCALE);
                if (distance_sq > radius * radius) {
                    continue;
                }
                const bool better = drawn_edge
                                        ? marker.depth < hit_depth ||
                                              (marker.depth == hit_depth && distance_sq < hit_distance_sq)
                                        : distance_sq < hit_distance_sq;
                if (better) {
                    hit_axis = marker.encoded_axis;
                    hit_depth = marker.depth;
                    hit_distance_sq = distance_sq;
                }
            }
            if (hit_axis >= 0) {
                return hit_axis;
            }
        }
        return -1;
    }

    [[nodiscard]] inline bool viewportGizmoFits(const float width, const float height, const float scale) {
        // Reserve room beside the two-column tool rail and below the axis widget
        // for its home/focus controls. Hidden widgets must not capture input.
        const float dpi = std::max(scale, 1.0f);
        return width >= (VIEWPORT_AXIS_WIDGET_SIZE + 105.0f) * dpi &&
               height >= (VIEWPORT_AXIS_WIDGET_SIZE + 60.0f) * dpi;
    }

    // Input and drawing share this layout, so a marker is picked exactly where it is drawn.
    [[nodiscard]] inline std::optional<ViewportGizmoLayout> buildViewportGizmoLayout(
        const glm::mat3& camera_rotation,
        const glm::vec2& panel_pos,
        const glm::vec2& panel_size,
        const float ui_scale) {
        const float scale = std::max(ui_scale, 1.0f);
        if (!viewportGizmoFits(panel_size.x, panel_size.y, scale)) {
            return std::nullopt;
        }

        ViewportGizmoLayout layout;
        layout.size = VIEWPORT_AXIS_WIDGET_SIZE * scale;
        const float margin = VIEWPORT_AXIS_WIDGET_MARGIN * scale;
        layout.top_left = {panel_pos.x + panel_size.x - layout.size - margin, panel_pos.y + margin};
        layout.center = layout.top_left + glm::vec2(layout.size * 0.5f);

        glm::mat4 view = lfs::rendering::makeViewMatrix(camera_rotation, glm::vec3(0.0f));
        view[3][2] = -VIEWPORT_AXIS_CAMERA_DISTANCE;
        const float half_fov = glm::radians(VIEWPORT_AXIS_FOV_DEGREES) * 0.5f;
        const glm::mat4 proj = glm::perspective(half_fov * 2.0f, 1.0f, 0.1f, 10.0f);
        const float marker_radius = VIEWPORT_AXIS_MARKER_RADIUS / std::tan(half_fov) /
                                    VIEWPORT_AXIS_CAMERA_DISTANCE * layout.size * 0.5f;

        for (int encoded_axis = 0; encoded_axis < 6; ++encoded_axis) {
            auto& marker = layout.markers[static_cast<size_t>(encoded_axis)];
            marker.encoded_axis = encoded_axis;
            glm::vec3 position(0.0f);
            position[encoded_axis % 3] = marker.negative() ? -VIEWPORT_AXIS_MARKER_DISTANCE
                                                           : VIEWPORT_AXIS_MARKER_DISTANCE;
            const glm::vec4 clip = proj * view * glm::vec4(position, 1.0f);
            if (clip.w <= 0.0f) {
                continue;
            }
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            marker.screen_pos = layout.top_left + glm::vec2((ndc.x * 0.5f + 0.5f) * layout.size,
                                                            (0.5f - ndc.y * 0.5f) * layout.size);
            marker.radius = marker_radius;
            marker.depth = ndc.z;
            marker.visible = true;
        }
        return layout;
    }

} // namespace lfs::vis::gui
