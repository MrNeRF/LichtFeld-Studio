/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/line_renderer.hpp"

#include <glm/glm.hpp>

namespace lfs::vis::gui {

    enum class BoundsGizmoHandle {
        None = -1,
        FaceXNegative = 0,
        FaceXPositive = 1,
        FaceYNegative = 2,
        FaceYPositive = 3,
        FaceZNegative = 4,
        FaceZPositive = 5,
        Corner = 6
    };

    struct BoundsGizmoConfig {
        int id = 0;
        glm::vec2 viewport_pos{0.0f};
        glm::vec2 viewport_size{0.0f};
        glm::mat4 view{1.0f};
        glm::mat4 projection{1.0f};
        glm::vec3 center_world{0.0f};
        glm::mat3 orientation_world{1.0f};
        glm::vec3 half_extents_world{0.5f};
        glm::vec3 min_half_extents_world{0.0005f};
        NativeOverlayDrawList* draw_list = nullptr;
        NativeGizmoInput input;
        bool input_enabled = true;
        bool snap = false;
        float snap_ratio = 0.1f;
    };

    struct BoundsGizmoResult {
        bool changed = false;
        bool hovered = false;
        bool active = false;
        BoundsGizmoHandle hovered_handle = BoundsGizmoHandle::None;
        BoundsGizmoHandle active_handle = BoundsGizmoHandle::None;
        glm::vec3 center_world{0.0f};
        glm::vec3 half_extents_world{0.0f};
    };

    BoundsGizmoResult drawBoundsGizmo(const BoundsGizmoConfig& config);

    [[nodiscard]] bool isBoundsGizmoHovered();
    [[nodiscard]] bool isBoundsGizmoActive();

} // namespace lfs::vis::gui
