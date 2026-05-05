/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/line_renderer.hpp"

#include <glm/glm.hpp>

namespace lfs::vis::gui {

    enum class RotationGizmoAxis {
        None = -1,
        X = 0,
        Y = 1,
        Z = 2,
        View = 3
    };

    struct RotationGizmoConfig {
        int id = 0;
        glm::vec2 viewport_pos{0.0f};
        glm::vec2 viewport_size{0.0f};
        glm::mat4 view{1.0f};
        glm::mat4 projection{1.0f};
        glm::vec3 pivot_world{0.0f};
        glm::mat3 orientation_world{1.0f};
        NativeOverlayDrawList* draw_list = nullptr;
        NativeGizmoInput input;
        bool input_enabled = true;
        bool snap = false;
        float snap_degrees = 5.0f;
    };

    struct RotationGizmoResult {
        bool changed = false;
        bool hovered = false;
        bool active = false;
        RotationGizmoAxis hovered_axis = RotationGizmoAxis::None;
        RotationGizmoAxis active_axis = RotationGizmoAxis::None;
        glm::mat3 delta_rotation{1.0f};
    };

    LFS_VIS_API RotationGizmoResult drawRotationGizmo(const RotationGizmoConfig& config);

    [[nodiscard]] bool isRotationGizmoHovered();
    [[nodiscard]] bool isRotationGizmoActive();

} // namespace lfs::vis::gui
