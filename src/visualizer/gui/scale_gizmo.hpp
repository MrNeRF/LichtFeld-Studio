/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/line_renderer.hpp"

#include <glm/glm.hpp>

namespace lfs::vis::gui {

    enum class ScaleGizmoHandle {
        None = -1,
        X = 0,
        Y = 1,
        Z = 2,
        XY = 3,
        YZ = 4,
        ZX = 5,
        Uniform = 6
    };

    struct ScaleGizmoConfig {
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
        float snap_ratio = 0.1f;
    };

    struct ScaleGizmoResult {
        bool changed = false;
        bool hovered = false;
        bool active = false;
        ScaleGizmoHandle hovered_handle = ScaleGizmoHandle::None;
        ScaleGizmoHandle active_handle = ScaleGizmoHandle::None;
        glm::vec3 delta_scale{1.0f};
        glm::vec3 total_scale{1.0f};
    };

    LFS_VIS_API ScaleGizmoResult drawScaleGizmo(const ScaleGizmoConfig& config);

    [[nodiscard]] bool isScaleGizmoHovered();
    [[nodiscard]] bool isScaleGizmoActive();

} // namespace lfs::vis::gui
