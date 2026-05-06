/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/line_renderer.hpp"

#include <glm/glm.hpp>

namespace lfs::vis::gui {

    enum class TranslationGizmoHandle {
        None = -1,
        X = 0,
        Y = 1,
        Z = 2,
        XY = 3,
        YZ = 4,
        ZX = 5,
        View = 6
    };

    struct TranslationGizmoConfig {
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
        float snap_units = 0.1f;
    };

    struct TranslationGizmoResult {
        bool changed = false;
        bool hovered = false;
        bool active = false;
        TranslationGizmoHandle hovered_handle = TranslationGizmoHandle::None;
        TranslationGizmoHandle active_handle = TranslationGizmoHandle::None;
        glm::vec3 delta_translation{0.0f};
        glm::vec3 total_translation{0.0f};
    };

    LFS_VIS_API TranslationGizmoResult drawTranslationGizmo(const TranslationGizmoConfig& config);

    [[nodiscard]] bool isTranslationGizmoHovered();
    [[nodiscard]] bool isTranslationGizmoActive();

} // namespace lfs::vis::gui
