/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/editor_context.hpp"
#include "gui/ui_context.hpp"
#include <imgui.h>
#include <ImGuizmo.h>

class Viewport; // Forward declare global Viewport class

namespace lfs::vis {
    class RenderingManager;
}

namespace lfs::vis::gui::panels {

    // Re-export ToolType from EditorContext for convenience
    using ToolType = lfs::vis::ToolType;

    enum class CropBoxOperation {
        Bounds,
        Translate,
        Rotate,
        Scale
    };

    enum class SelectionSubMode {
        Centers,
        Rectangle,
        Polygon,
        Lasso,
        Rings
    };

    enum class TransformSpace {
        Local,
        World
    };

    enum class RenderVisualization {
        Splat,      // Normal gaussian splat rendering
        PointCloud, // Point cloud mode
        Rings,      // Show gaussian rings
        Centers     // Show center markers
    };

    struct GizmoToolbarState {
        ImGuizmo::OPERATION current_operation = ImGuizmo::TRANSLATE;
        CropBoxOperation cropbox_operation = CropBoxOperation::Bounds;
        bool reset_cropbox_requested = false;
        SelectionSubMode selection_mode = SelectionSubMode::Centers;
        TransformSpace transform_space = TransformSpace::Local;
        bool initialized = false;

        // Icon textures
        unsigned int selection_texture = 0;
        unsigned int rectangle_texture = 0;
        unsigned int polygon_texture = 0;
        unsigned int lasso_texture = 0;
        unsigned int ring_texture = 0;
        unsigned int translation_texture = 0;
        unsigned int rotation_texture = 0;
        unsigned int scaling_texture = 0;
        unsigned int brush_texture = 0;
        unsigned int painting_texture = 0;
        unsigned int align_texture = 0;
        unsigned int cropbox_texture = 0;
        unsigned int bounds_texture = 0;
        unsigned int reset_texture = 0;
        unsigned int local_texture = 0;
        unsigned int world_texture = 0;
        unsigned int hide_ui_texture = 0;
        unsigned int fullscreen_texture = 0;
        unsigned int exit_fullscreen_texture = 0;

        // Render visualization icons
        unsigned int splat_texture = 0;
        unsigned int pointcloud_texture = 0;
        unsigned int rings_texture = 0;
        unsigned int centers_texture = 0;
        unsigned int home_texture = 0;

        // Projection mode icons
        unsigned int perspective_texture = 0;
        unsigned int orthographic_texture = 0;

        // Mirror tool icons
        unsigned int mirror_texture = 0;
        unsigned int mirror_x_texture = 0;
        unsigned int mirror_y_texture = 0;
        unsigned int mirror_z_texture = 0;
    };

    void InitGizmoToolbar(GizmoToolbarState& state);
    void ShutdownGizmoToolbar(GizmoToolbarState& state);
    void DrawGizmoToolbar(const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size);
    void DrawUtilityToolbar(GizmoToolbarState& state,
                            const ImVec2& viewport_pos, const ImVec2& viewport_size,
                            bool ui_hidden, bool is_fullscreen,
                            lfs::vis::RenderingManager* render_manager = nullptr,
                            const ::Viewport* viewport = nullptr);

} // namespace lfs::vis::gui::panels
