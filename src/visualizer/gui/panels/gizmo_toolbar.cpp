/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <glad/glad.h>

#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "core/splat_data_mirror.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include "gui/ui_widgets.hpp"
#include "internal/resource_paths.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    constexpr float SUBTOOLBAR_OFFSET_Y = 8.0f;

    constexpr ImGuiWindowFlags TOOLBAR_FLAGS =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;

    static ImVec2 ComputeToolbarSize(int num_buttons) {
        const auto& t = theme();
        const float width = num_buttons * t.sizes.toolbar_button_size +
                            (num_buttons - 1) * t.sizes.toolbar_spacing +
                            2.0f * t.sizes.toolbar_padding;
        const float height = t.sizes.toolbar_button_size + 2.0f * t.sizes.toolbar_padding;
        return ImVec2(width, height);
    }

    // RAII helper for toolbar style setup
    struct ToolbarStyle {
        ToolbarStyle() {
            const auto& t = theme();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.sizes.toolbar_padding, t.sizes.toolbar_padding));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(t.sizes.toolbar_spacing, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, t.toolbar_background());
        }
        ~ToolbarStyle() {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(4);
        }
    };

    // Secondary toolbar style (slightly darker)
    struct SubToolbarStyle {
        SubToolbarStyle() {
            const auto& t = theme();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.sizes.toolbar_padding, t.sizes.toolbar_padding));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(t.sizes.toolbar_spacing, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, t.subtoolbar_background());
        }
        ~SubToolbarStyle() {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(4);
        }
    };

    static ImVec2 ComputeVerticalToolbarSize(const int num_buttons, const int num_separators = 0) {
        const auto& t = theme();
        const float separator_extra = t.sizes.toolbar_spacing;
        return {
            t.sizes.toolbar_button_size + 2.0f * t.sizes.toolbar_padding,
            num_buttons * t.sizes.toolbar_button_size +
                (num_buttons - 1) * t.sizes.toolbar_spacing +
                num_separators * separator_extra +
                2.0f * t.sizes.toolbar_padding};
    }

    // Draws vertical gap between toolbar button groups
    static void DrawToolbarSeparator(const float width) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        ImGui::Dummy(ImVec2(width, theme().sizes.toolbar_spacing));
        ImGui::PopStyleVar();
    }

    struct VerticalToolbarStyle {
        VerticalToolbarStyle() {
            const auto& t = theme();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {t.sizes.toolbar_padding, t.sizes.toolbar_padding});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.0f, t.sizes.toolbar_spacing});
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0.0f, 0.0f});
            ImGui::PushStyleColor(ImGuiCol_WindowBg, t.toolbar_background());
        }
        ~VerticalToolbarStyle() {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(4);
        }
    };

    static unsigned int LoadIconTexture(const std::string& icon_name) {
        try {
            const auto path = lfs::vis::getAssetPath("icon/" + icon_name);
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

            unsigned int texture_id;
            glGenTextures(1, &texture_id);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            const GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

            lfs::core::free_image(data);
            glBindTexture(GL_TEXTURE_2D, 0);
            return texture_id;
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load toolbar icon {}: {}", icon_name, e.what());
            return 0;
        }
    }

    void InitGizmoToolbar(GizmoToolbarState& state) {
        if (state.initialized)
            return;

        state.selection_texture = LoadIconTexture("selection.png");
        state.rectangle_texture = LoadIconTexture("rectangle.png");
        state.polygon_texture = LoadIconTexture("polygon.png");
        state.lasso_texture = LoadIconTexture("lasso.png");
        state.ring_texture = LoadIconTexture("ring.png");
        state.translation_texture = LoadIconTexture("translation.png");
        state.rotation_texture = LoadIconTexture("rotation.png");
        state.scaling_texture = LoadIconTexture("scaling.png");
        state.brush_texture = LoadIconTexture("brush.png");
        state.painting_texture = LoadIconTexture("painting.png");
        state.align_texture = LoadIconTexture("align.png");
        state.cropbox_texture = LoadIconTexture("cropbox.png");
        state.bounds_texture = LoadIconTexture("bounds.png");
        state.reset_texture = LoadIconTexture("reset.png");
        state.local_texture = LoadIconTexture("local.png");
        state.world_texture = LoadIconTexture("world.png");
        state.hide_ui_texture = LoadIconTexture("layout-off.png");
        state.fullscreen_texture = LoadIconTexture("arrows-maximize.png");
        state.exit_fullscreen_texture = LoadIconTexture("arrows-minimize.png");
        state.splat_texture = LoadIconTexture("blob.png");
        state.pointcloud_texture = LoadIconTexture("dots-diagonal.png");
        state.rings_texture = LoadIconTexture("ring.png");
        state.centers_texture = LoadIconTexture("circle-dot.png");
        state.home_texture = LoadIconTexture("home.png");
        state.perspective_texture = LoadIconTexture("perspective.png");
        state.orthographic_texture = LoadIconTexture("box.png");
        state.mirror_texture = LoadIconTexture("mirror.png");
        state.mirror_x_texture = LoadIconTexture("mirror-x.png");
        state.mirror_y_texture = LoadIconTexture("mirror-y.png");
        state.mirror_z_texture = LoadIconTexture("mirror-z.png");
        state.initialized = true;
    }

    void ShutdownGizmoToolbar(GizmoToolbarState& state) {
        if (!state.initialized)
            return;

        if (state.selection_texture)
            glDeleteTextures(1, &state.selection_texture);
        if (state.rectangle_texture)
            glDeleteTextures(1, &state.rectangle_texture);
        if (state.polygon_texture)
            glDeleteTextures(1, &state.polygon_texture);
        if (state.lasso_texture)
            glDeleteTextures(1, &state.lasso_texture);
        if (state.ring_texture)
            glDeleteTextures(1, &state.ring_texture);
        if (state.translation_texture)
            glDeleteTextures(1, &state.translation_texture);
        if (state.rotation_texture)
            glDeleteTextures(1, &state.rotation_texture);
        if (state.scaling_texture)
            glDeleteTextures(1, &state.scaling_texture);
        if (state.brush_texture)
            glDeleteTextures(1, &state.brush_texture);
        if (state.painting_texture)
            glDeleteTextures(1, &state.painting_texture);
        if (state.align_texture)
            glDeleteTextures(1, &state.align_texture);
        if (state.cropbox_texture)
            glDeleteTextures(1, &state.cropbox_texture);
        if (state.bounds_texture)
            glDeleteTextures(1, &state.bounds_texture);
        if (state.reset_texture)
            glDeleteTextures(1, &state.reset_texture);
        if (state.local_texture)
            glDeleteTextures(1, &state.local_texture);
        if (state.world_texture)
            glDeleteTextures(1, &state.world_texture);
        if (state.hide_ui_texture)
            glDeleteTextures(1, &state.hide_ui_texture);
        if (state.fullscreen_texture)
            glDeleteTextures(1, &state.fullscreen_texture);
        if (state.exit_fullscreen_texture)
            glDeleteTextures(1, &state.exit_fullscreen_texture);
        if (state.splat_texture)
            glDeleteTextures(1, &state.splat_texture);
        if (state.pointcloud_texture)
            glDeleteTextures(1, &state.pointcloud_texture);
        if (state.rings_texture)
            glDeleteTextures(1, &state.rings_texture);
        if (state.centers_texture)
            glDeleteTextures(1, &state.centers_texture);
        if (state.home_texture)
            glDeleteTextures(1, &state.home_texture);
        if (state.perspective_texture)
            glDeleteTextures(1, &state.perspective_texture);
        if (state.orthographic_texture)
            glDeleteTextures(1, &state.orthographic_texture);
        if (state.mirror_texture)
            glDeleteTextures(1, &state.mirror_texture);
        if (state.mirror_x_texture)
            glDeleteTextures(1, &state.mirror_x_texture);
        if (state.mirror_y_texture)
            glDeleteTextures(1, &state.mirror_y_texture);
        if (state.mirror_z_texture)
            glDeleteTextures(1, &state.mirror_z_texture);

        state.selection_texture = 0;
        state.rectangle_texture = 0;
        state.polygon_texture = 0;
        state.lasso_texture = 0;
        state.ring_texture = 0;
        state.translation_texture = 0;
        state.rotation_texture = 0;
        state.scaling_texture = 0;
        state.brush_texture = 0;
        state.align_texture = 0;
        state.cropbox_texture = 0;
        state.bounds_texture = 0;
        state.reset_texture = 0;
        state.local_texture = 0;
        state.world_texture = 0;
        state.hide_ui_texture = 0;
        state.fullscreen_texture = 0;
        state.exit_fullscreen_texture = 0;
        state.mirror_texture = 0;
        state.mirror_x_texture = 0;
        state.mirror_y_texture = 0;
        state.mirror_z_texture = 0;
        state.initialized = false;
    }

    void DrawGizmoToolbar(const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        if (!state.initialized) {
            InitGizmoToolbar(state);
        }

        auto* const editor = ctx.editor;
        if (!editor || editor->isToolsDisabled())
            return;

        editor->validateActiveTool();

        constexpr int NUM_MAIN_BUTTONS = 8;
        constexpr float TOOLBAR_MARGIN_Y = 5.0f;
        const ImVec2 toolbar_size = ComputeToolbarSize(NUM_MAIN_BUTTONS);
        const float pos_x = viewport_pos.x + (viewport_size.x - toolbar_size.x) * 0.5f;
        const float pos_y = viewport_pos.y + TOOLBAR_MARGIN_Y;

        widgets::DrawWindowShadow({pos_x, pos_y}, toolbar_size, theme().sizes.window_rounding);
        ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(toolbar_size, ImGuiCond_Always);

        {
            const ToolbarStyle style;
            if (ImGui::Begin("##GizmoToolbar", nullptr, TOOLBAR_FLAGS)) {
                const auto& t = theme();
                const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                // Tool button helper
                const auto ToolButton = [&](const char* id, unsigned int texture,
                                            ToolType tool, ImGuizmo::OPERATION op,
                                            const char* fallback, const char* tooltip) {
                    const bool is_selected = (editor->getActiveTool() == tool);
                    const bool enabled = editor->isToolAvailable(tool);
                    const char* disabled_reason = editor->getToolUnavailableReason(tool);

                    if (!enabled)
                        ImGui::BeginDisabled();

                    const bool clicked = widgets::IconButton(id, texture, btn_size, is_selected, fallback);

                    if (!enabled)
                        ImGui::EndDisabled();

                    if (clicked && enabled) {
                        editor->setActiveTool(is_selected ? ToolType::None : tool);
                        if (!is_selected)
                            state.current_operation = op;
                    }

                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                        if (!enabled && disabled_reason) {
                            ImGui::SetTooltip("%s (%s)", tooltip, disabled_reason);
                        } else {
                            ImGui::SetTooltip("%s", tooltip);
                        }
                    }
                };

                ToolButton("##selection", state.selection_texture, ToolType::Selection, ImGuizmo::TRANSLATE, "S", "Selection");
                ImGui::SameLine();
                ToolButton("##translate", state.translation_texture, ToolType::Translate, ImGuizmo::TRANSLATE, "T", "Translate");
                ImGui::SameLine();
                ToolButton("##rotate", state.rotation_texture, ToolType::Rotate, ImGuizmo::ROTATE, "R", "Rotate");
                ImGui::SameLine();
                ToolButton("##scale", state.scaling_texture, ToolType::Scale, ImGuizmo::SCALE, "S", "Scale");
                ImGui::SameLine();
                ToolButton("##mirror", state.mirror_texture, ToolType::Mirror, ImGuizmo::TRANSLATE, "M", "Mirror");
                ImGui::SameLine();
                ToolButton("##brush", state.painting_texture, ToolType::Brush, ImGuizmo::TRANSLATE, "P", "Painting");
                ImGui::SameLine();
                ToolButton("##align", state.align_texture, ToolType::Align, ImGuizmo::TRANSLATE, "A", "3-Point Align");
                ImGui::SameLine();
                ToolButton("##cropbox", state.cropbox_texture, ToolType::CropBox, ImGuizmo::BOUNDS, "C", "Crop Box");
            }
            ImGui::End();
        }

        const ToolType active_tool = editor->getActiveTool();

        // Secondary toolbar for selection mode
        if (active_tool == ToolType::Selection) {
            constexpr int NUM_SEL_BUTTONS = 5;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_SEL_BUTTONS);

            const float sub_pos_x = viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_pos_y = viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            widgets::DrawWindowShadow({sub_pos_x, sub_pos_y}, sub_size, theme().sizes.window_rounding);
            ImGui::SetNextWindowPos(ImVec2(sub_pos_x, sub_pos_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            {
                const SubToolbarStyle style;
                if (ImGui::Begin("##SelectionModeToolbar", nullptr, TOOLBAR_FLAGS)) {
                    const auto& t = theme();
                    const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                    const auto SelectionModeButton = [&](const char* id, unsigned int texture,
                                                         SelectionSubMode mode, const char* fallback,
                                                         const char* tooltip) {
                        const bool is_selected = (state.selection_mode == mode);
                        if (widgets::IconButton(id, texture, btn_size, is_selected, fallback)) {
                            state.selection_mode = mode;
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("%s", tooltip);
                    };

                    SelectionModeButton("##centers", state.brush_texture, SelectionSubMode::Centers,
                                        "B", "Brush selection (Ctrl+1)");
                    ImGui::SameLine();
                    SelectionModeButton("##rect", state.rectangle_texture, SelectionSubMode::Rectangle,
                                        "R", "Rectangle selection (Ctrl+2)");
                    ImGui::SameLine();
                    SelectionModeButton("##polygon", state.polygon_texture, SelectionSubMode::Polygon,
                                        "P", "Polygon selection (Ctrl+3)");
                    ImGui::SameLine();
                    SelectionModeButton("##lasso", state.lasso_texture, SelectionSubMode::Lasso,
                                        "L", "Lasso selection (Ctrl+4)");
                    ImGui::SameLine();
                    SelectionModeButton("##rings", state.ring_texture, SelectionSubMode::Rings,
                                        "O", "Ring selection (Ctrl+5)");
                }
                ImGui::End();
            }
        }

        // Transform space toolbar (Local/World toggle)
        const bool is_transform_tool = (active_tool == ToolType::Translate ||
                                        active_tool == ToolType::Rotate ||
                                        active_tool == ToolType::Scale);
        if (is_transform_tool) {
            constexpr int NUM_BUTTONS = 2;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_BUTTONS);
            const float sub_x = viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_y = viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            widgets::DrawWindowShadow({sub_x, sub_y}, sub_size, theme().sizes.window_rounding);
            ImGui::SetNextWindowPos(ImVec2(sub_x, sub_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            const SubToolbarStyle style;
            if (ImGui::Begin("##TransformSpaceToolbar", nullptr, TOOLBAR_FLAGS)) {
                const auto& t = theme();
                const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                const auto SpaceButton = [&](const char* id, unsigned int tex,
                                             TransformSpace space, const char* fallback,
                                             const char* tooltip) {
                    const bool selected = (state.transform_space == space);
                    if (widgets::IconButton(id, tex, btn_size, selected, fallback)) {
                        state.transform_space = space;
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", tooltip);
                };

                SpaceButton("##local", state.local_texture, TransformSpace::Local, "L", "Local Space");
                ImGui::SameLine();
                SpaceButton("##world", state.world_texture, TransformSpace::World, "W", "World Space");
            }
            ImGui::End();
        }

        // Cropbox operations toolbar
        if (active_tool == ToolType::CropBox) {
            constexpr int NUM_CROP_BUTTONS = 5;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_CROP_BUTTONS);
            const float sub_x = viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_y = viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            widgets::DrawWindowShadow({sub_x, sub_y}, sub_size, theme().sizes.window_rounding);
            ImGui::SetNextWindowPos(ImVec2(sub_x, sub_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            const SubToolbarStyle style;
            if (ImGui::Begin("##CropBoxToolbar", nullptr, TOOLBAR_FLAGS)) {
                const auto& t = theme();
                const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                const auto CropOpButton = [&](const char* id, unsigned int tex,
                                              CropBoxOperation op, const char* fallback,
                                              const char* tooltip) {
                    const bool selected = (state.cropbox_operation == op);
                    if (widgets::IconButton(id, tex, btn_size, selected, fallback)) {
                        state.cropbox_operation = op;
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", tooltip);
                };

                CropOpButton("##crop_bounds", state.bounds_texture, CropBoxOperation::Bounds, "B", "Resize Bounds");
                ImGui::SameLine();
                CropOpButton("##crop_translate", state.translation_texture, CropBoxOperation::Translate, "T", "Translate");
                ImGui::SameLine();
                CropOpButton("##crop_rotate", state.rotation_texture, CropBoxOperation::Rotate, "R", "Rotate");
                ImGui::SameLine();
                CropOpButton("##crop_scale", state.scaling_texture, CropBoxOperation::Scale, "S", "Scale");
                ImGui::SameLine();

                if (widgets::IconButton("##crop_reset", state.reset_texture, btn_size, false, "X")) {
                    state.reset_cropbox_requested = true;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Reset to Default");
            }
            ImGui::End();
        }

        // Mirror axis sub-toolbar
        if (active_tool == ToolType::Mirror) {
            constexpr int NUM_MIRROR_BUTTONS = 3;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_MIRROR_BUTTONS);
            const float sub_x = viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_y = viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            widgets::DrawWindowShadow({sub_x, sub_y}, sub_size, theme().sizes.window_rounding);
            ImGui::SetNextWindowPos(ImVec2(sub_x, sub_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            {
                const SubToolbarStyle style;
                if (ImGui::Begin("##MirrorToolbar", nullptr, TOOLBAR_FLAGS)) {
                    const auto& t = theme();
                    const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                    const auto MirrorButton = [&](const char* id, const unsigned int tex,
                                                  const lfs::core::MirrorAxis axis,
                                                  const char* label) {
                        if (widgets::IconButton(id, tex, btn_size, false, label)) {
                            services().scene().executeMirror(axis);
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Mirror %s", label);
                        }
                    };

                    MirrorButton("##mirror_x", state.mirror_x_texture, lfs::core::MirrorAxis::X, "X");
                    ImGui::SameLine();
                    MirrorButton("##mirror_y", state.mirror_y_texture, lfs::core::MirrorAxis::Y, "Y");
                    ImGui::SameLine();
                    MirrorButton("##mirror_z", state.mirror_z_texture, lfs::core::MirrorAxis::Z, "Z");
                }
                ImGui::End();
            }
        }
    }

    namespace {
        RenderVisualization getCurrentVisualization(const RenderSettings& settings) {
            if (settings.point_cloud_mode)
                return RenderVisualization::PointCloud;
            if (settings.show_rings)
                return RenderVisualization::Rings;
            if (settings.show_center_markers)
                return RenderVisualization::Centers;
            return RenderVisualization::Splat;
        }

        void setVisualization(RenderingManager* mgr, RenderVisualization mode) {
            if (!mgr)
                return;
            auto settings = mgr->getSettings();
            settings.point_cloud_mode = (mode == RenderVisualization::PointCloud);
            settings.show_rings = (mode == RenderVisualization::Rings);
            settings.show_center_markers = (mode == RenderVisualization::Centers);
            mgr->updateSettings(settings);
        }
    } // namespace

    void DrawUtilityToolbar(GizmoToolbarState& state,
                            const ImVec2& viewport_pos, const ImVec2& viewport_size,
                            bool ui_hidden, bool is_fullscreen,
                            RenderingManager* render_manager,
                            const ::Viewport* viewport) {
        if (!state.initialized)
            InitGizmoToolbar(state);

        constexpr float MARGIN_RIGHT = 10.0f;
        constexpr float MARGIN_TOP = 5.0f;
        constexpr int FULL_BUTTON_COUNT = 8;    // Home, Fullscreen, ToggleUI, Splat, PointCloud, Rings, Centers, Projection
        constexpr int MINIMAL_BUTTON_COUNT = 3; // Home, Fullscreen, ToggleUI
        constexpr int SEPARATOR_COUNT = 2;
        const int num_buttons = render_manager ? FULL_BUTTON_COUNT : MINIMAL_BUTTON_COUNT;
        const int num_separators = render_manager ? SEPARATOR_COUNT : 0;
        const ImVec2 size = ComputeVerticalToolbarSize(num_buttons, num_separators);
        const ImVec2 pos = {
            viewport_pos.x + viewport_size.x - size.x - MARGIN_RIGHT,
            viewport_pos.y + MARGIN_TOP};

        widgets::DrawWindowShadow(pos, size, theme().sizes.window_rounding);
        ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);

        const VerticalToolbarStyle style;
        if (ImGui::Begin("##UtilityToolbar", nullptr, TOOLBAR_FLAGS)) {
            const auto& t = theme();
            const ImVec2 btn_size{t.sizes.toolbar_button_size, t.sizes.toolbar_button_size};

            // Home
            if (widgets::IconButton("##home", state.home_texture, btn_size, false, "H")) {
                lfs::core::events::cmd::ResetCamera{}.emit();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Home (H)");

            // Fullscreen
            const auto fs_tex = is_fullscreen ? state.exit_fullscreen_texture : state.fullscreen_texture;
            if (widgets::IconButton("##fullscreen", fs_tex, btn_size, is_fullscreen, "F")) {
                lfs::core::events::ui::ToggleFullscreen{}.emit();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Fullscreen (F11)");

            // Toggle UI
            if (widgets::IconButton("##hide_ui", state.hide_ui_texture, btn_size, ui_hidden, "U")) {
                lfs::core::events::ui::ToggleUI{}.emit();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Toggle UI (F12)");

            if (render_manager) {
                DrawToolbarSeparator(btn_size.x);

                const auto current = getCurrentVisualization(render_manager->getSettings());

                const auto vizButton = [&](const char* id, unsigned int tex, const char* fallback,
                                           RenderVisualization mode, const char* tooltip) {
                    if (widgets::IconButton(id, tex, btn_size, current == mode, fallback)) {
                        setVisualization(render_manager, mode);
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", tooltip);
                };

                vizButton("##splat", state.splat_texture, "S", RenderVisualization::Splat, "Splat Rendering");
                vizButton("##pointcloud", state.pointcloud_texture, "P", RenderVisualization::PointCloud, "Point Cloud");
                vizButton("##rings", state.rings_texture, "R", RenderVisualization::Rings, "Gaussian Rings");
                vizButton("##centers", state.centers_texture, "C", RenderVisualization::Centers, "Center Markers");

                DrawToolbarSeparator(btn_size.x);

                const auto settings = render_manager->getSettings();
                const bool is_ortho = settings.orthographic;
                const auto proj_tex = is_ortho ? state.orthographic_texture : state.perspective_texture;
                const char* proj_tooltip = is_ortho ? "Orthographic (O)" : "Perspective (O)";

                if (widgets::IconButton("##projection", proj_tex, btn_size, is_ortho, "O")) {
                    if (viewport) {
                        const float distance_to_pivot = glm::length(viewport->camera.getPivot() - viewport->camera.t);
                        render_manager->setOrthographic(!is_ortho, viewport_size.y, distance_to_pivot);
                    } else {
                        auto new_settings = settings;
                        new_settings.orthographic = !is_ortho;
                        render_manager->updateSettings(new_settings);
                    }
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", proj_tooltip);
            }
        }
        ImGui::End();
    }

} // namespace lfs::vis::gui::panels
