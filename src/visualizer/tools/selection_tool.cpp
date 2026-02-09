/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/selection_tool.hpp"
#include "core/tensor.hpp"
#include "input/input_bindings.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include <GLFW/glfw3.h>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis::tools {

    SelectionTool::SelectionTool() = default;

    bool SelectionTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void SelectionTool::shutdown() {
        tool_context_ = nullptr;
    }

    void SelectionTool::update([[maybe_unused]] const ToolContext& ctx) {
        if (isEnabled()) {
            double mx, my;
            glfwGetCursorPos(ctx.getWindow(), &mx, &my);
            last_mouse_pos_ = glm::vec2(static_cast<float>(mx), static_cast<float>(my));

            if (depth_filter_enabled_) {
                updateSelectionCropBox(ctx);
            }
        }
    }

    SelectionOp SelectionTool::getOpFromModifiers(const int mods) const {
        if (input_bindings_) {
            const auto action = input_bindings_->getActionForDrag(
                input::ToolMode::SELECTION, input::MouseButton::LEFT, mods);
            if (action == input::Action::SELECTION_REMOVE)
                return SelectionOp::Remove;
            if (action == input::Action::SELECTION_ADD)
                return SelectionOp::Add;
            if (action == input::Action::SELECTION_REPLACE)
                return SelectionOp::Replace;
        }
        // Fallback
        if (mods & GLFW_MOD_CONTROL)
            return SelectionOp::Remove;
        if (mods & GLFW_MOD_SHIFT)
            return SelectionOp::Add;
        return SelectionOp::Replace;
    }

    void SelectionTool::onEnabledChanged(const bool enabled) {
        resetPolygon();

        if (depth_filter_enabled_ && tool_context_) {
            disableDepthFilter(*tool_context_);
        }
        depth_filter_enabled_ = false;

        if (crop_filter_enabled_ && tool_context_) {
            if (auto* const rm = tool_context_->getRenderingManager()) {
                auto settings = rm->getSettings();
                settings.crop_filter_for_selection = enabled;
                if (enabled) {
                    settings.show_crop_box = true;
                    settings.show_ellipsoid = true;
                }
                rm->updateSettings(settings);
            }
        }

        if (tool_context_) {
            if (auto* const rm = tool_context_->getRenderingManager()) {
                rm->setOutputScreenPositions(enabled);
                rm->clearBrushState();
                rm->clearPreviewSelection();
                rm->markDirty();
            }
        }
    }

    void SelectionTool::resetPolygon() {
        polygon_points_.clear();
        polygon_closed_ = false;
    }

    void SelectionTool::clearPolygon() {
        if (polygon_points_.empty())
            return;

        if (tool_context_) {
            if (auto* const rm = tool_context_->getRenderingManager()) {
                rm->clearPreviewSelection();
                rm->markDirty();
            }
        }
        resetPolygon();
    }

    void SelectionTool::onSelectionModeChanged() {
        clearPolygon();
    }

    int SelectionTool::findPolygonVertexAt(const float x, const float y) const {
        constexpr float RADIUS_SQ = POLYGON_VERTEX_RADIUS * POLYGON_VERTEX_RADIUS;
        const glm::vec2 p(x, y);
        for (size_t i = 0; i < polygon_points_.size(); ++i) {
            const glm::vec2 d = p - polygon_points_[i];
            if (glm::dot(d, d) <= RADIUS_SQ)
                return static_cast<int>(i);
        }
        return -1;
    }

    void SelectionTool::resetDepthFilter() {
        depth_filter_enabled_ = false;
        depth_far_ = 100.0f;
        frustum_half_width_ = 50.0f;
    }

    void SelectionTool::updateSelectionCropBox(const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        if (!rm)
            return;

        const auto& viewport = ctx.getViewport();
        const glm::quat cam_quat = glm::quat_cast(viewport.camera.R);
        const lfs::geometry::EuclideanTransform filter_transform(cam_quat, viewport.camera.t);

        constexpr float Y_BOUND = 10000.0f;
        const glm::vec3 filter_min(-frustum_half_width_, -Y_BOUND, 0.0f);
        const glm::vec3 filter_max(frustum_half_width_, Y_BOUND, depth_far_);

        auto settings = rm->getSettings();
        settings.depth_filter_enabled = true;
        settings.depth_filter_transform = filter_transform;
        settings.depth_filter_min = filter_min;
        settings.depth_filter_max = filter_max;
        rm->updateSettings(settings);
    }

    void SelectionTool::disableDepthFilter(const ToolContext& ctx) {
        depth_filter_enabled_ = false;

        auto* const rm = ctx.getRenderingManager();
        if (rm) {
            auto settings = rm->getSettings();
            settings.depth_filter_enabled = false;
            rm->updateSettings(settings);
        }
    }

    void SelectionTool::setCropFilterEnabled(const bool enabled) {
        crop_filter_enabled_ = enabled;

        if (!tool_context_)
            return;

        auto* const rm = tool_context_->getRenderingManager();
        auto* const sm = tool_context_->getSceneManager();
        if (!rm)
            return;

        auto settings = rm->getSettings();
        settings.crop_filter_for_selection = enabled;

        if (enabled) {
            settings.show_crop_box = true;
            settings.show_ellipsoid = true;

            if (sm) {
                node_before_crop_filter_ = sm->getSelectedNodeName();
                const auto& cropboxes = sm->getScene().getVisibleCropBoxes();
                const auto& ellipsoids = sm->getScene().getVisibleEllipsoids();

                if (!cropboxes.empty()) {
                    if (const auto* node = sm->getScene().getNodeById(cropboxes[0].node_id)) {
                        sm->selectNode(node->name);
                    }
                } else if (!ellipsoids.empty()) {
                    if (const auto* node = sm->getScene().getNodeById(ellipsoids[0].node_id)) {
                        sm->selectNode(node->name);
                    }
                }
            }
        } else if (sm && !node_before_crop_filter_.empty()) {
            if (sm->getScene().getNode(node_before_crop_filter_))
                sm->selectNode(node_before_crop_filter_);
            node_before_crop_filter_.clear();
        }

        rm->updateSettings(settings);
    }

    void SelectionTool::drawDepthFrustum(const ToolContext& ctx) const {
        constexpr float BAR_HEIGHT = 8.0f;
        constexpr float BAR_WIDTH = 200.0f;
        const auto& t = theme();

        const auto& bounds = ctx.getViewportBounds();
        const float bar_x = bounds.x + 10.0f;
        const float bar_y = bounds.y + bounds.height - 45.0f;

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();

        draw_list->AddRectFilled({bar_x, bar_y}, {bar_x + BAR_WIDTH, bar_y + BAR_HEIGHT}, t.progress_bar_bg_u32());

        const float log_range = std::log10(DEPTH_MAX) - std::log10(DEPTH_MIN);
        const float far_pos = bar_x + (std::log10(depth_far_) - std::log10(DEPTH_MIN)) / log_range * BAR_WIDTH;

        draw_list->AddRectFilled({bar_x, bar_y}, {far_pos, bar_y + BAR_HEIGHT}, t.progress_bar_fill_u32());
        draw_list->AddLine({far_pos, bar_y - 3}, {far_pos, bar_y + BAR_HEIGHT + 3}, t.progress_marker_u32(), 2.0f);

        char info_text[64];
        if (frustum_half_width_ < WIDTH_MAX - 1.0f) {
            snprintf(info_text, sizeof(info_text), "Depth: %.1f  Width: %.1f", depth_far_, frustum_half_width_ * 2.0f);
        } else {
            snprintf(info_text, sizeof(info_text), "Depth: %.1f", depth_far_);
        }
        const ImVec2 text_pos(bar_x, bar_y - 20.0f);
        draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, {text_pos.x + 1, text_pos.y + 1}, t.overlay_shadow_u32(), info_text);
        draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, text_pos, t.overlay_text_u32(), info_text);

        draw_list->AddText(ImGui::GetFont(), t.fonts.small_size, {bar_x, bar_y + BAR_HEIGHT + 5.0f}, t.overlay_hint_u32(),
                           "Alt+Scroll: depth | Ctrl+Alt+Scroll: width | Esc: off");
    }

    void SelectionTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                                 [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_ || ImGui::GetIO().WantCaptureMouse)
            return;

        auto sel_mode = lfs::rendering::SelectionMode::Centers;
        const auto* const rm = tool_context_->getRenderingManager();
        if (rm)
            sel_mode = rm->getSelectionMode();

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const auto& vp_bounds = tool_context_->getViewportBounds();
        draw_list->PushClipRect({vp_bounds.x, vp_bounds.y},
                                {vp_bounds.x + vp_bounds.width, vp_bounds.y + vp_bounds.height}, true);
        const ImVec2 mouse_pos = ImGui::GetMousePos();
        const auto& t = theme();

        // Draw polygon if in polygon mode
        if (sel_mode == lfs::rendering::SelectionMode::Polygon && !polygon_points_.empty()) {
            const ImU32 VERTEX_COLOR = t.polygon_vertex_u32();
            const ImU32 VERTEX_HOVER_COLOR = t.polygon_vertex_hover_u32();
            const ImU32 CLOSE_HINT_COLOR = t.polygon_close_hint_u32();
            const ImU32 FILL_COLOR = t.selection_fill_u32();
            const ImU32 LINE_TO_MOUSE_COLOR = t.selection_line_u32();

            for (size_t i = 1; i < polygon_points_.size(); ++i) {
                draw_list->AddLine(ImVec2(polygon_points_[i - 1].x, polygon_points_[i - 1].y),
                                   ImVec2(polygon_points_[i].x, polygon_points_[i].y), t.selection_border_u32(), 2.0f);
            }

            if (polygon_closed_) {
                draw_list->AddLine(ImVec2(polygon_points_.back().x, polygon_points_.back().y),
                                   ImVec2(polygon_points_.front().x, polygon_points_.front().y), t.selection_border_u32(), 2.0f);
                if (polygon_points_.size() >= 3) {
                    std::vector<ImVec2> im_points;
                    im_points.reserve(polygon_points_.size());
                    for (const auto& pt : polygon_points_)
                        im_points.emplace_back(pt.x, pt.y);
                    draw_list->AddConvexPolyFilled(im_points.data(), static_cast<int>(im_points.size()), FILL_COLOR);
                }
            } else {
                draw_list->AddLine(ImVec2(polygon_points_.back().x, polygon_points_.back().y),
                                   mouse_pos, LINE_TO_MOUSE_COLOR, 1.0f);
                if (polygon_points_.size() >= 3) {
                    const glm::vec2 d = glm::vec2(mouse_pos.x, mouse_pos.y) - polygon_points_.front();
                    if (glm::dot(d, d) < POLYGON_CLOSE_THRESHOLD * POLYGON_CLOSE_THRESHOLD) {
                        draw_list->AddCircle(ImVec2(polygon_points_.front().x, polygon_points_.front().y),
                                             POLYGON_VERTEX_RADIUS + 3.0f, CLOSE_HINT_COLOR, 16, 2.0f);
                    }
                }
            }

            const int hovered_idx = findPolygonVertexAt(mouse_pos.x, mouse_pos.y);
            for (size_t i = 0; i < polygon_points_.size(); ++i) {
                const auto& pt = polygon_points_[i];
                const ImU32 color = (static_cast<int>(i) == hovered_idx) ? VERTEX_HOVER_COLOR : VERTEX_COLOR;
                draw_list->AddCircleFilled(ImVec2(pt.x, pt.y), POLYGON_VERTEX_RADIUS, color);
                draw_list->AddCircle(ImVec2(pt.x, pt.y), POLYGON_VERTEX_RADIUS, t.selection_border_u32(), 16, 1.5f);
            }
        }

        // Modifier indicator
        const char* mod_suffix = "";
        GLFWwindow* const win = tool_context_->getWindow();
        const bool shift = glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                           glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        const bool ctrl = glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                          glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        int mods = 0;
        if (shift)
            mods |= GLFW_MOD_SHIFT;
        if (ctrl)
            mods |= GLFW_MOD_CONTROL;

        if (mods != 0) {
            const auto op = getOpFromModifiers(mods);
            if (op == SelectionOp::Add)
                mod_suffix = " +";
            else if (op == SelectionOp::Remove)
                mod_suffix = " -";
        }

        // Build label
        static char label_buf[24];
        float text_offset = 15.0f;
        const bool is_brush = (sel_mode == lfs::rendering::SelectionMode::Centers);

        if (is_brush) {
            draw_list->AddCircle(mouse_pos, brush_radius_, t.selection_border_u32(), 32, 2.0f);
            draw_list->AddCircleFilled(mouse_pos, 3.0f, t.selection_border_u32());
            snprintf(label_buf, sizeof(label_buf), "SEL%s", mod_suffix);
            text_offset = brush_radius_ + 10.0f;
        } else {
            constexpr float CROSS_SIZE = 8.0f;
            draw_list->AddLine(ImVec2(mouse_pos.x - CROSS_SIZE, mouse_pos.y),
                               ImVec2(mouse_pos.x + CROSS_SIZE, mouse_pos.y), t.selection_border_u32(), 2.0f);
            draw_list->AddLine(ImVec2(mouse_pos.x, mouse_pos.y - CROSS_SIZE),
                               ImVec2(mouse_pos.x, mouse_pos.y + CROSS_SIZE), t.selection_border_u32(), 2.0f);

            const char* mode_name = "";
            const char* suffix = "";
            switch (sel_mode) {
            case lfs::rendering::SelectionMode::Rings: mode_name = "RING"; break;
            case lfs::rendering::SelectionMode::Rectangle: mode_name = "RECT"; break;
            case lfs::rendering::SelectionMode::Polygon:
                mode_name = "POLY";
                suffix = polygon_closed_ ? " [Enter]" : "";
                break;
            case lfs::rendering::SelectionMode::Lasso: mode_name = "LASSO"; break;
            default: break;
            }
            snprintf(label_buf, sizeof(label_buf), "%s%s%s", mode_name, mod_suffix, suffix);
        }

        const ImVec2 text_pos(mouse_pos.x + text_offset, mouse_pos.y - t.fonts.heading_size / 2);
        draw_list->AddText(ImGui::GetFont(), t.fonts.heading_size, ImVec2(text_pos.x + 1, text_pos.y + 1), t.overlay_shadow_u32(), label_buf);
        draw_list->AddText(ImGui::GetFont(), t.fonts.heading_size, text_pos, t.overlay_text_u32(), label_buf);

        // Draw depth filter
        if (depth_filter_enabled_) {
            drawDepthFrustum(*tool_context_);
        }

        if (crop_filter_enabled_) {
            constexpr float MARGIN_X = 10.0f;
            constexpr float MARGIN_SINGLE = 45.0f;
            constexpr float MARGIN_STACKED = 70.0f;
            constexpr float LINE_SPACING = 18.0f;
            constexpr ImU32 CROP_FILTER_COLOR = IM_COL32(100, 200, 255, 255);

            const float text_x = vp_bounds.x + MARGIN_X;
            const float text_y = vp_bounds.y + vp_bounds.height - (depth_filter_enabled_ ? MARGIN_STACKED : MARGIN_SINGLE);

            draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, {text_x + 1.0f, text_y + 1.0f},
                               t.overlay_shadow_u32(), "Crop Filter: ON");
            draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, {text_x, text_y}, CROP_FILTER_COLOR,
                               "Crop Filter: ON");
            draw_list->AddText(ImGui::GetFont(), t.fonts.small_size, {text_x, text_y + LINE_SPACING},
                               t.overlay_hint_u32(), "Ctrl+Shift+F: off");
        }
        draw_list->PopClipRect();
    }

} // namespace lfs::vis::tools
