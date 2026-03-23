/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/selection_tool.hpp"
#include "geometry/euclidean_transform.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_bindings.hpp"
#include "input/key_codes.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"
#include "theme/theme.hpp"
#include <SDL3/SDL.h>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <imgui.h>

namespace lfs::vis::tools {

    SelectionTool::SelectionTool() = default;

    bool SelectionTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void SelectionTool::shutdown() {
        tool_context_ = nullptr;
    }

    void SelectionTool::update(const ToolContext& ctx) {
        if (!isEnabled()) {
            return;
        }

        float mx, my;
        SDL_GetMouseState(&mx, &my);
        last_mouse_pos_ = glm::vec2(mx, my);

        if (depth_filter_enabled_ || crop_filter_enabled_) {
            applySelectionFilterSettings(ctx);
        }

        if (auto* const sm = ctx.getSceneManager()) {
            if (auto* const service = sm->getSelectionService()) {
                service->refreshInteractivePreview();
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

        if (mods & input::KEYMOD_CTRL)
            return SelectionOp::Remove;
        if (mods & input::KEYMOD_SHIFT)
            return SelectionOp::Add;
        return SelectionOp::Replace;
    }

    void SelectionTool::onEnabledChanged(const bool enabled) {
        if (!tool_context_) {
            return;
        }

        auto* const sm = tool_context_->getSceneManager();
        if (sm) {
            if (auto* const service = sm->getSelectionService()) {
                service->cancelInteractiveSelection();
            }
        }

        if (enabled) {
            applySelectionFilterSettings(*tool_context_);
        } else {
            clearSelectionRenderState(*tool_context_);
        }
    }

    void SelectionTool::setDepthFilterEnabled(const bool enabled) {
        depth_filter_enabled_ = enabled;
        if (!tool_context_ || !isEnabled()) {
            return;
        }
        applySelectionFilterSettings(*tool_context_);
    }

    void SelectionTool::resetDepthFilter() {
        depth_near_ = 0.0f;
        depth_far_ = 100.0f;
        frustum_half_width_ = 50.0f;
        if (tool_context_ && isEnabled() && depth_filter_enabled_) {
            applySelectionFilterSettings(*tool_context_);
        }
    }

    void SelectionTool::adjustDepthNear(const float scale) {
        if (scale > 1.0f) {
            const float current = (depth_near_ > 0.0f) ? depth_near_ : DEPTH_MIN;
            depth_near_ = current * scale;
        } else if (depth_near_ <= DEPTH_MIN) {
            depth_near_ = 0.0f;
        } else {
            depth_near_ *= scale;
        }

        depth_near_ = std::clamp(depth_near_, 0.0f, std::max(0.0f, depth_far_ - DEPTH_MIN));
        if (tool_context_ && isEnabled() && depth_filter_enabled_) {
            applySelectionFilterSettings(*tool_context_);
        }
    }

    void SelectionTool::adjustDepthFar(const float scale) {
        depth_far_ = std::clamp(depth_far_ * scale, std::max(DEPTH_MIN, depth_near_ + DEPTH_MIN), DEPTH_MAX);
        if (tool_context_ && isEnabled() && depth_filter_enabled_) {
            applySelectionFilterSettings(*tool_context_);
        }
    }

    void SelectionTool::adjustDepthWidth(const float scale) {
        frustum_half_width_ = std::clamp(frustum_half_width_ * scale, WIDTH_MIN * 0.5f, WIDTH_MAX * 0.5f);
        if (tool_context_ && isEnabled() && depth_filter_enabled_) {
            applySelectionFilterSettings(*tool_context_);
        }
    }

    void SelectionTool::setCropFilterEnabled(const bool enabled) {
        crop_filter_enabled_ = enabled;
        if (!tool_context_ || !isEnabled()) {
            return;
        }
        applySelectionFilterSettings(*tool_context_);
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
        const auto depth_to_pos = [&](const float depth) {
            if (depth <= 0.0f) {
                return bar_x;
            }
            return bar_x + (std::log10(std::clamp(depth, DEPTH_MIN, DEPTH_MAX)) - std::log10(DEPTH_MIN)) /
                               log_range * BAR_WIDTH;
        };
        const float near_pos = depth_to_pos(depth_near_);
        const float far_pos = depth_to_pos(depth_far_);

        draw_list->AddRectFilled({near_pos, bar_y}, {far_pos, bar_y + BAR_HEIGHT}, t.progress_bar_fill_u32());
        draw_list->AddLine({near_pos, bar_y - 2}, {near_pos, bar_y + BAR_HEIGHT + 2}, t.overlay_text_u32(), 1.5f);
        draw_list->AddLine({far_pos, bar_y - 3}, {far_pos, bar_y + BAR_HEIGHT + 3}, t.progress_marker_u32(), 2.0f);

        char info_text[96];
        snprintf(info_text, sizeof(info_text), "Near: %.2f  Far: %.1f  Width: %.1f",
                 depth_near_, depth_far_, frustum_half_width_ * 2.0f);
        const ImVec2 text_pos(bar_x, bar_y - 20.0f);
        draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, {text_pos.x + 1, text_pos.y + 1},
                           t.overlay_shadow_u32(), info_text);
        draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, text_pos, t.overlay_text_u32(), info_text);

        draw_list->AddText(ImGui::GetFont(), t.fonts.small_size, {bar_x, bar_y + BAR_HEIGHT + 5.0f},
                           t.overlay_hint_u32(),
                           "Ctrl+Alt+D: toggle | Alt+Shift+Scroll: near | Alt+Scroll: far | Ctrl+Alt+Scroll: width");
    }

    void SelectionTool::applySelectionFilterSettings(const ToolContext& ctx) const {
        auto* const rm = ctx.getRenderingManager();
        if (!rm) {
            return;
        }

        auto settings = rm->getSettings();
        settings.crop_filter_for_selection = crop_filter_enabled_;
        if (crop_filter_enabled_) {
            settings.show_crop_box = true;
            settings.show_ellipsoid = true;
        }
        settings.depth_filter_enabled = depth_filter_enabled_;
        if (depth_filter_enabled_) {
            const auto& viewport = ctx.getViewport();
            const glm::quat camera_quat = glm::quat_cast(viewport.camera.R);
            settings.depth_filter_transform = lfs::geometry::EuclideanTransform(camera_quat, viewport.camera.t);
            settings.depth_filter_min = glm::vec3(-frustum_half_width_, -10000.0f, depth_near_);
            settings.depth_filter_max = glm::vec3(frustum_half_width_, 10000.0f, depth_far_);
        }
        rm->updateSettings(settings);
        rm->markDirty(DirtyFlag::SELECTION);
    }

    void SelectionTool::clearSelectionRenderState(const ToolContext& ctx) const {
        auto* const rm = ctx.getRenderingManager();
        if (!rm) {
            return;
        }

        auto settings = rm->getSettings();
        settings.crop_filter_for_selection = false;
        settings.depth_filter_enabled = false;
        rm->updateSettings(settings);
        rm->clearSelectionPreviews();
        rm->markDirty(DirtyFlag::SELECTION);
    }

    void SelectionTool::onSelectionModeChanged() {
        if (!tool_context_) {
            return;
        }
        if (auto* const sm = tool_context_->getSceneManager()) {
            if (auto* const service = sm->getSelectionService()) {
                service->cancelInteractiveSelection();
            }
        }
    }

    void SelectionTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                                 [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_ || gui::guiFocusState().want_capture_mouse) {
            return;
        }

        auto selection_mode = lfs::vis::SelectionPreviewMode::Centers;
        const auto* const rm = tool_context_->getRenderingManager();
        if (rm) {
            selection_mode = rm->getSelectionPreviewMode();
        }

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const auto& viewport_bounds = tool_context_->getViewportBounds();
        draw_list->PushClipRect({viewport_bounds.x, viewport_bounds.y},
                                {viewport_bounds.x + viewport_bounds.width, viewport_bounds.y + viewport_bounds.height},
                                true);
        const ImVec2 mouse_pos = ImGui::GetMousePos();
        const auto& t = theme();

        const SDL_Keymod keymods = SDL_GetModState();
        int mods = 0;
        if (keymods & SDL_KMOD_SHIFT)
            mods |= input::KEYMOD_SHIFT;
        if (keymods & SDL_KMOD_CTRL)
            mods |= input::KEYMOD_CTRL;

        const auto op = getOpFromModifiers(mods);
        const char* op_suffix = "";
        if (op == SelectionOp::Add)
            op_suffix = " +";
        else if (op == SelectionOp::Remove)
            op_suffix = " -";

        static char label_buf[32];
        float text_offset = 15.0f;
        if (selection_mode == lfs::vis::SelectionPreviewMode::Centers) {
            draw_list->AddCircle(mouse_pos, brush_radius_, t.selection_border_u32(), 32, 2.0f);
            draw_list->AddCircleFilled(mouse_pos, 3.0f, t.selection_border_u32());
            snprintf(label_buf, sizeof(label_buf), "SEL%s", op_suffix);
            text_offset = brush_radius_ + 10.0f;
        } else {
            constexpr float CROSS_SIZE = 8.0f;
            draw_list->AddLine({mouse_pos.x - CROSS_SIZE, mouse_pos.y},
                               {mouse_pos.x + CROSS_SIZE, mouse_pos.y}, t.selection_border_u32(), 2.0f);
            draw_list->AddLine({mouse_pos.x, mouse_pos.y - CROSS_SIZE},
                               {mouse_pos.x, mouse_pos.y + CROSS_SIZE}, t.selection_border_u32(), 2.0f);

            const char* mode_name = "";
            switch (selection_mode) {
            case lfs::vis::SelectionPreviewMode::Rings: mode_name = "RING"; break;
            case lfs::vis::SelectionPreviewMode::Rectangle: mode_name = "RECT"; break;
            case lfs::vis::SelectionPreviewMode::Polygon: mode_name = "POLY"; break;
            case lfs::vis::SelectionPreviewMode::Lasso: mode_name = "LASSO"; break;
            default: break;
            }
            snprintf(label_buf, sizeof(label_buf), "%s%s", mode_name, op_suffix);
        }

        const ImVec2 text_pos(mouse_pos.x + text_offset, mouse_pos.y - t.fonts.heading_size / 2.0f);
        draw_list->AddText(ImGui::GetFont(), t.fonts.heading_size, {text_pos.x + 1.0f, text_pos.y + 1.0f},
                           t.overlay_shadow_u32(), label_buf);
        draw_list->AddText(ImGui::GetFont(), t.fonts.heading_size, text_pos, t.overlay_text_u32(), label_buf);

        if (depth_filter_enabled_) {
            drawDepthFrustum(*tool_context_);
        }

        if (crop_filter_enabled_) {
            constexpr float LINE_SPACING = 18.0f;
            constexpr ImU32 CROP_FILTER_COLOR = IM_COL32(100, 200, 255, 255);
            constexpr ImU32 CROP_FILTER_WARN_COLOR = IM_COL32(255, 180, 90, 255);
            const float text_x = viewport_bounds.x + 10.0f;
            const float text_y = viewport_bounds.y + viewport_bounds.height - (depth_filter_enabled_ ? 70.0f : 45.0f);
            std::string crop_target = "Target: select cropbox, ellipsoid, or owning splat";
            ImU32 target_color = CROP_FILTER_WARN_COLOR;

            if (auto* const sm = tool_context_->getSceneManager()) {
                std::string targets;

                if (const auto cropbox_id = sm->getActiveSelectionCropBoxId(); cropbox_id != core::NULL_NODE) {
                    if (const auto* const node = sm->getScene().getNodeById(cropbox_id)) {
                        targets = std::string("Box: ") + node->name;
                    }
                }

                if (const auto ellipsoid_id = sm->getActiveSelectionEllipsoidId(); ellipsoid_id != core::NULL_NODE) {
                    if (const auto* const node = sm->getScene().getNodeById(ellipsoid_id)) {
                        if (!targets.empty()) {
                            targets += " | ";
                        }
                        targets += std::string("Ellipsoid: ") + node->name;
                    }
                }

                if (!targets.empty()) {
                    crop_target = targets;
                    target_color = t.overlay_text_u32();
                }
            }

            draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, {text_x + 1.0f, text_y + 1.0f},
                               t.overlay_shadow_u32(), "Crop Filter: ON");
            draw_list->AddText(ImGui::GetFont(), t.fonts.large_size, {text_x, text_y},
                               CROP_FILTER_COLOR, "Crop Filter: ON");
            draw_list->AddText(ImGui::GetFont(), t.fonts.small_size, {text_x, text_y + LINE_SPACING},
                               target_color, crop_target.c_str());
            draw_list->AddText(ImGui::GetFont(), t.fonts.small_size, {text_x, text_y + LINE_SPACING * 2.0f},
                               t.overlay_hint_u32(), "Ctrl+Alt+C: toggle");
        }

        draw_list->PopClipRect();
    }

} // namespace lfs::vis::tools
