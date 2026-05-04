/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/selection_tool.hpp"
#include "geometry/euclidean_transform.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_bindings.hpp"
#include "input/key_codes.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"
#include "theme/theme.hpp"
#include <SDL3/SDL.h>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <string>

namespace lfs::vis::tools {

    namespace {

        constexpr lfs::rendering::OverlayColor kOverlayShadow{0.0f, 0.0f, 0.0f, 180.0f / 255.0f};

        [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const auto& c) {
            return {c.x, c.y, c.z, c.w};
        }

        [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const auto& c, float alpha) {
            return {c.x, c.y, c.z, alpha};
        }

        [[nodiscard]] lfs::rendering::ScreenOverlayRenderer* getOverlayRenderer(const ToolContext& ctx) {
            auto* const rm = ctx.getRenderingManager();
            if (!rm)
                return nullptr;
            auto* const engine = rm->getRenderingEngineIfInitialized();
            if (!engine)
                return nullptr;
            return engine->getScreenOverlayRenderer();
        }

        [[nodiscard]] float depthBoxHalfHeight(const ToolContext& ctx, const float half_width) {
            const auto& bounds = ctx.getViewportBounds();
            const float aspect = (bounds.height > 0.0f)
                                     ? std::max(bounds.width / bounds.height, 0.1f)
                                     : 1.0f;
            return std::max(half_width / aspect, 0.05f);
        }

        [[nodiscard]] const Viewport& selectionFilterViewport(const ToolContext& ctx) {
            auto* const rm = ctx.getRenderingManager();
            if (rm) {
                return rm->resolveFocusedViewport(ctx.getViewport());
            }
            return ctx.getViewport();
        }

    } // namespace

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
            syncDepthFilterRenderMode(*tool_context_);
            applySelectionFilterSettings(*tool_context_);
        } else {
            clearSelectionRenderState(*tool_context_);
        }
    }

    void SelectionTool::setDepthFilterEnabled(const bool enabled) {
        if (depth_filter_enabled_ == enabled) {
            return;
        }

        depth_filter_enabled_ = enabled;
        if (!tool_context_ || !isEnabled()) {
            return;
        }

        syncDepthFilterRenderMode(*tool_context_);
        applySelectionFilterSettings(*tool_context_);
    }

    void SelectionTool::adjustDepthFar(const float scale) {
        depth_far_ = std::clamp(depth_far_ * scale, std::max(DEPTH_MIN, depth_near_ + DEPTH_MIN), DEPTH_MAX);
        if (tool_context_ && isEnabled() && depth_filter_enabled_) {
            applySelectionFilterSettings(*tool_context_);
        }
    }

    void SelectionTool::syncDepthFilterToCamera(const Viewport& viewport) {
        if (!tool_context_ || !isEnabled() || !depth_filter_enabled_) {
            return;
        }

        auto* const rm = tool_context_->getRenderingManager();
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
        const glm::quat camera_quat = glm::quat_cast(viewport.camera.R);
        const float half_height = depthBoxHalfHeight(*tool_context_, frustum_half_width_);
        settings.depth_filter_transform = lfs::geometry::EuclideanTransform(camera_quat, viewport.camera.t);
        settings.depth_filter_min = glm::vec3(-frustum_half_width_, -half_height, -depth_far_);
        settings.depth_filter_max = glm::vec3(frustum_half_width_, half_height, -depth_near_);
        rm->updateSettings(settings);
        rm->markDirty(DirtyFlag::SELECTION);
    }

    void SelectionTool::setCropFilterEnabled(const bool enabled) {
        crop_filter_enabled_ = enabled;
        if (!tool_context_ || !isEnabled()) {
            return;
        }
        applySelectionFilterSettings(*tool_context_);
    }

    void SelectionTool::drawDepthFrustum(const ToolContext& ctx) const {
        auto* const overlay = getOverlayRenderer(ctx);
        if (!overlay || !overlay->isFrameActive())
            return;

        const auto& t = theme();
        const auto& bounds = ctx.getViewportBounds();
        const float text_x = bounds.x + 10.0f;
        const float text_y = bounds.y + bounds.height - 42.0f;
        const float half_height = depthBoxHalfHeight(ctx, frustum_half_width_);

        char info_text[128];
        snprintf(info_text, sizeof(info_text), "Depth Box  Near %.2f  Far %.1f  Size %.1f x %.1f",
                 depth_near_, depth_far_, frustum_half_width_ * 2.0f, half_height * 2.0f);
        overlay->addTextWithShadow({text_x, text_y}, info_text,
                                   toOverlay(t.overlay.text), kOverlayShadow,
                                   t.fonts.large_size);
        overlay->addText({text_x, text_y + 18.0f}, "X toggle | Alt+Scroll depth",
                         toOverlay(t.overlay.text_dim, 0.78f), t.fonts.small_size);
    }

    void SelectionTool::syncDepthFilterRenderMode(const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        if (!rm) {
            return;
        }

        auto settings = rm->getSettings();

        if (depth_filter_enabled_) {
            if (!depth_filter_render_mode_snapshot_.valid) {
                depth_filter_render_mode_snapshot_.valid = true;
                depth_filter_render_mode_snapshot_.point_cloud_mode = settings.point_cloud_mode;
                depth_filter_render_mode_snapshot_.show_rings = settings.show_rings;
                depth_filter_render_mode_snapshot_.show_center_markers = settings.show_center_markers;
            }

            settings.point_cloud_mode = false;
            settings.show_rings = false;
            settings.show_center_markers = true;
            rm->updateSettings(settings);
            return;
        }

        if (!depth_filter_render_mode_snapshot_.valid) {
            return;
        }

        settings.point_cloud_mode = depth_filter_render_mode_snapshot_.point_cloud_mode;
        settings.show_rings = depth_filter_render_mode_snapshot_.show_rings;
        settings.show_center_markers = depth_filter_render_mode_snapshot_.show_center_markers;
        depth_filter_render_mode_snapshot_.valid = false;
        rm->updateSettings(settings);
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
            const auto& viewport = selectionFilterViewport(ctx);
            const glm::quat camera_quat = glm::quat_cast(viewport.camera.R);
            const float half_height = depthBoxHalfHeight(ctx, frustum_half_width_);
            settings.depth_filter_transform = lfs::geometry::EuclideanTransform(camera_quat, viewport.camera.t);
            settings.depth_filter_min = glm::vec3(-frustum_half_width_, -half_height, -depth_far_);
            settings.depth_filter_max = glm::vec3(frustum_half_width_, half_height, -depth_near_);
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

        auto* const overlay = getOverlayRenderer(*tool_context_);
        if (!overlay || !overlay->isFrameActive())
            return;

        auto selection_mode = lfs::vis::SelectionPreviewMode::Centers;
        const auto* const rm = tool_context_->getRenderingManager();
        if (rm) {
            selection_mode = rm->getSelectionPreviewMode();
        }

        const auto& viewport_bounds = tool_context_->getViewportBounds();
        const lfs::rendering::ScreenOverlayRenderer::ScopedClipRect clip(
            *overlay,
            {viewport_bounds.x, viewport_bounds.y},
            {viewport_bounds.x + viewport_bounds.width, viewport_bounds.y + viewport_bounds.height});

        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        const glm::vec2 mp{mouse_x, mouse_y};
        const auto& t = theme();
        const auto sel_border = toOverlay(t.palette.primary, 0.85f);

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

        char label_buf[32];
        float text_offset = 15.0f;
        if (selection_mode == lfs::vis::SelectionPreviewMode::Centers) {
            overlay->addCircle(mp, brush_radius_, sel_border, 32, 2.0f);
            overlay->addCircleFilled(mp, 3.0f, sel_border);
            snprintf(label_buf, sizeof(label_buf), "SEL%s", op_suffix);
            text_offset = brush_radius_ + 10.0f;
        } else {
            constexpr float CROSS_SIZE = 8.0f;
            overlay->addLine({mp.x - CROSS_SIZE, mp.y}, {mp.x + CROSS_SIZE, mp.y}, sel_border, 2.0f);
            overlay->addLine({mp.x, mp.y - CROSS_SIZE}, {mp.x, mp.y + CROSS_SIZE}, sel_border, 2.0f);

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

        const float label_size = t.fonts.large_size;
        const glm::vec2 text_pos{mp.x + text_offset, mp.y - label_size / 2.0f};
        overlay->addTextWithShadow(text_pos, label_buf,
                                   toOverlay(t.overlay.text), kOverlayShadow,
                                   label_size);

        if (depth_filter_enabled_) {
            drawDepthFrustum(*tool_context_);
        }

        if (crop_filter_enabled_) {
            constexpr float LINE_SPACING = 18.0f;
            const lfs::rendering::OverlayColor CROP_FILTER_COLOR{100.0f / 255.0f, 200.0f / 255.0f, 255.0f / 255.0f, 1.0f};
            const lfs::rendering::OverlayColor CROP_FILTER_WARN_COLOR{255.0f / 255.0f, 180.0f / 255.0f, 90.0f / 255.0f, 1.0f};
            const float text_x = viewport_bounds.x + 10.0f;
            const float text_y = viewport_bounds.y + viewport_bounds.height - (depth_filter_enabled_ ? 70.0f : 45.0f);
            std::string crop_target = "Target: select cropbox, ellipsoid, or owning splat";
            lfs::rendering::OverlayColor target_color = CROP_FILTER_WARN_COLOR;

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
                    target_color = toOverlay(t.overlay.text);
                }
            }

            overlay->addTextWithShadow({text_x, text_y}, "Crop Filter: ON",
                                       CROP_FILTER_COLOR, kOverlayShadow,
                                       t.fonts.large_size);
            overlay->addText({text_x, text_y + LINE_SPACING}, crop_target, target_color, t.fonts.small_size);
            overlay->addText({text_x, text_y + LINE_SPACING * 2.0f}, "Ctrl+Alt+C: toggle",
                             toOverlay(t.overlay.text_dim, 0.78f), t.fonts.small_size);
        }
    }

} // namespace lfs::vis::tools
