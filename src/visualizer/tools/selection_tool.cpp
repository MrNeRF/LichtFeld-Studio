/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/selection_tool.hpp"
#include "geometry/euclidean_transform.hpp"
#include "gui/gui_focus_state.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"
#include "theme/theme.hpp"
#include <SDL3/SDL.h>
#include <cmath>
#include <cstdio>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

namespace lfs::vis::tools {

    namespace {

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
        if (const auto* const rm = tool_context_->getRenderingManager()) {
            selection_mode = rm->getSelectionPreviewMode();
        }

        const auto& viewport_bounds = tool_context_->getViewportBounds();
        const lfs::rendering::ScreenOverlayRenderer::ScopedClipRect clip(
            *overlay,
            {viewport_bounds.x, viewport_bounds.y},
            {viewport_bounds.x + viewport_bounds.width, viewport_bounds.y + viewport_bounds.height});

        // ImGui::GetMousePos returns the cached, NewFrame-aligned cursor — matches what the
        // viewport pass will see this frame. SDL_GetMouseState samples one extra event-pump
        // late, which surfaces as a visible lag on the selection ring.
        const ImVec2 mouse_imv = ImGui::GetMousePos();
        const glm::vec2 mp{mouse_imv.x, mouse_imv.y};
        const auto& t = theme();
        const auto sel_border = toOverlay(t.palette.primary, 0.85f);

        // Cursor circle / cross is drawn by GuiManager in a late-stage pass so the
        // sample-to-present path is as short as possible and the ring tracks the mouse
        // without a visible frame trail. Labels stay here; their offset from the cursor
        // makes any small drift much less perceptible than the ring itself.
        (void)sel_border;

        const SDL_Keymod kmods = SDL_GetModState();
        const char* op_suffix = "";
        if (kmods & SDL_KMOD_CTRL) {
            op_suffix = " -";
        } else if (kmods & SDL_KMOD_SHIFT) {
            op_suffix = " +";
        }

        const char* mode_name = nullptr;
        float text_offset = 15.0f;
        if (selection_mode == lfs::vis::SelectionPreviewMode::Centers) {
            mode_name = "SEL";
            text_offset = brush_radius_ + 10.0f;
        } else {
            switch (selection_mode) {
            case lfs::vis::SelectionPreviewMode::Rings: mode_name = "RING"; break;
            case lfs::vis::SelectionPreviewMode::Rectangle: mode_name = "RECT"; break;
            case lfs::vis::SelectionPreviewMode::Polygon: mode_name = "POLY"; break;
            case lfs::vis::SelectionPreviewMode::Lasso: mode_name = "LASSO"; break;
            default: break;
            }
        }

        if (mode_name) {
            char label_buf[32];
            std::snprintf(label_buf, sizeof(label_buf), "%s%s", mode_name, op_suffix);
            const float label_size = t.fonts.large_size;
            const glm::vec2 text_pos{mp.x + text_offset, mp.y - label_size * 0.5f};
            constexpr lfs::rendering::OverlayColor kShadow{0.0f, 0.0f, 0.0f, 180.0f / 255.0f};
            overlay->addTextWithShadow(text_pos, label_buf, toOverlay(t.overlay.text), kShadow, label_size);
        }
    }

} // namespace lfs::vis::tools
