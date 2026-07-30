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
            return rm ? rm->getScreenOverlayRenderer() : nullptr;
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

        [[nodiscard]] bool pointInViewportBounds(const ViewportBounds& bounds, const glm::vec2 point) {
            return point.x >= bounds.x &&
                   point.y >= bounds.y &&
                   point.x < bounds.x + bounds.width &&
                   point.y < bounds.y + bounds.height;
        }

        [[nodiscard]] lfs::vis::SelectionMode selectionModeFromModifiers(const SDL_Keymod kmods) {
            if (kmods & SDL_KMOD_SHIFT) {
                return lfs::vis::SelectionMode::Add;
            }
            if (kmods & SDL_KMOD_CTRL) {
                return lfs::vis::SelectionMode::Remove;
            }
            if (kmods & SDL_KMOD_ALT) {
                return lfs::vis::SelectionMode::Intersect;
            }
            return lfs::vis::SelectionMode::Replace;
        }

        [[nodiscard]] const char* selectionModeSuffixFromModifiers(const SDL_Keymod kmods) {
            switch (selectionModeFromModifiers(kmods)) {
            case lfs::vis::SelectionMode::Add: return " +";
            case lfs::vis::SelectionMode::Remove: return " -";
            case lfs::vis::SelectionMode::Intersect: return " &";
            case lfs::vis::SelectionMode::Replace: return "";
            }
            return "";
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
        const SDL_MouseButtonFlags mouse_buttons = SDL_GetMouseState(&mx, &my);
        last_mouse_pos_ = glm::vec2(mx, my);

        if (depth_filter_enabled_ || crop_filter_enabled_) {
            applySelectionFilterSettings(ctx);
        }

        if (auto* const sm = ctx.getSceneManager()) {
            if (auto* const service = sm->getSelectionService()) {
                auto* const rm = ctx.getRenderingManager();
                const bool passive_ring_mode =
                    rm &&
                    rm->getSelectionPreviewMode() == lfs::vis::SelectionPreviewMode::Rings &&
                    !service->isInteractiveSelectionActive();
                if (passive_ring_mode) {
                    const bool update_hover =
                        mouse_buttons == 0 &&
                        !gui::guiFocusState().want_capture_mouse &&
                        pointInViewportBounds(ctx.getViewportBounds(), last_mouse_pos_);
                    if (update_hover) {
                        const SDL_Keymod kmods = SDL_GetModState();
                        const auto mode = selectionModeFromModifiers(kmods);
                        SelectionFilterState filters{};
                        filters.crop_filter = crop_filter_enabled_;
                        filters.depth_filter = depth_filter_enabled_;
                        filters.restrict_to_selected_nodes = true;
                        service->updatePassiveRingHoverPreview(last_mouse_pos_, mode, filters);
                    } else {
                        if (rm->isCursorPreviewActive()) {
                            rm->clearCursorPreviewState();
                        }
                    }
                    return;
                }
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

        applySelectionFilterSettings(*tool_context_);
    }

    void SelectionTool::setDepthFilterRange(const bool enabled,
                                            const float depth_near,
                                            const float depth_far,
                                            const float frustum_half_width) {
        depth_near_ = std::clamp(depth_near, 0.0f, DEPTH_MAX - DEPTH_MIN);
        depth_far_ = std::clamp(depth_far, depth_near_ + DEPTH_MIN, DEPTH_MAX);
        frustum_half_width_ = std::max(frustum_half_width, 0.05f);

        depth_filter_enabled_ = enabled;
        if (!tool_context_ || !isEnabled()) {
            return;
        }

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

        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        const glm::vec2 mp{mouse_x, mouse_y};
        const auto& t = theme();

        // Cursor outlines are drawn by GuiManager in a late-stage Vulkan overlay pass so
        // their sample-to-present path stays as short as possible.

        const SDL_Keymod kmods = SDL_GetModState();
        const char* op_suffix = selectionModeSuffixFromModifiers(kmods);

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
            case lfs::vis::SelectionPreviewMode::Box: mode_name = "BOX"; break;
            case lfs::vis::SelectionPreviewMode::Sphere: mode_name = "SPHERE"; break;
            case lfs::vis::SelectionPreviewMode::Color: mode_name = "COLOR"; break;
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
