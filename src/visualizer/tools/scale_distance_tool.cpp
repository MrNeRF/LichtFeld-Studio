/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/scale_distance_tool.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/number_format.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/string_keys.hpp"
#include "input/sdl_coordinate_utils.hpp"
#include "internal/viewport.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/align_ops.hpp"
#include "operator/ops/scale_distance_ops.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis::tools {

    ScaleDistanceTool::ScaleDistanceTool() = default;

    bool ScaleDistanceTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void ScaleDistanceTool::shutdown() {
        setEnabled(false);
        if (op::operators().activeModalId() == op::to_string(op::BuiltinOp::ScaleDistancePickPoint)) {
            op::operators().cancelModalOperator();
        }
        tool_context_ = nullptr;
        services().clearScaleDistancePickedPoints();
    }

    void ScaleDistanceTool::update(const ToolContext& ctx) {
        if (isEnabled() && op::operators().activeModalId() == op::to_string(op::BuiltinOp::ScaleDistancePickPoint)) {
            op::ModalEvent event{};
            event.type = op::ModalEvent::Type::NONE;
            op::operators().dispatchModalEvent(event);
        }
        auto* const rm = ctx.getRenderingManager();
        const bool has_status = services().getScaleDistanceStatusMessage() != nullptr;
        if (had_status_ && !has_status && rm) {
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }
        had_status_ = has_status;
    }

    namespace {

        [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const lfs::rendering::OverlayColor& c) {
            return c;
        }

        [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const lfs::rendering::OverlayColor& c, const float alpha) {
            return {c.r, c.g, c.b, alpha};
        }

        [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const auto& c) {
            return {c.x, c.y, c.z, c.w};
        }

        [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const auto& c, const float alpha) {
            return {c.x, c.y, c.z, alpha};
        }

        [[nodiscard]] lfs::rendering::ScreenOverlayRenderer* getOverlayRenderer(const ToolContext& ctx) {
            auto* const rm = ctx.getRenderingManager();
            return rm ? rm->getScreenOverlayRenderer() : nullptr;
        }

        void drawLabel(lfs::rendering::ScreenOverlayRenderer& overlay, const glm::vec2& pos,
                       const char* text, const lfs::rendering::OverlayColor& color, float size) {
            const auto extent = overlay.measureText(text, size);
            const glm::vec2 padding(5.0f, 3.0f);
            overlay.addRectFilled(pos - padding, pos + extent + padding,
                                  toOverlay(theme().overlay.background, 0.96f));
            overlay.addText(pos, text, color, size);
        }

        std::vector<std::string> wrapHint(lfs::rendering::ScreenOverlayRenderer& overlay,
                                          std::string_view text, float size, float width) {
            std::vector<std::string> lines;
            while (!text.empty()) {
                size_t end = 0, last_space = 0;
                while (end < text.size()) {
                    size_t next = end + 1;
                    while (next < text.size() && (static_cast<unsigned char>(text[next]) & 0xc0) == 0x80)
                        ++next;
                    if (end > 0 && overlay.measureText(text.substr(0, next), size).x > width)
                        break;
                    if (text[end] == ' ')
                        last_space = end;
                    end = next;
                }
                if (end < text.size() && last_space > 0)
                    end = last_space;
                lines.emplace_back(text.substr(0, end));
                text.remove_prefix(end);
                while (!text.empty() && text.front() == ' ')
                    text.remove_prefix(1);
            }
            return lines;
        }

        struct PanelProjection {
            lfs::vis::RenderingManager::ViewerPanelInfo info{};
            Viewport viewport;
            float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            bool orthographic = false;
            float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
            float screen_scale_x = 1.0f;
            float screen_scale_y = 1.0f;
        };

        [[nodiscard]] std::optional<PanelProjection> resolvePanelProjection(const ToolContext& ctx,
                                                                            const glm::vec2& screen_point,
                                                                            const float fallback_focal_length_mm) {
            auto* const rm = ctx.getRenderingManager();
            if (!rm) {
                return std::nullopt;
            }

            const auto& bounds = ctx.getViewportBounds();
            const glm::vec2 viewport_pos(bounds.x, bounds.y);
            const glm::vec2 viewport_size(bounds.width, bounds.height);
            const auto panel_info = rm->resolveViewerPanel(
                ctx.getViewport(),
                viewport_pos,
                viewport_size,
                screen_point);
            if (!panel_info || !panel_info->valid()) {
                return std::nullopt;
            }

            PanelProjection proj{};
            proj.info = *panel_info;
            const auto settings = rm->getSettings();
            proj.focal_length_mm = settings.focal_length_mm;
            if (proj.focal_length_mm <= 0.0f) {
                proj.focal_length_mm = fallback_focal_length_mm;
            }
            proj.orthographic = settings.orthographic;
            proj.ortho_scale = op::alignPanelOrthoScale(*panel_info->viewport, settings);

            proj.viewport = *panel_info->viewport;
            proj.viewport.windowSize = {panel_info->render_width, panel_info->render_height};
            proj.screen_scale_x = panel_info->width / static_cast<float>(std::max(panel_info->render_width, 1));
            proj.screen_scale_y = panel_info->height / static_cast<float>(std::max(panel_info->render_height, 1));
            return proj;
        }

        [[nodiscard]] glm::vec2 screenToRender(const PanelProjection& proj, const glm::vec2& screen_point) {
            const float scale_x =
                static_cast<float>(proj.info.render_width) / std::max(proj.info.width, 1.0f);
            const float scale_y =
                static_cast<float>(proj.info.render_height) / std::max(proj.info.height, 1.0f);
            return {(screen_point.x - proj.info.x) * scale_x,
                    (screen_point.y - proj.info.y) * scale_y};
        }

        [[nodiscard]] glm::vec2 renderToScreen(const PanelProjection& proj, const glm::vec2& render_point) {
            return {proj.info.x + render_point.x * proj.screen_scale_x,
                    proj.info.y + render_point.y * proj.screen_scale_y};
        }

        struct ProjectedScreen {
            glm::vec2 pos{-1000.0f, -1000.0f};
            bool valid = false;
        };

        [[nodiscard]] ProjectedScreen projectToScreenChecked(const PanelProjection& proj,
                                                             const glm::vec3& world_pos) {
            const auto projected = lfs::rendering::projectWorldPoint(
                proj.viewport.camera.R,
                proj.viewport.camera.t,
                proj.viewport.windowSize,
                world_pos,
                proj.focal_length_mm,
                proj.orthographic,
                proj.ortho_scale);
            if (!projected) {
                return {};
            }
            return {renderToScreen(proj, glm::vec2(projected->x, projected->y)), true};
        }

        [[nodiscard]] glm::vec2 projectToScreen(const PanelProjection& proj, const glm::vec3& world_pos) {
            return projectToScreenChecked(proj, world_pos).pos;
        }

        [[nodiscard]] float markerScreenRadius(const glm::vec3& world_pos, const PanelProjection& panel_proj) {
            return op::alignMarkerScreenRadius(
                world_pos,
                panel_proj.viewport.getViewMatrix(),
                panel_proj.viewport.getProjectionMatrix(panel_proj.focal_length_mm),
                static_cast<float>(panel_proj.viewport.windowSize.y),
                panel_proj.orthographic,
                panel_proj.ortho_scale,
                panel_proj.screen_scale_x,
                panel_proj.screen_scale_y);
        }

    } // namespace

    void ScaleDistanceTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                                     [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_)
            return;

        auto* const overlay = getOverlayRenderer(*tool_context_);
        if (!overlay || !overlay->isFrameActive())
            return;

        float mx = 0.0f;
        float my = 0.0f;
        input::mouseStateInPixels(tool_context_->getWindow(), &mx, &my);
        const glm::vec2 mouse_pos{mx, my};
        auto* const rendering_manager = tool_context_->getRenderingManager();
        const float fallback_focal_length_mm = rendering_manager
                                                   ? rendering_manager->getFocalLengthMm()
                                                   : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        const bool over_gui = gui::guiFocusState().want_capture_mouse;

        const auto& bounds = tool_context_->getViewportBounds();

        const auto panel_proj_opt = resolvePanelProjection(
            *tool_context_,
            mouse_pos,
            fallback_focal_length_mm);

        const glm::ivec2 rendered_size = rendering_manager
                                             ? rendering_manager->getRenderedSize()
                                             : glm::ivec2(0, 0);
        const int fallback_render_width =
            rendered_size.x > 0 ? rendered_size.x : std::max(tool_context_->getViewport().windowSize.x, 1);
        const int fallback_render_height =
            rendered_size.y > 0 ? rendered_size.y : std::max(tool_context_->getViewport().windowSize.y, 1);

        PanelProjection panel_proj_fallback{};
        panel_proj_fallback.info.panel = SplitViewPanelId::Left;
        panel_proj_fallback.info.viewport = &tool_context_->getViewport();
        panel_proj_fallback.info.x = bounds.x;
        panel_proj_fallback.info.y = bounds.y;
        panel_proj_fallback.info.width = bounds.width;
        panel_proj_fallback.info.height = bounds.height;
        panel_proj_fallback.info.render_width = fallback_render_width;
        panel_proj_fallback.info.render_height = fallback_render_height;
        panel_proj_fallback.focal_length_mm = fallback_focal_length_mm;
        if (rendering_manager) {
            const auto settings = rendering_manager->getSettings();
            panel_proj_fallback.focal_length_mm = settings.focal_length_mm;
            panel_proj_fallback.orthographic = settings.orthographic;
            panel_proj_fallback.ortho_scale = op::alignPanelOrthoScale(tool_context_->getViewport(), settings);
        }
        panel_proj_fallback.viewport = tool_context_->getViewport();
        panel_proj_fallback.viewport.windowSize = {fallback_render_width, fallback_render_height};
        panel_proj_fallback.screen_scale_x = bounds.width / static_cast<float>(fallback_render_width);
        panel_proj_fallback.screen_scale_y = bounds.height / static_cast<float>(fallback_render_height);

        const PanelProjection& panel_proj = panel_proj_opt ? *panel_proj_opt : panel_proj_fallback;

        const lfs::rendering::ScreenOverlayRenderer::ScopedClipRect clip(
            *overlay,
            {bounds.x, bounds.y},
            {bounds.x + bounds.width, bounds.y + bounds.height});

        const auto& t = theme();
        const float ui_scale = t.fonts.base_size / 13.0f;
        const float info_x = panel_proj.info.x + 56.0f * ui_scale;
        const auto SPHERE_COLOR = toOverlay(t.palette.info);
        const auto SPHERE_OUTLINE = toOverlay(t.overlay.text);
        const auto SELECTED_OUTLINE = toOverlay(t.palette.primary);
        const auto PREVIEW_COLOR = toOverlay(t.palette.info, 0.6f);
        const auto CROSSHAIR_COLOR = toOverlay(t.palette.info, 0.8f);
        constexpr lfs::rendering::OverlayColor LINE_COLOR{1.0f, 1.0f, 1.0f, 0.95f};
        const float label_size = std::max(t.fonts.base_size, 14.0f * ui_scale);

        const auto& picked_points = services().getScaleDistancePickedPoints();
        const auto selected_point = services().getScaleDistanceSelectedPoint();

        // Draw placed markers
        for (size_t i = 0; i < picked_points.size(); ++i) {
            const auto proj_check = projectToScreenChecked(panel_proj, picked_points[i]);
            if (!proj_check.valid) {
                continue;
            }
            const glm::vec2 screen_pos = proj_check.pos;
            const float screen_radius = markerScreenRadius(picked_points[i], panel_proj);

            const bool is_selected = selected_point && *selected_point == static_cast<int>(i);
            overlay->addCircleFilled(screen_pos, screen_radius, SPHERE_COLOR, 32);
            overlay->addCircle(screen_pos, screen_radius,
                               is_selected ? SELECTED_OUTLINE : SPHERE_OUTLINE,
                               32, is_selected ? 2.5f : 1.5f);

            const char label[2] = {static_cast<char>('1' + static_cast<char>(i)), '\0'};
            overlay->addText({screen_pos.x - 4.0f, screen_pos.y - 6.0f},
                             label, toOverlay(t.overlay.text), label_size);
        }

        // Draw line and distance between 2 placed points
        if (picked_points.size() == 2) {
            const auto s0 = projectToScreenChecked(panel_proj, picked_points[0]);
            const auto s1 = projectToScreenChecked(panel_proj, picked_points[1]);
            if (s0.valid && s1.valid) {
                overlay->addLine(s0.pos, s1.pos, LINE_COLOR, 2.5f);

                const glm::vec2 mid = (s0.pos + s1.pos) * 0.5f;
                const float len = glm::length(picked_points[1] - picked_points[0]);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(len));
                drawLabel(*overlay, mid + glm::vec2(0.0f, -14.0f), buf, LINE_COLOR, label_size);
            }
        }

        // HUD box
        const float hud_size = std::max(t.fonts.base_size, 14.0f * ui_scale);
        const float padding = 10.0f * ui_scale;
        const float max_width = std::max(40.0f, std::min(560.0f * ui_scale,
                                                         panel_proj.info.width - (info_x - panel_proj.info.x) - 2.0f * padding));

        const char* hint_key = (picked_points.size() < 2)
                                   ? lichtfeld::Strings::ScaleDistance::HINT_PICKING
                                   : lichtfeld::Strings::ScaleDistance::HINT_REVIEW;
        const auto lines = wrapHint(*overlay, LOC(hint_key), hud_size, max_width);
        const auto count_text = LOCF(lichtfeld::Strings::ScaleDistance::POINTS_COUNT, lfs::core::format_count(picked_points.size()));
        const auto* status = services().getScaleDistanceStatusMessage();

        float width = overlay->measureText(count_text, t.fonts.large_size).x;
        for (const auto& line : lines)
            width = std::max(width, overlay->measureText(line, hud_size).x);
        const auto status_lines = status ? wrapHint(*overlay, *status, hud_size, max_width)
                                         : std::vector<std::string>{};
        for (const auto& line : status_lines)
            width = std::max(width, overlay->measureText(line, hud_size).x);

        const float line_height = hud_size + 5.0f * ui_scale;
        const float height = t.fonts.large_size + 8.0f * ui_scale +
                             (lines.size() + status_lines.size()) * line_height;
        const glm::vec2 hud_pos(info_x, std::max(panel_proj.info.y + padding,
                                                 panel_proj.info.y + panel_proj.info.height - height - padding - 12.0f * ui_scale));
        overlay->addRectFilled(hud_pos - glm::vec2(padding),
                               hud_pos + glm::vec2(width + padding, height + padding),
                               toOverlay(t.overlay.background, 0.96f));
        overlay->addText(hud_pos, count_text.c_str(), toOverlay(t.overlay.text), t.fonts.large_size);
        glm::vec2 text_pos(hud_pos.x, hud_pos.y + t.fonts.large_size + 8.0f * ui_scale);
        for (const auto& line : lines) {
            overlay->addText(text_pos, line.c_str(), toOverlay(t.overlay.text), hud_size);
            text_pos.y += line_height;
        }
        for (const auto& line : status_lines) {
            overlay->addText(text_pos, line.c_str(), toOverlay(t.palette.warning), hud_size);
            text_pos.y += line_height;
        }

        if (over_gui)
            return;

        overlay->addCircle(mouse_pos, 5.0f, CROSSHAIR_COLOR, 16, 2.0f);

        // Hover marker preview
        std::optional<float> hover_depth;
        if (picked_points.size() < 2 && rendering_manager) {
            const glm::vec2 render_point = screenToRender(panel_proj, mouse_pos);
            const int depth_x = static_cast<int>(render_point.x);
            const int depth_y = static_cast<int>(render_point.y);
            const float depth = rendering_manager->getDepthAtPixel(
                depth_x,
                depth_y,
                panel_proj_opt ? std::optional<SplitViewPanelId>(panel_proj.info.panel) : std::nullopt, true);
            if (depth > 0.0f && depth < 1e9f) {
                hover_depth = depth;
            }
        }

        if (hover_depth && picked_points.size() < 2) {
            const glm::vec2 render_point = screenToRender(panel_proj, mouse_pos);
            const glm::vec3 preview_point = panel_proj.viewport.unprojectPixel(
                render_point.x,
                render_point.y,
                *hover_depth,
                panel_proj.focal_length_mm,
                panel_proj.orthographic,
                panel_proj.ortho_scale);
            if (Viewport::isValidWorldPosition(preview_point)) {
                const glm::vec2 screen_pos = projectToScreen(panel_proj, preview_point);
                const float screen_radius = markerScreenRadius(preview_point, panel_proj);

                overlay->addCircleFilled(screen_pos, screen_radius, PREVIEW_COLOR, 32);
                overlay->addCircle(screen_pos, screen_radius, toOverlay(t.palette.text, 0.6f), 32, 1.5f);

                const char label[2] = {static_cast<char>('1' + static_cast<char>(picked_points.size())), '\0'};
                overlay->addText({screen_pos.x - 4.0f, screen_pos.y - 6.0f},
                                 label, toOverlay(t.palette.text, 0.7f), label_size);

                // If 1 point placed, draw live line to hover point
                if (picked_points.size() == 1) {
                    const auto s0 = projectToScreenChecked(panel_proj, picked_points[0]);
                    if (s0.valid) {
                        overlay->addLine(s0.pos, screen_pos, toOverlay(LINE_COLOR, 0.6f), 1.5f);
                        const float live_len = glm::length(preview_point - picked_points[0]);
                        char live_buf[32];
                        std::snprintf(live_buf, sizeof(live_buf), "%.3f", static_cast<double>(live_len));
                        const glm::vec2 mid = (s0.pos + screen_pos) * 0.5f;
                        drawLabel(*overlay, mid + glm::vec2(0.0f, -14.0f), live_buf,
                                  toOverlay(LINE_COLOR, 0.8f), label_size);
                    }
                }
            }
        }

        const char* instruction_key = nullptr;
        switch (picked_points.size()) {
        case 0: instruction_key = lichtfeld::Strings::ScaleDistance::CLICK_1ST; break;
        case 1: instruction_key = lichtfeld::Strings::ScaleDistance::CLICK_2ND; break;
        default: break;
        }
        if (instruction_key) {
            drawLabel(*overlay, {mouse_pos.x + 15.0f, mouse_pos.y - 10.0f},
                      LOC(instruction_key), toOverlay(t.overlay.text), label_size);
        }
    }

    void ScaleDistanceTool::onEnabledChanged(bool enabled) {
        if (tool_context_ && tool_context_->getRenderingManager()) {
            tool_context_->getRenderingManager()->setExactSurfacePicking(enabled);
        }
        if (!enabled) {
            if (op::operators().activeModalId() == op::to_string(op::BuiltinOp::ScaleDistancePickPoint)) {
                op::operators().cancelModalOperator();
            }
            services().clearScaleDistancePickedPoints();
        }
        if (tool_context_) {
            tool_context_->requestRender();
        }
    }

} // namespace lfs::vis::tools

