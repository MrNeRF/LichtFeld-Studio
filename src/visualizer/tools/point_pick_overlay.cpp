/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_pick_overlay.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/sdl_coordinate_utils.hpp"
#include "operator/ops/point_pick_operator.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "theme/theme.hpp"
#include <algorithm>

namespace lfs::vis::tools {

    [[nodiscard]] std::string pointPickShortcut(const input::Action action) {
        const auto* const controller = InputController::instance();
        return controller ? controller->getBindings().getLocalizedTriggerDescription(action, input::ToolMode::ALIGN)
                          : std::string{};
    }

    [[nodiscard]] lfs::rendering::ScreenOverlayRenderer* getOverlayRenderer(const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        return rm ? rm->getScreenOverlayRenderer() : nullptr;
    }

    void drawLabel(lfs::rendering::ScreenOverlayRenderer& overlay, const glm::vec2& pos,
                   const char* text, const lfs::rendering::OverlayColor& color, float size) {
        const auto extent = overlay.measureText(text, size);
        const glm::vec2 padding(4.0f, 3.0f);
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
        const auto panel_info = rm->resolveViewerPanel(rm->activeViewId(),
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
        proj.ortho_scale = op::pointPickPanelOrthoScale(*rm, rm->activeViewId());

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
        return op::pointPickMarkerScreenRadius(
            world_pos,
            panel_proj.viewport.getViewMatrix(),
            panel_proj.viewport.getProjectionMatrix(panel_proj.focal_length_mm),
            static_cast<float>(panel_proj.viewport.windowSize.y),
            panel_proj.orthographic,
            panel_proj.ortho_scale,
            panel_proj.screen_scale_x,
            panel_proj.screen_scale_y);
    }

    PointPickOverlayFrame pointPickOverlayFrame(const ToolContext& ctx) {
        float mx = 0.0f;
        float my = 0.0f;
        input::mouseStateInPixels(ctx.getWindow(), &mx, &my);
        const glm::vec2 mouse_pos{mx, my};
        auto* const rendering_manager = ctx.getRenderingManager();
        const float fallback_focal_length_mm = rendering_manager
                                                   ? rendering_manager->getFocalLengthMm()
                                                   : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        const bool over_gui = gui::guiFocusState().want_capture_mouse;

        const auto& bounds = ctx.getViewportBounds();

        const auto panel_proj_opt = resolvePanelProjection(
            ctx,
            mouse_pos,
            fallback_focal_length_mm);

        const glm::ivec2 rendered_size = rendering_manager
                                             ? rendering_manager->getRenderedSize()
                                             : glm::ivec2(0, 0);
        const int fallback_render_width =
            rendered_size.x > 0 ? rendered_size.x : std::max(ctx.getViewport().windowSize.x, 1);
        const int fallback_render_height =
            rendered_size.y > 0 ? rendered_size.y : std::max(ctx.getViewport().windowSize.y, 1);

        PanelProjection panel_proj_fallback{};
        panel_proj_fallback.info.panel = SplitViewPanelId::Left;
        panel_proj_fallback.info.viewport = &ctx.getViewport();
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
            panel_proj_fallback.ortho_scale = op::pointPickPanelOrthoScale(*rendering_manager, rendering_manager->activeViewId());
        }
        panel_proj_fallback.viewport = ctx.getViewport();
        panel_proj_fallback.viewport.windowSize = {fallback_render_width, fallback_render_height};
        panel_proj_fallback.screen_scale_x = bounds.width / static_cast<float>(fallback_render_width);
        panel_proj_fallback.screen_scale_y = bounds.height / static_cast<float>(fallback_render_height);

        return {mouse_pos, panel_proj_opt ? *panel_proj_opt : panel_proj_fallback, panel_proj_opt.has_value(), over_gui};
    }

    void drawPointMarkers(lfs::rendering::ScreenOverlayRenderer& overlay, const PanelProjection& panel_proj,
                          const std::vector<glm::vec3>& picked_points, std::optional<int> selected_point,
                          const lfs::rendering::OverlayColor& color, float label_size) {
        const auto& t = theme();
        for (size_t i = 0; i < picked_points.size(); ++i) {
            const glm::vec2 screen_pos = projectToScreen(panel_proj, picked_points[i]);
            const float screen_radius = markerScreenRadius(picked_points[i], panel_proj);

            const bool is_selected = selected_point && *selected_point == static_cast<int>(i);
            overlay.addCircleFilled(screen_pos, screen_radius, color, 32);
            overlay.addCircle(screen_pos, screen_radius,
                              is_selected ? toOverlay(t.palette.primary) : toOverlay(t.overlay.text),
                              32, is_selected ? 2.5f : 1.5f);

            const char label[2] = {static_cast<char>('1' + static_cast<char>(i)), '\0'};
            overlay.addText({screen_pos.x - 4.0f, screen_pos.y - 6.0f},
                            label, toOverlay(t.overlay.text), label_size);
        }
    }

    void drawPointPickHud(lfs::rendering::ScreenOverlayRenderer& overlay, const PanelProjection& panel_proj,
                          std::string_view hint, const std::string& count_text, const std::string* status,
                          std::string_view option_text) {
        const auto& t = theme();
        const float ui_scale = t.fonts.base_size / 13.0f;
        const float info_x = panel_proj.info.x + 56.0f * ui_scale;
        const float hud_size = std::max(t.fonts.base_size, 14.0f * ui_scale);
        const float padding = 10.0f * ui_scale;
        const float max_width = std::max(40.0f, std::min(560.0f * ui_scale,
                                                         panel_proj.info.width - (info_x - panel_proj.info.x) - 2.0f * padding));
        const auto lines = wrapHint(overlay, hint, hud_size, max_width);
        const auto option_lines = option_text.empty() ? std::vector<std::string>{}
                                                      : wrapHint(overlay, option_text, hud_size, max_width);
        float width = overlay.measureText(count_text, t.fonts.large_size).x;
        for (const auto& line : lines)
            width = std::max(width, overlay.measureText(line, hud_size).x);
        const auto status_lines = status ? wrapHint(overlay, *status, hud_size, max_width)
                                         : std::vector<std::string>{};
        for (const auto& line : status_lines)
            width = std::max(width, overlay.measureText(line, hud_size).x);
        for (const auto& line : option_lines)
            width = std::max(width, overlay.measureText(line, hud_size).x);
        const float line_height = hud_size + 5.0f * ui_scale;
        const float height = t.fonts.large_size + 8.0f * ui_scale +
                             (lines.size() + option_lines.size() + status_lines.size()) * line_height;
        const glm::vec2 hud_pos(info_x, std::max(panel_proj.info.y + padding,
                                                 panel_proj.info.y + panel_proj.info.height - height - padding - 12.0f * ui_scale));
        overlay.addRectFilled(hud_pos - glm::vec2(padding),
                              hud_pos + glm::vec2(width + padding, height + padding),
                              toOverlay(t.overlay.background, 0.96f));
        overlay.addText(hud_pos, count_text.c_str(), toOverlay(t.overlay.text), t.fonts.large_size);
        glm::vec2 text_pos(hud_pos.x, hud_pos.y + t.fonts.large_size + 8.0f * ui_scale);
        for (const auto& line : lines) {
            overlay.addText(text_pos, line.c_str(), toOverlay(t.overlay.text), hud_size);
            text_pos.y += line_height;
        }
        for (const auto& line : option_lines) {
            overlay.addText(text_pos, line.c_str(), toOverlay(t.overlay.text), hud_size);
            text_pos.y += line_height;
        }
        for (const auto& line : status_lines) {
            overlay.addText(text_pos, line.c_str(), toOverlay(t.palette.warning), hud_size);
            text_pos.y += line_height;
        }
    }

    std::optional<glm::vec3> drawPointPickHover(lfs::rendering::ScreenOverlayRenderer& overlay,
                                                RenderingManager* rendering_manager,
                                                const PointPickOverlayFrame& frame,
                                                size_t point_count, size_t max_points, bool sample_hover,
                                                const lfs::rendering::OverlayColor& color, float label_size) {
        const auto& panel_proj = frame.panel;
        const auto& mouse_pos = frame.mouse_pos;
        const auto& t = theme();
        overlay.addCircle(mouse_pos, 5.0f, toOverlay(color, 0.8f), 16, 2.0f);
        std::optional<float> hover_depth;
        if (sample_hover && rendering_manager) {
            const glm::vec2 render_point = screenToRender(panel_proj, mouse_pos);
            const int depth_x = static_cast<int>(render_point.x);
            const int depth_y = static_cast<int>(render_point.y);
            const float depth = rendering_manager->getDepthAtPixel(rendering_manager->activeViewId(),
                                                                   depth_x,
                                                                   depth_y,
                                                                   frame.has_panel ? std::optional<SplitViewPanelId>(panel_proj.info.panel) : std::nullopt, true);
            if (depth > 0.0f && depth < 1e9f) {
                hover_depth = depth;
            }
        }

        if (hover_depth && point_count < max_points) {
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

                overlay.addCircleFilled(screen_pos, screen_radius, toOverlay(color, 0.6f), 32);
                overlay.addCircle(screen_pos, screen_radius, toOverlay(t.palette.text, 0.6f), 32, 1.5f);

                const char label[2] = {static_cast<char>('1' + static_cast<char>(point_count)), '\0'};
                overlay.addText({screen_pos.x - 4.0f, screen_pos.y - 6.0f},
                                label, toOverlay(t.palette.text, 0.7f), label_size);

                return preview_point;
            }
        }
        return std::nullopt;
    }

} // namespace lfs::vis::tools
