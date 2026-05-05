/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/align_tool.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "theme/theme.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <optional>
#include <string_view>

namespace lfs::vis::tools {

    AlignTool::AlignTool() = default;

    bool AlignTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void AlignTool::shutdown() {
        tool_context_ = nullptr;
        services().clearAlignPickedPoints();
    }

    void AlignTool::update([[maybe_unused]] const ToolContext& ctx) {}

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

        struct PanelProjection {
            lfs::vis::RenderingManager::ViewerPanelInfo info{};
            Viewport viewport;
            float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
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
            proj.focal_length_mm = rm->getFocalLengthMm();
            if (proj.focal_length_mm <= 0.0f) {
                proj.focal_length_mm = fallback_focal_length_mm;
            }

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

        [[nodiscard]] glm::vec2 projectToScreen(const PanelProjection& proj, const glm::vec3& world_pos) {
            const auto projected = lfs::rendering::projectWorldPoint(
                proj.viewport.camera.R,
                proj.viewport.camera.t,
                proj.viewport.windowSize,
                world_pos,
                proj.focal_length_mm);
            if (!projected) {
                return {-1000.0f, -1000.0f};
            }

            return renderToScreen(proj, glm::vec2(projected->x, projected->y));
        }
    } // namespace

    static float calculateScreenRadius(const glm::vec3& world_pos,
                                       const float world_radius,
                                       const Viewport& viewport,
                                       const float focal_length_mm) {
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix(focal_length_mm);
        const glm::vec4 view_pos = view * glm::vec4(world_pos, 1.0f);
        const float depth = -view_pos.z;

        if (depth <= 0.0f)
            return 0.0f;

        const float screen_radius = (world_radius * proj[1][1] * viewport.windowSize.y) / (2.0f * depth);
        return screen_radius;
    }

    void AlignTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                             [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_)
            return;

        auto* const overlay = getOverlayRenderer(*tool_context_);
        if (!overlay || !overlay->isFrameActive())
            return;

        float mx = 0.0f;
        float my = 0.0f;
        SDL_GetMouseState(&mx, &my);
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
        panel_proj_fallback.viewport = tool_context_->getViewport();
        panel_proj_fallback.viewport.windowSize = {fallback_render_width, fallback_render_height};
        panel_proj_fallback.screen_scale_x = bounds.width / static_cast<float>(fallback_render_width);
        panel_proj_fallback.screen_scale_y = bounds.height / static_cast<float>(fallback_render_height);

        const PanelProjection& panel_proj = panel_proj_opt ? *panel_proj_opt : panel_proj_fallback;

        const lfs::rendering::ScreenOverlayRenderer::ScopedClipRect clip(
            *overlay,
            {bounds.x, bounds.y},
            {bounds.x + bounds.width, bounds.y + bounds.height});

        constexpr float SPHERE_RADIUS = 0.05f;
        constexpr lfs::rendering::OverlayColor kShadow{0.0f, 0.0f, 0.0f, 180.0f / 255.0f};
        const auto& t = theme();
        const auto SPHERE_COLOR = toOverlay(t.palette.error);
        const auto SPHERE_OUTLINE = toOverlay(t.overlay.text);
        const auto PREVIEW_COLOR = toOverlay(t.palette.error, 0.6f);
        const auto CROSSHAIR_COLOR = toOverlay(t.palette.error, 0.8f);
        const float label_size = t.fonts.base_size;

        const auto& picked_points = services().getAlignPickedPoints();

        for (size_t i = 0; i < picked_points.size(); ++i) {
            const glm::vec2 screen_pos = projectToScreen(panel_proj, picked_points[i]);
            const float radius_render = calculateScreenRadius(
                picked_points[i], SPHERE_RADIUS, panel_proj.viewport, panel_proj.focal_length_mm);
            const float screen_radius =
                glm::clamp(radius_render * glm::min(panel_proj.screen_scale_x, panel_proj.screen_scale_y), 5.0f, 50.0f);

            overlay->addCircleFilled(screen_pos, screen_radius, SPHERE_COLOR, 32);
            overlay->addCircle(screen_pos, screen_radius, SPHERE_OUTLINE, 32, 1.5f);

            const char label[2] = {static_cast<char>('1' + static_cast<char>(i)), '\0'};
            overlay->addText({screen_pos.x - 4.0f, screen_pos.y - 6.0f},
                             label, toOverlay(t.overlay.text), label_size);
        }

        if (over_gui)
            return;

        overlay->addCircle(mouse_pos, 5.0f, CROSSHAIR_COLOR, 16, 2.0f);

        if (picked_points.size() < 3 && rendering_manager) {
            const glm::vec2 render_point = screenToRender(panel_proj, mouse_pos);
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(render_point.x),
                static_cast<int>(render_point.y),
                panel_proj_opt ? std::optional<SplitViewPanelId>(panel_proj.info.panel) : std::nullopt);

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 preview_point = panel_proj.viewport.unprojectPixel(
                    render_point.x, render_point.y, depth, panel_proj.focal_length_mm);
                if (Viewport::isValidWorldPosition(preview_point)) {
                    const glm::vec2 screen_pos = projectToScreen(panel_proj, preview_point);
                    const float radius_render = calculateScreenRadius(
                        preview_point, SPHERE_RADIUS, panel_proj.viewport, panel_proj.focal_length_mm);
                    const float screen_radius = glm::clamp(
                        radius_render * glm::min(panel_proj.screen_scale_x, panel_proj.screen_scale_y), 5.0f, 50.0f);

                    overlay->addCircleFilled(screen_pos, screen_radius, PREVIEW_COLOR, 32);
                    overlay->addCircle(screen_pos, screen_radius, toOverlay(t.palette.text, 0.6f), 32, 1.5f);

                    const char label[2] = {static_cast<char>('1' + static_cast<char>(picked_points.size())), '\0'};
                    overlay->addText({screen_pos.x - 4.0f, screen_pos.y - 6.0f},
                                     label, toOverlay(t.palette.text, 0.7f), label_size);
                }
            }
        }

        if (picked_points.size() == 2 && rendering_manager) {
            const glm::vec2 render_point = screenToRender(panel_proj, mouse_pos);
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(render_point.x),
                static_cast<int>(render_point.y),
                panel_proj_opt ? std::optional<SplitViewPanelId>(panel_proj.info.panel) : std::nullopt);

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 p2 = panel_proj.viewport.unprojectPixel(
                    render_point.x, render_point.y, depth, panel_proj.focal_length_mm);
                if (Viewport::isValidWorldPosition(p2)) {
                    const glm::vec3& p0 = picked_points[0];
                    const glm::vec3& p1 = picked_points[1];

                    const glm::vec3 v01 = p1 - p0;
                    const glm::vec3 v02 = p2 - p0;
                    glm::vec3 normal = glm::normalize(glm::cross(v01, v02));
                    constexpr glm::vec3 kTargetUp(0.0f, 1.0f, 0.0f);
                    if (glm::dot(normal, kTargetUp) < 0.0f)
                        normal = -normal;

                    const glm::vec3 center = (p0 + p1 + p2) / 3.0f;
                    const float line_length = glm::max(glm::length(v01) * 0.5f, 0.1f);
                    const glm::vec3 normal_end = center + normal * line_length;

                    const glm::vec2 center_screen = projectToScreen(panel_proj, center);
                    const glm::vec2 normal_screen = projectToScreen(panel_proj, normal_end);

                    constexpr lfs::rendering::OverlayColor YELLOW{1.0f, 1.0f, 0.0f, 1.0f};
                    constexpr lfs::rendering::OverlayColor TRI_RED{1.0f, 0.0f, 0.0f, 200.0f / 255.0f};
                    constexpr lfs::rendering::OverlayColor TRI_GREEN{0.0f, 1.0f, 0.0f, 200.0f / 255.0f};
                    constexpr lfs::rendering::OverlayColor TRI_BLUE{0.0f, 0.0f, 1.0f, 200.0f / 255.0f};

                    overlay->addLine(center_screen, normal_screen, YELLOW, 4.0f);
                    overlay->addCircleFilled(normal_screen, 10.0f, YELLOW);
                    overlay->addText({normal_screen.x + 12.0f, normal_screen.y - 8.0f},
                                     "UP", YELLOW, label_size);

                    const glm::vec2 p0_screen = projectToScreen(panel_proj, p0);
                    const glm::vec2 p1_screen = projectToScreen(panel_proj, p1);
                    const glm::vec2 p2_screen = projectToScreen(panel_proj, p2);
                    overlay->addLine(p0_screen, p1_screen, TRI_RED, 2.0f);
                    overlay->addLine(p1_screen, p2_screen, TRI_GREEN, 2.0f);
                    overlay->addLine(p2_screen, p0_screen, TRI_BLUE, 2.0f);
                }
            }
        }

        const char* instruction = nullptr;
        switch (picked_points.size()) {
        case 0: instruction = "Click 1st point"; break;
        case 1: instruction = "Click 2nd point"; break;
        case 2: instruction = "Click 3rd point"; break;
        default: break;
        }
        if (instruction) {
            overlay->addText({mouse_pos.x + 15.0f, mouse_pos.y - 10.0f},
                             instruction, CROSSHAIR_COLOR, label_size);
        }

        char count_text[16];
        snprintf(count_text, sizeof(count_text), "Points: %zu/3", picked_points.size());
        overlay->addTextWithShadow({bounds.x + 10.0f, bounds.y + 40.0f},
                                   count_text, toOverlay(t.overlay.text), kShadow,
                                   t.fonts.large_size);
    }

    void AlignTool::onEnabledChanged(bool enabled) {
        if (!enabled) {
            services().clearAlignPickedPoints();
        }
        if (tool_context_) {
            tool_context_->requestRender();
        }
    }

} // namespace lfs::vis::tools
