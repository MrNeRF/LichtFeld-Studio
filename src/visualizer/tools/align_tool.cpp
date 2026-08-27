/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/align_tool.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/string_keys.hpp"
#include "internal/viewport.hpp"
#include "operator/ops/align_ops.hpp"
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

    AlignTool::AlignTool() = default;

    bool AlignTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void AlignTool::shutdown() {
        restoreGridIfNeeded();
        tool_context_ = nullptr;
        services().clearAlignPickedPoints();
    }

    void AlignTool::forceGridOn() {
        if (!tool_context_) {
            return;
        }
        auto* const rm = tool_context_->getRenderingManager();
        if (!rm) {
            return;
        }
        auto settings = rm->getSettings();
        saved_show_grid_ = settings.show_grid;
        user_changed_grid_ = false;
        if (!settings.show_grid) {
            settings.show_grid = true;
            rm->updateSettings(settings);
            rm->markDirty(DirtyFlag::OVERLAY);
        }
        grid_override_active_ = true;
    }

    void AlignTool::restoreGridIfNeeded() {
        if (!grid_override_active_) {
            return;
        }
        grid_override_active_ = false;
        if (user_changed_grid_ || !tool_context_) {
            return;
        }
        auto* const rm = tool_context_->getRenderingManager();
        if (!rm) {
            return;
        }
        auto settings = rm->getSettings();
        if (settings.show_grid != saved_show_grid_) {
            settings.show_grid = saved_show_grid_;
            rm->updateSettings(settings);
            rm->markDirty(DirtyFlag::OVERLAY);
        }
    }

    void AlignTool::update(const ToolContext& ctx) {
        auto* const rm = ctx.getRenderingManager();
        const bool has_status = services().getAlignStatusMessage() != nullptr;
        if (had_align_status_ && !has_status && rm) {
            rm->markDirty(DirtyFlag::OVERLAY);
        }
        had_align_status_ = has_status;

        if (!isEnabled() || !grid_override_active_ || user_changed_grid_) {
            return;
        }
        if (!rm) {
            return;
        }
        if (!rm->getSettings().show_grid) {
            user_changed_grid_ = true;
        }
    }

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
            proj.ortho_scale = settings.ortho_scale;

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

        void drawEdgeLengthLabels(lfs::rendering::ScreenOverlayRenderer& overlay,
                                  const PanelProjection& panel_proj,
                                  const std::vector<glm::vec3>& points,
                                  const lfs::rendering::OverlayColor& text_color,
                                  const lfs::rendering::OverlayColor& shadow_color,
                                  const float label_size) {
            if (points.size() < 2) {
                return;
            }

            std::vector<ProjectedScreen> screens;
            screens.reserve(points.size());
            for (const auto& p : points) {
                screens.push_back(projectToScreenChecked(panel_proj, p));
            }

            glm::vec2 centroid_screen(0.0f);
            int valid_count = 0;
            for (const auto& s : screens) {
                if (s.valid) {
                    centroid_screen += s.pos;
                    ++valid_count;
                }
            }
            if (valid_count > 0) {
                centroid_screen /= static_cast<float>(valid_count);
            }

            const size_t n = points.size();
            for (size_t i = 0; i < n; ++i) {
                const size_t j = (i + 1) % n;
                if (!screens[i].valid || !screens[j].valid) {
                    continue;
                }
                if (n == 2 && i > 0) {
                    break;
                }

                const glm::vec2 mid = (screens[i].pos + screens[j].pos) * 0.5f;
                glm::vec2 outward = mid - centroid_screen;
                const float outward_len = glm::length(outward);
                if (outward_len > 1e-3f) {
                    outward = outward / outward_len * 10.0f;
                } else {
                    outward = glm::vec2(0.0f, -10.0f);
                }
                const glm::vec2 label_pos = mid + outward;
                const float len = glm::length(points[j] - points[i]);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.2f", static_cast<double>(len));
                overlay.addTextWithShadow(label_pos, buf, text_color, shadow_color, label_size);
            }
        }

        void drawTrianglePreview(lfs::rendering::ScreenOverlayRenderer& overlay,
                                 const PanelProjection& panel_proj,
                                 const glm::vec3& p0,
                                 const glm::vec3& p1,
                                 const glm::vec3& p2,
                                 const glm::vec3& camera_pos,
                                 const float label_size,
                                 const bool filled,
                                 const std::optional<glm::mat4>& snap_target_world) {
            const glm::vec3 v01 = p1 - p0;
            const glm::vec3 v02 = p2 - p0;
            const glm::vec3 cross_v = glm::cross(v01, v02);
            const float cross_len = glm::length(cross_v);
            if (cross_len <= 1e-6f) {
                return;
            }

            glm::vec3 normal = cross_v / cross_len;
            const glm::vec3 center = (p0 + p1 + p2) / 3.0f;
            op::faceNormalTowardCamera(normal, center, camera_pos);

            bool snapped = false;
            if (services().getAlignAxisSnapEnabled() && snap_target_world) {
                snapped = op::snapAlignNormalToNodeAxes(normal, *snap_target_world);
            }

            const float line_length = glm::max(glm::length(v01) * 0.5f, 0.1f);
            const glm::vec3 normal_end = center + normal * line_length;

            const glm::vec2 center_screen = projectToScreen(panel_proj, center);
            const glm::vec2 normal_screen = projectToScreen(panel_proj, normal_end);
            const glm::vec2 p0_screen = projectToScreen(panel_proj, p0);
            const glm::vec2 p1_screen = projectToScreen(panel_proj, p1);
            const glm::vec2 p2_screen = projectToScreen(panel_proj, p2);

            constexpr lfs::rendering::OverlayColor YELLOW{1.0f, 1.0f, 0.0f, 1.0f};
            constexpr lfs::rendering::OverlayColor TRI_RED{1.0f, 0.0f, 0.0f, 200.0f / 255.0f};
            constexpr lfs::rendering::OverlayColor TRI_GREEN{0.0f, 1.0f, 0.0f, 200.0f / 255.0f};
            constexpr lfs::rendering::OverlayColor TRI_BLUE{0.0f, 0.0f, 1.0f, 200.0f / 255.0f};

            const auto& t = theme();
            const auto up_color = snapped ? toOverlay(t.palette.primary) : YELLOW;

            if (filled) {
                overlay.addTriangleFilled(p0_screen, p1_screen, p2_screen, toOverlay(t.palette.info, 0.15f));
            }

            overlay.addLine(center_screen, normal_screen, up_color, 4.0f);
            overlay.addCircleFilled(normal_screen, 10.0f, up_color);
            overlay.addText({normal_screen.x + 12.0f, normal_screen.y - 8.0f},
                            LOC(lichtfeld::Strings::Align::UP), up_color, label_size);
            if (snapped) {
                overlay.addText({normal_screen.x + 12.0f, normal_screen.y + label_size + 2.0f},
                                LOC(lichtfeld::Strings::Align::SNAPPED), up_color, label_size);
            }

            if (services().getAlignEdgeToAxisEnabled()) {
                // Target direction for the first edge after apply: world +X from centroid.
                const float tick_len = glm::max(line_length * 0.75f, 0.08f);
                const glm::vec3 x_end = center + glm::vec3(tick_len, 0.0f, 0.0f);
                const glm::vec2 x_screen = projectToScreen(panel_proj, x_end);
                constexpr lfs::rendering::OverlayColor X_COLOR{1.0f, 0.35f, 0.35f, 1.0f};
                overlay.addLine(center_screen, x_screen, X_COLOR, 3.0f);
                overlay.addCircleFilled(x_screen, 7.0f, X_COLOR);
                overlay.addText({x_screen.x + 10.0f, x_screen.y - 6.0f}, "X", X_COLOR, label_size);
            }

            overlay.addLine(p0_screen, p1_screen, TRI_RED, 2.0f);
            overlay.addLine(p1_screen, p2_screen, TRI_GREEN, 2.0f);
            overlay.addLine(p2_screen, p0_screen, TRI_BLUE, 2.0f);

            constexpr lfs::rendering::OverlayColor kShadow{0.0f, 0.0f, 0.0f, 180.0f / 255.0f};
            drawEdgeLengthLabels(overlay, panel_proj, {p0, p1, p2},
                                 toOverlay(t.overlay.text), kShadow, label_size);
        }
    } // namespace

    [[nodiscard]] static float markerScreenRadius(const glm::vec3& world_pos, const PanelProjection& panel_proj) {
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
        if (rendering_manager) {
            const auto settings = rendering_manager->getSettings();
            panel_proj_fallback.focal_length_mm = settings.focal_length_mm;
            panel_proj_fallback.orthographic = settings.orthographic;
            panel_proj_fallback.ortho_scale = settings.ortho_scale;
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

        constexpr lfs::rendering::OverlayColor kShadow{0.0f, 0.0f, 0.0f, 180.0f / 255.0f};
        const auto& t = theme();
        const float ui_scale = t.fonts.base_size / 13.0f;
        const float info_x = bounds.x + 56.0f * ui_scale;
        const auto SPHERE_COLOR = toOverlay(t.palette.error);
        const auto SPHERE_OUTLINE = toOverlay(t.overlay.text);
        const auto SELECTED_OUTLINE = toOverlay(t.palette.primary);
        const auto PREVIEW_COLOR = toOverlay(t.palette.error, 0.6f);
        const auto CROSSHAIR_COLOR = toOverlay(t.palette.error, 0.8f);
        const float label_size = t.fonts.base_size;

        const auto& picked_points = services().getAlignPickedPoints();
        const auto selected_point = services().getAlignSelectedPoint();
        const bool in_review = picked_points.size() == 3;
        const glm::vec3 camera_pos = panel_proj.viewport.camera.t;
        auto* const sm = tool_context_->getSceneManager();
        const auto snap_target_world =
            sm ? op::resolveAlignSnapTargetWorld(*sm) : std::optional<glm::mat4>{};

        for (size_t i = 0; i < picked_points.size(); ++i) {
            const glm::vec2 screen_pos = projectToScreen(panel_proj, picked_points[i]);
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

        if (in_review) {
            drawTrianglePreview(*overlay, panel_proj,
                                picked_points[0], picked_points[1], picked_points[2],
                                camera_pos, label_size, true, snap_target_world);
        }

        if (over_gui)
            return;

        overlay->addCircle(mouse_pos, 5.0f, CROSSHAIR_COLOR, 16, 2.0f);

        // Single depth sample per frame for hover preview + live triangle (B7).
        std::optional<float> hover_depth;
        if (!in_review && rendering_manager) {
            const glm::vec2 render_point = screenToRender(panel_proj, mouse_pos);
            const int depth_x = static_cast<int>(render_point.x);
            const int depth_y = static_cast<int>(render_point.y);
            const float depth = rendering_manager->getDepthAtPixel(
                depth_x,
                depth_y,
                panel_proj_opt ? std::optional<SplitViewPanelId>(panel_proj.info.panel) : std::nullopt);
            if (depth > 0.0f && depth < 1e9f) {
                hover_depth = depth;
            }
        }

        bool drew_live_triangle = false;
        if (hover_depth && picked_points.size() < 3) {
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

                if (picked_points.size() == 2) {
                    drawTrianglePreview(*overlay, panel_proj,
                                        picked_points[0], picked_points[1], preview_point,
                                        camera_pos, label_size, false, snap_target_world);
                    drew_live_triangle = true;
                }
            }
        }

        if (!in_review && !drew_live_triangle && picked_points.size() == 2) {
            drawEdgeLengthLabels(*overlay, panel_proj,
                                 {picked_points[0], picked_points[1]},
                                 toOverlay(t.overlay.text), kShadow, label_size);
        }

        const char* instruction_key = nullptr;
        switch (picked_points.size()) {
        case 0: instruction_key = lichtfeld::Strings::Align::CLICK_1ST; break;
        case 1: instruction_key = lichtfeld::Strings::Align::CLICK_2ND; break;
        case 2: instruction_key = lichtfeld::Strings::Align::CLICK_3RD; break;
        default: break;
        }
        if (instruction_key) {
            overlay->addTextWithShadow({mouse_pos.x + 15.0f, mouse_pos.y - 10.0f},
                                       LOC(instruction_key), toOverlay(t.overlay.text), kShadow, label_size);
        }

        const std::size_t point_count = picked_points.size();
        const std::string count_text = LOCF(lichtfeld::Strings::Align::POINTS_COUNT, point_count);
        overlay->addTextWithShadow({info_x, bounds.y + 40.0f},
                                   count_text.c_str(), toOverlay(t.overlay.text), kShadow,
                                   t.fonts.large_size);

        const char* const hint_key = in_review ? lichtfeld::Strings::Align::HINT_REVIEW
                                               : lichtfeld::Strings::Align::HINT_PICKING;
        overlay->addTextWithShadow({info_x, bounds.y + 40.0f + t.fonts.large_size + 6.0f},
                                   LOC(hint_key), toOverlay(t.overlay.text, 0.85f), kShadow,
                                   t.fonts.base_size);

        if (const std::string* const status = services().getAlignStatusMessage()) {
            overlay->addTextWithShadow({info_x, bounds.y + 40.0f + 2.0f * t.fonts.large_size + 12.0f},
                                       status->c_str(), toOverlay(t.palette.warning, 0.95f), kShadow,
                                       t.fonts.base_size);
        }
    }

    void AlignTool::onEnabledChanged(bool enabled) {
        if (enabled) {
            forceGridOn();
        } else {
            restoreGridIfNeeded();
            services().clearAlignPickedPoints();
        }
        if (tool_context_) {
            tool_context_->requestRender();
        }
    }

} // namespace lfs::vis::tools
