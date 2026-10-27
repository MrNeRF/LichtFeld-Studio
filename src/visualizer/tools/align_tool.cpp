/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/align_tool.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/number_format.hpp"
#include "core/services.hpp"
#include "gui/string_keys.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/align_ops.hpp"
#include "point_pick_overlay.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include <algorithm>
#include <cstdio>

namespace lfs::vis::tools {

    AlignTool::AlignTool() = default;

    bool AlignTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void AlignTool::shutdown() {
        setEnabled(false);
        if (op::operators().activeModalId() == op::to_string(op::BuiltinOp::AlignPickPoint)) {
            op::operators().cancelModalOperator();
        }
        restoreGridIfNeeded();
        tool_context_ = nullptr;
        services().alignPointPick().clearPickedPoints();
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
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
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
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }
    }

    void AlignTool::update(const ToolContext& ctx) {
        if (isEnabled() && op::operators().activeModalId() == op::to_string(op::BuiltinOp::AlignPickPoint)) {
            op::ModalEvent event{};
            event.type = op::ModalEvent::Type::NONE;
            op::operators().dispatchModalEvent(event);
        }
        auto* const rm = ctx.getRenderingManager();
        const bool has_status = services().alignPointPick().getStatusMessage() != nullptr;
        if (had_align_status_ && !has_status && rm) {
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
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

        void drawEdgeLengthLabels(lfs::rendering::ScreenOverlayRenderer& overlay,
                                  const PanelProjection& panel_proj,
                                  const std::vector<glm::vec3>& points,
                                  const lfs::rendering::OverlayColor& text_color,
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
                drawLabel(overlay, label_pos, buf, text_color, label_size);
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
            if (services().getAlignPreviewEnabled()) {
                if (normal.y < 0.0f)
                    normal = -normal;
            } else {
                op::faceNormalTowardCamera(normal, center, camera_pos);
            }

            bool snapped = false;
            if (services().getAlignAxisSnapEnabled() && snap_target_world) {
                snapped = op::snapAlignNormalToNodeAxes(normal, *snap_target_world);
            }

            const float line_length = glm::max(glm::length(v01) * 0.5f, 0.1f);
            const glm::vec3 normal_end = center + normal * line_length;

            if (!projectToScreenChecked(panel_proj, p0).valid ||
                !projectToScreenChecked(panel_proj, p1).valid ||
                !projectToScreenChecked(panel_proj, p2).valid ||
                !projectToScreenChecked(panel_proj, normal_end).valid) {
                return;
            }
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
            drawLabel(overlay, {normal_screen.x + 12.0f, normal_screen.y - 8.0f},
                      LOC(lichtfeld::Strings::Align::UP), up_color, label_size);
            if (snapped) {
                drawLabel(overlay, {normal_screen.x + 12.0f, normal_screen.y + label_size + 2.0f},
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
                drawLabel(overlay, {x_screen.x + 10.0f, x_screen.y - 6.0f}, "X", X_COLOR, label_size);
            }

            overlay.addLine(p0_screen, p1_screen, TRI_RED, 2.0f);
            overlay.addLine(p1_screen, p2_screen, TRI_GREEN, 2.0f);
            overlay.addLine(p2_screen, p0_screen, TRI_BLUE, 2.0f);

            drawEdgeLengthLabels(overlay, panel_proj, {p0, p1, p2},
                                 toOverlay(t.overlay.text), label_size);
        }
    } // namespace

    void AlignTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                             [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_)
            return;

        auto* const overlay = getOverlayRenderer(*tool_context_);
        if (!overlay || !overlay->isFrameActive())
            return;

        const auto frame = pointPickOverlayFrame(*tool_context_);
        const auto& panel_proj = frame.panel;
        const auto& mouse_pos = frame.mouse_pos;
        const auto& bounds = tool_context_->getViewportBounds();
        auto* const rendering_manager = tool_context_->getRenderingManager();

        const lfs::rendering::ScreenOverlayRenderer::ScopedClipRect clip(
            *overlay,
            {bounds.x, bounds.y},
            {bounds.x + bounds.width, bounds.y + bounds.height});

        const auto& t = theme();
        const float ui_scale = t.fonts.base_size / 13.0f;
        const auto SPHERE_COLOR = toOverlay(t.palette.error);
        const float label_size = std::max(t.fonts.base_size, 14.0f * ui_scale);

        const auto& picked_points = services().alignPointPick().getPickedPoints();
        const auto selected_point = services().alignPointPick().getSelectedPoint();
        const bool in_review = picked_points.size() == 3;
        const glm::vec3 camera_pos = in_review ? services().getAlignCameraPosition()
                                               : panel_proj.viewport.camera.t;
        auto* const sm = tool_context_->getSceneManager();
        const auto snap_target_world =
            sm ? op::resolveAlignSnapTargetWorld(*sm) : std::optional<glm::mat4>{};

        drawPointMarkers(*overlay, panel_proj, picked_points, selected_point, SPHERE_COLOR, label_size);

        if (in_review) {
            drawTrianglePreview(*overlay, panel_proj,
                                picked_points[0], picked_points[1], picked_points[2],
                                camera_pos, label_size, true, snap_target_world);
        }

        const auto cancel_key = pointPickShortcut(input::Action::CANCEL_POLYGON);
        const auto hint = services().getAlignPreviewEnabled()
                              ? LOCF("align.hint_preview", cancel_key)
                          : in_review ? LOCF(lichtfeld::Strings::Align::HINT_REVIEW, cancel_key)
                                      : LOCF(lichtfeld::Strings::Align::HINT_PICKING,
                                             pointPickShortcut(input::Action::DELETE_SELECTED),
                                             pointPickShortcut(input::Action::REMOVE_LAST_POINT), cancel_key);
        drawPointPickHud(*overlay, panel_proj, hint,
                         LOCF(lichtfeld::Strings::Align::POINTS_COUNT, lfs::core::format_count(picked_points.size())),
                         services().alignPointPick().getStatusMessage(),
                         in_review ? LOC(services().getAlignEdgeToAxisEnabled() ? "align.edge_on_help" : "align.edge_off_help") : "");

        if (frame.over_gui)
            return;

        bool drew_live_triangle = false;
        if (const auto hover = drawPointPickHover(*overlay, rendering_manager, frame,
                                                  picked_points.size(), 3, !in_review,
                                                  SPHERE_COLOR, label_size)) {
            const auto& preview_point = *hover;
            if (picked_points.size() == 2) {
                drawTrianglePreview(*overlay, panel_proj,
                                    picked_points[0], picked_points[1], preview_point,
                                    camera_pos, label_size, false, snap_target_world);
                drew_live_triangle = true;
            }
        }

        if (!in_review && !drew_live_triangle && picked_points.size() == 2) {
            drawEdgeLengthLabels(*overlay, panel_proj,
                                 {picked_points[0], picked_points[1]},
                                 toOverlay(t.overlay.text), label_size);
        }

        const char* instruction_key = nullptr;
        switch (picked_points.size()) {
        case 0: instruction_key = lichtfeld::Strings::Align::CLICK_1ST; break;
        case 1: instruction_key = lichtfeld::Strings::Align::CLICK_2ND; break;
        case 2: instruction_key = lichtfeld::Strings::Align::CLICK_3RD; break;
        default: break;
        }
        if (instruction_key) {
            drawLabel(*overlay, {mouse_pos.x + 15.0f, mouse_pos.y - 10.0f},
                      LOC(instruction_key), toOverlay(t.overlay.text), label_size);
        }
    }

    void AlignTool::onEnabledChanged(bool enabled) {
        if (tool_context_ && tool_context_->getRenderingManager())
            tool_context_->getRenderingManager()->setExactSurfacePicking(enabled);
        if (enabled) {
            forceGridOn();
        } else {
            if (op::operators().activeModalId() == op::to_string(op::BuiltinOp::AlignPickPoint)) {
                op::operators().cancelModalOperator();
            }
            restoreGridIfNeeded();
            services().alignPointPick().clearPickedPoints();
        }
        if (tool_context_) {
            tool_context_->requestRender();
        }
    }

} // namespace lfs::vis::tools
