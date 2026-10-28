/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/scale_distance_tool.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/number_format.hpp"
#include "core/services.hpp"
#include "gui/string_keys.hpp"
#include "operator/operator_registry.hpp"
#include "point_pick_overlay.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "theme/theme.hpp"
#include <algorithm>
#include <cstdio>

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
        services().scaleDistancePointPick().clearPickedPoints();
    }

    void ScaleDistanceTool::update(const ToolContext& ctx) {
        if (isEnabled() && op::operators().activeModalId() == op::to_string(op::BuiltinOp::ScaleDistancePickPoint)) {
            op::ModalEvent event{};
            event.type = op::ModalEvent::Type::NONE;
            op::operators().dispatchModalEvent(event);
        }
        auto* const rm = ctx.getRenderingManager();
        const bool has_status = services().scaleDistancePointPick().getStatusMessage() != nullptr;
        if (had_status_ && !has_status && rm) {
            rm->markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }
        had_status_ = has_status;
    }

    void ScaleDistanceTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
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
        const auto SPHERE_COLOR = toOverlay(t.palette.info);
        constexpr lfs::rendering::OverlayColor LINE_COLOR{1.0f, 1.0f, 1.0f, 0.95f};
        const float label_size = std::max(t.fonts.base_size, 14.0f * ui_scale);

        const auto& picked_points = services().scaleDistancePointPick().getPickedPoints();
        const auto selected_point = services().scaleDistancePointPick().getSelectedPoint();

        // Draw placed markers
        drawPointMarkers(*overlay, panel_proj, picked_points, selected_point, SPHERE_COLOR, label_size);

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
        const auto cancel_key = pointPickShortcut(input::Action::CANCEL_POLYGON);
        const auto hint = (picked_points.size() < 2)
                              ? LOCF(lichtfeld::Strings::ScaleDistance::HINT_PICKING,
                                     pointPickShortcut(input::Action::DELETE_SELECTED),
                                     cancel_key)
                              : LOCF(lichtfeld::Strings::ScaleDistance::HINT_REVIEW, cancel_key);
        drawPointPickHud(*overlay, panel_proj, hint,
                         LOCF(lichtfeld::Strings::ScaleDistance::POINTS_COUNT, lfs::core::format_count(picked_points.size())),
                         services().scaleDistancePointPick().getStatusMessage(),
                         "");

        if (frame.over_gui)
            return;

        if (const auto hover = drawPointPickHover(*overlay, rendering_manager, frame,
                                                  picked_points.size(), 2, picked_points.size() < 2,
                                                  SPHERE_COLOR, label_size)) {
            const auto& preview_point = *hover;
            const auto screen_pos = projectToScreen(panel_proj, preview_point);
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
            services().scaleDistancePointPick().clearPickedPoints();
        }
        if (tool_context_) {
            tool_context_->requestRender();
        }
    }

} // namespace lfs::vis::tools
