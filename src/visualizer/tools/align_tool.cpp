/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/align_tool.hpp"
#include "core/services.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering_manager.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

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

    static ImVec2 projectToScreen(const glm::vec3& world_pos, const Viewport& viewport) {
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix();
        const glm::vec4 clip_pos = proj * view * glm::vec4(world_pos, 1.0f);

        if (clip_pos.w <= 0.0f)
            return ImVec2(-1000, -1000);

        const glm::vec3 ndc = glm::vec3(clip_pos) / clip_pos.w;
        return ImVec2(
            (ndc.x * 0.5f + 0.5f) * viewport.windowSize.x,
            (1.0f - (ndc.y * 0.5f + 0.5f)) * viewport.windowSize.y);
    }

    static float calculateScreenRadius(const glm::vec3& world_pos, const float world_radius, const Viewport& viewport) {
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix();
        const glm::vec4 view_pos = view * glm::vec4(world_pos, 1.0f);
        const float depth = -view_pos.z;

        if (depth <= 0.0f)
            return 0.0f;

        const float screen_radius = (world_radius * proj[1][1] * viewport.windowSize.y) / (2.0f * depth);
        return glm::clamp(screen_radius, 5.0f, 50.0f);
    }

    void AlignTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                             [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_)
            return;

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 mouse_pos = ImGui::GetMousePos();
        const auto& viewport = tool_context_->getViewport();
        auto* const rendering_manager = tool_context_->getRenderingManager();
        const bool over_gui = ImGui::GetIO().WantCaptureMouse;

        constexpr float SPHERE_RADIUS = 0.05f;
        const auto& t = theme();
        const ImU32 SPHERE_COLOR = t.error_u32();
        const ImU32 SPHERE_OUTLINE = t.overlay_text_u32();
        const ImU32 PREVIEW_COLOR = toU32WithAlpha(t.palette.error, 0.6f);
        const ImU32 CROSSHAIR_COLOR = toU32WithAlpha(t.palette.error, 0.8f);

        // Get picked points from services
        const auto& picked_points = services().getAlignPickedPoints();

        // Draw picked points
        for (size_t i = 0; i < picked_points.size(); ++i) {
            const ImVec2 screen_pos = projectToScreen(picked_points[i], viewport);
            const float screen_radius = calculateScreenRadius(picked_points[i], SPHERE_RADIUS, viewport);

            draw_list->AddCircleFilled(screen_pos, screen_radius, SPHERE_COLOR, 32);
            draw_list->AddCircle(screen_pos, screen_radius, SPHERE_OUTLINE, 32, 1.5f);

            const char label = '1' + static_cast<char>(i);
            draw_list->AddText(ImVec2(screen_pos.x - 4, screen_pos.y - 6), t.overlay_text_u32(), &label, &label + 1);
        }

        if (over_gui)
            return;

        draw_list->AddCircle(mouse_pos, 5.0f, CROSSHAIR_COLOR, 16, 2.0f);

        // Live preview at mouse position
        if (picked_points.size() < 3 && rendering_manager) {
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(mouse_pos.x), static_cast<int>(mouse_pos.y));

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 preview_point = viewport.unprojectPixel(
                    mouse_pos.x, mouse_pos.y, depth, rendering_manager->getFocalLengthMm());
                if (preview_point.x > -1e9f) {
                    const ImVec2 screen_pos = projectToScreen(preview_point, viewport);
                    const float screen_radius = calculateScreenRadius(preview_point, SPHERE_RADIUS, viewport);

                    draw_list->AddCircleFilled(screen_pos, screen_radius, PREVIEW_COLOR, 32);
                    draw_list->AddCircle(screen_pos, screen_radius, toU32WithAlpha(t.palette.text, 0.6f), 32, 1.5f);

                    const char label = '1' + static_cast<char>(picked_points.size());
                    draw_list->AddText(ImVec2(screen_pos.x - 4, screen_pos.y - 6), toU32WithAlpha(t.palette.text, 0.7f), &label, &label + 1);
                }
            }
        }

        // Normal preview when 2 points picked
        if (picked_points.size() == 2 && rendering_manager) {
            const float depth = rendering_manager->getDepthAtPixel(
                static_cast<int>(mouse_pos.x), static_cast<int>(mouse_pos.y));

            if (depth > 0.0f && depth < 1e9f) {
                const glm::vec3 p2 = viewport.unprojectPixel(
                    mouse_pos.x, mouse_pos.y, depth, rendering_manager->getFocalLengthMm());
                if (p2.x > -1e9f) {
                    const glm::vec3& p0 = picked_points[0];
                    const glm::vec3& p1 = picked_points[1];

                    const glm::vec3 v01 = p1 - p0;
                    const glm::vec3 v02 = p2 - p0;
                    glm::vec3 normal = glm::normalize(glm::cross(v01, v02));
                    if (normal.y > 0.0f)
                        normal = -normal;

                    const glm::vec3 center = (p0 + p1 + p2) / 3.0f;
                    const float line_length = glm::max(glm::length(v01) * 0.5f, 0.1f);
                    const glm::vec3 normal_end = center + normal * line_length;

                    const ImVec2 center_screen = projectToScreen(center, viewport);
                    const ImVec2 normal_screen = projectToScreen(normal_end, viewport);

                    draw_list->AddLine(center_screen, normal_screen, IM_COL32(255, 255, 0, 255), 4.0f);
                    draw_list->AddCircleFilled(normal_screen, 10.0f, IM_COL32(255, 255, 0, 255));
                    draw_list->AddText(ImVec2(normal_screen.x + 12, normal_screen.y - 8), IM_COL32(255, 255, 0, 255), "UP");

                    const ImVec2 p0_screen = projectToScreen(p0, viewport);
                    const ImVec2 p1_screen = projectToScreen(p1, viewport);
                    const ImVec2 p2_screen = projectToScreen(p2, viewport);
                    draw_list->AddLine(p0_screen, p1_screen, IM_COL32(255, 0, 0, 200), 2.0f);
                    draw_list->AddLine(p1_screen, p2_screen, IM_COL32(0, 255, 0, 200), 2.0f);
                    draw_list->AddLine(p2_screen, p0_screen, IM_COL32(0, 0, 255, 200), 2.0f);
                }
            }
        }

        // Instructions
        const char* instruction = nullptr;
        switch (picked_points.size()) {
        case 0: instruction = "Click 1st point"; break;
        case 1: instruction = "Click 2nd point"; break;
        case 2: instruction = "Click 3rd point"; break;
        default: break;
        }
        if (instruction) {
            draw_list->AddText(ImVec2(mouse_pos.x + 15, mouse_pos.y - 10), CROSSHAIR_COLOR, instruction);
        }

        char count_text[16];
        snprintf(count_text, sizeof(count_text), "Points: %zu/3", picked_points.size());
        draw_list->AddText(ImVec2(10, 50), t.overlay_text_u32(), count_text);
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
