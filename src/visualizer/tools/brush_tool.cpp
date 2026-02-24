/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/brush_tool.hpp"
#include "rendering/rendering_manager.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::tools {

    BrushTool::BrushTool() = default;

    bool BrushTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void BrushTool::shutdown() {
        tool_context_ = nullptr;
    }

    void BrushTool::update([[maybe_unused]] const ToolContext& ctx) {
        if (isEnabled()) {
            const ImVec2 mouse = ImGui::GetMousePos();
            last_mouse_pos_ = glm::vec2(mouse.x, mouse.y);
        }
    }

    void BrushTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                             [[maybe_unused]] bool* p_open) {
        if (!isEnabled() || !tool_context_ || ImGui::GetIO().WantCaptureMouse)
            return;

        const auto& t = theme();
        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const auto& bounds = tool_context_->getViewportBounds();
        draw_list->PushClipRect({bounds.x, bounds.y},
                                {bounds.x + bounds.width, bounds.y + bounds.height}, true);
        const ImVec2 mouse_pos(last_mouse_pos_.x, last_mouse_pos_.y);

        // Selection mode uses primary color, saturation uses warning (orange)
        const ImU32 brush_color = (current_mode_ == BrushMode::Select)
                                      ? t.selection_border_u32()
                                      : t.polygon_vertex_u32();

        draw_list->AddCircle(mouse_pos, brush_radius_, brush_color, 32, 2.0f);
        draw_list->AddCircleFilled(mouse_pos, 3.0f, brush_color);

        // Show mode and value next to circle
        static char info_text[32];
        if (current_mode_ == BrushMode::Select) {
            snprintf(info_text, sizeof(info_text), "SEL");
        } else {
            snprintf(info_text, sizeof(info_text), "SAT %+.0f%%", saturation_amount_ * 100.0f);
        }

        const ImVec2 text_pos(mouse_pos.x + brush_radius_ + 10, mouse_pos.y - t.fonts.heading_size / 2);
        draw_list->AddText(ImGui::GetFont(), t.fonts.heading_size, ImVec2(text_pos.x + 1, text_pos.y + 1), t.overlay_shadow_u32(), info_text);
        draw_list->AddText(ImGui::GetFont(), t.fonts.heading_size, text_pos, t.overlay_text_u32(), info_text);
        draw_list->PopClipRect();
    }

    void BrushTool::onEnabledChanged(bool enabled) {
        if (tool_context_) {
            auto* const rm = tool_context_->getRenderingManager();
            if (rm) {
                rm->setOutputScreenPositions(enabled);
                if (enabled)
                    rm->markDirty();
            }
        }
    }

} // namespace lfs::vis::tools
