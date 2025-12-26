/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "exit_confirmation_popup.hpp"
#include "gui/ui_widgets.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        // Fixed dimensions to prevent DPI-related resize feedback loop
        constexpr float POPUP_WIDTH = 340.0f;
        constexpr float POPUP_HEIGHT = 150.0f;

        constexpr float BUTTON_WIDTH = 100.0f;
        constexpr float BUTTON_SPACING = 12.0f;
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 WINDOW_PADDING{24.0f, 20.0f};

        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoMove |
                                                 ImGuiWindowFlags_NoDocking |
                                                 ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoScrollbar;
        constexpr const char* POPUP_TITLE = "Exit Application?";
    } // namespace

    void ExitConfirmationPopup::show(Callback on_confirm, Callback on_cancel) {
        on_confirm_ = std::move(on_confirm);
        on_cancel_ = std::move(on_cancel);
        pending_open_ = true;
    }

    void ExitConfirmationPopup::render() {
        if (pending_open_) {
            ImGui::OpenPopup(POPUP_TITLE);
            open_ = true;
            pending_open_ = false;
        }

        if (!open_) {
            return;
        }

        const auto& t = theme();
        const ImVec4 popup_bg{t.palette.surface.x, t.palette.surface.y,
                              t.palette.surface.z, POPUP_ALPHA};
        const ImVec4 title_bg = darken(t.palette.surface, 0.1f);
        const ImVec4 title_bg_active = darken(t.palette.surface, 0.05f);

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        ImGui::SetNextWindowSize({POPUP_WIDTH, POPUP_HEIGHT}, ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.primary);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, WINDOW_PADDING);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal(POPUP_TITLE, nullptr, POPUP_FLAGS)) {
            ImGui::TextUnformatted("Are you sure you want to exit?");
            ImGui::Spacing();
            ImGui::TextColored(t.palette.text_dim, "Any unsaved changes will be lost.");
            ImGui::Dummy({0.0f, 8.0f});

            // Center buttons
            const float total_width = BUTTON_WIDTH * 2.0f + BUTTON_SPACING;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x - total_width) * 0.5f);

            if (widgets::ColoredButton("Cancel", widgets::ButtonStyle::Secondary, {BUTTON_WIDTH, 0}) ||
                ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                open_ = false;
                if (on_cancel_)
                    on_cancel_();
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine(0.0f, BUTTON_SPACING);

            if (widgets::ColoredButton("Exit", widgets::ButtonStyle::Error, {BUTTON_WIDTH, 0}) ||
                ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                open_ = false;
                if (on_confirm_)
                    on_confirm_();
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(5);
    }

} // namespace lfs::vis::gui
