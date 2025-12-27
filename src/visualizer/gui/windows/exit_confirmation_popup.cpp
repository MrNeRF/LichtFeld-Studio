/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "exit_confirmation_popup.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    namespace {
        // Base dimensions (scaled by DPI factor at runtime)
        constexpr float BASE_POPUP_WIDTH = 340.0f;
        constexpr float BASE_POPUP_HEIGHT = 150.0f;

        constexpr float BASE_BUTTON_WIDTH = 100.0f;
        constexpr float BASE_BUTTON_SPACING = 12.0f;
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 BASE_WINDOW_PADDING{24.0f, 20.0f};

        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoMove |
                                                 ImGuiWindowFlags_NoDocking |
                                                 ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoScrollbar;
    } // namespace

    void ExitConfirmationPopup::show(Callback on_confirm, Callback on_cancel) {
        on_confirm_ = std::move(on_confirm);
        on_cancel_ = std::move(on_cancel);
        pending_open_ = true;
    }

    void ExitConfirmationPopup::render() {
        const char* popup_title = LOC(ExitPopup::TITLE);

        if (pending_open_) {
            ImGui::OpenPopup(popup_title);
            open_ = true;
            pending_open_ = false;
        }

        if (!open_) {
            return;
        }

        const auto& t = theme();
        const float scale = getDpiScale();
        const ImVec4 popup_bg{t.palette.surface.x, t.palette.surface.y,
                              t.palette.surface.z, POPUP_ALPHA};
        const ImVec4 title_bg = darken(t.palette.surface, 0.1f);
        const ImVec4 title_bg_active = darken(t.palette.surface, 0.05f);

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        ImGui::SetNextWindowSize({BASE_POPUP_WIDTH * scale, BASE_POPUP_HEIGHT * scale}, ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.primary);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(BASE_WINDOW_PADDING.x * scale, BASE_WINDOW_PADDING.y * scale));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal(popup_title, nullptr, POPUP_FLAGS)) {
            ImGui::TextUnformatted(LOC(ExitPopup::MESSAGE));
            ImGui::Spacing();
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(ExitPopup::UNSAVED_WARNING));
            ImGui::Dummy({0.0f, 8.0f * scale});

            // Center buttons
            const float button_width = BASE_BUTTON_WIDTH * scale;
            const float button_spacing = BASE_BUTTON_SPACING * scale;
            const float total_width = button_width * 2.0f + button_spacing;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x - total_width) * 0.5f);

            if (widgets::ColoredButton(LOC(Common::CANCEL), widgets::ButtonStyle::Secondary, {button_width, 0}) ||
                ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                open_ = false;
                if (on_cancel_)
                    on_cancel_();
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine(0.0f, button_spacing);

            if (widgets::ColoredButton(LOC(ExitPopup::EXIT), widgets::ButtonStyle::Error, {button_width, 0}) ||
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
