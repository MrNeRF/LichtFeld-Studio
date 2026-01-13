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
        constexpr float POPUP_WIDTH = 340.0f;
        constexpr float POPUP_HEIGHT = 150.0f;
        constexpr float BUTTON_WIDTH = 100.0f;
        constexpr float BUTTON_SPACING = 12.0f;
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr float TITLE_DARKEN = 0.1f;
        constexpr float TITLE_ACTIVE_DARKEN = 0.05f;
        constexpr float DIM_ALPHA = 0.35f;
        constexpr ImVec2 WINDOW_PADDING{24.0f, 20.0f};

        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                                 ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;
    } // namespace

    void ExitConfirmationPopup::show(Callback on_confirm, Callback on_cancel) {
        on_confirm_ = std::move(on_confirm);
        on_cancel_ = std::move(on_cancel);
        pending_open_ = true;
    }

    void ExitConfirmationPopup::render() {
        const char* const popup_title = LOC(ExitPopup::TITLE);
        const auto& t = theme();
        const float scale = getDpiScale();

        const ImVec4 popup_bg = withAlpha(t.palette.surface, POPUP_ALPHA);
        const ImVec4 title_bg = darken(t.palette.surface, TITLE_DARKEN);
        const ImVec4 title_bg_active = darken(t.palette.surface, TITLE_ACTIVE_DARKEN);

        // Set global style before OpenPopup - ImGui caches colors at popup creation
        auto& colors = ImGui::GetStyle().Colors;
        colors[ImGuiCol_ModalWindowDimBg] = withAlpha(t.palette.background, DIM_ALPHA);
        colors[ImGuiCol_WindowBg] = popup_bg;
        colors[ImGuiCol_TitleBg] = title_bg;
        colors[ImGuiCol_TitleBgActive] = title_bg_active;
        colors[ImGuiCol_PopupBg] = popup_bg;

        if (pending_open_) {
            ImGui::OpenPopup(popup_title);
            open_ = true;
            pending_open_ = false;
        }

        if (!open_)
            return;

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        ImGui::SetNextWindowSize({POPUP_WIDTH * scale, POPUP_HEIGHT * scale}, ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.primary);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {WINDOW_PADDING.x * scale, WINDOW_PADDING.y * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal(popup_title, nullptr, POPUP_FLAGS)) {
            ImGui::TextUnformatted(LOC(ExitPopup::MESSAGE));
            ImGui::Spacing();
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(ExitPopup::UNSAVED_WARNING));
            ImGui::Dummy({0.0f, 8.0f * scale});

            const float scaled_btn_width = BUTTON_WIDTH * scale;
            const float scaled_btn_spacing = BUTTON_SPACING * scale;
            const float total_width = scaled_btn_width * 2.0f + scaled_btn_spacing;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x - total_width) * 0.5f);

            if (widgets::ColoredButton(LOC(Common::CANCEL), widgets::ButtonStyle::Secondary, {scaled_btn_width, 0}) ||
                ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                open_ = false;
                if (on_cancel_)
                    on_cancel_();
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine(0.0f, scaled_btn_spacing);

            if (widgets::ColoredButton(LOC(ExitPopup::EXIT), widgets::ButtonStyle::Error, {scaled_btn_width, 0}) ||
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
