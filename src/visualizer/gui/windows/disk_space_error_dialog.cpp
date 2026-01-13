/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "disk_space_error_dialog.hpp"
#include "core/path_utils.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    namespace {
        // Base dimensions (scaled by DPI factor at runtime)
        constexpr float BASE_POPUP_WIDTH = 480.0f;
        constexpr float BASE_POPUP_HEIGHT = 280.0f;
        constexpr float BASE_BUTTON_WIDTH = 120.0f;
        constexpr float BASE_BUTTON_SPACING = 12.0f;
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 BASE_WINDOW_PADDING{24.0f, 20.0f};

        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoMove |
                                                 ImGuiWindowFlags_NoDocking |
                                                 ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoScrollbar |
                                                 ImGuiWindowFlags_NoSavedSettings;

        [[nodiscard]] std::string formatBytes(const size_t bytes) {
            constexpr double KB = 1024.0;
            constexpr double MB = KB * 1024.0;
            constexpr double GB = MB * 1024.0;

            if (bytes >= GB) {
                return std::format("{:.2f} GB", static_cast<double>(bytes) / GB);
            }
            if (bytes >= MB) {
                return std::format("{:.2f} MB", static_cast<double>(bytes) / MB);
            }
            if (bytes >= KB) {
                return std::format("{:.2f} KB", static_cast<double>(bytes) / KB);
            }
            return std::format("{} bytes", bytes);
        }
    } // namespace

    void DiskSpaceErrorDialog::show(const ErrorInfo& info,
                                    RetryCallback on_retry,
                                    ChangeLocationCallback on_change_location,
                                    CancelCallback on_cancel) {
        info_ = info;
        on_retry_ = std::move(on_retry);
        on_change_location_ = std::move(on_change_location);
        on_cancel_ = std::move(on_cancel);
        pending_open_ = true;
    }

    void DiskSpaceErrorDialog::render() {
        const char* popup_title = LOC(DiskSpaceDialog::TITLE);

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
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.error);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(BASE_WINDOW_PADDING.x * scale, BASE_WINDOW_PADDING.y * scale));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal(popup_title, nullptr, POPUP_FLAGS)) {
            // Error icon and title
            ImGui::TextColored(t.palette.error, LOC(DiskSpaceDialog::ERROR_LABEL));
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "|");
            ImGui::SameLine();
            if (info_.is_checkpoint) {
                ImGui::Text("%s %d)", LOC(DiskSpaceDialog::CHECKPOINT_SAVE_FAILED), info_.iteration);
            } else {
                ImGui::TextUnformatted(LOC(DiskSpaceDialog::EXPORT_FAILED));
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Error message
            ImGui::TextWrapped(LOC(DiskSpaceDialog::INSUFFICIENT_SPACE_PREFIX));
            ImGui::Spacing();

            // Location
            ImGui::TextColored(t.palette.text_dim, LOC(DiskSpaceDialog::LOCATION_LABEL));
            ImGui::SameLine();
            ImGui::TextWrapped("%s", lfs::core::path_to_utf8(info_.path.parent_path()).c_str());

            // Space required vs available
            ImGui::Spacing();
            ImGui::TextColored(t.palette.text_dim, LOC(DiskSpaceDialog::REQUIRED_LABEL));
            ImGui::SameLine();
            ImGui::Text("%s", formatBytes(info_.required_bytes).c_str());

            if (info_.available_bytes > 0) {
                ImGui::TextColored(t.palette.text_dim, LOC(DiskSpaceDialog::AVAILABLE_LABEL));
                ImGui::SameLine();
                ImGui::TextColored(t.palette.error, "%s", formatBytes(info_.available_bytes).c_str());
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Instructions
            ImGui::TextColored(t.palette.warning, LOC(DiskSpaceDialog::INSTRUCTION));

            ImGui::Dummy({0.0f, 8.0f * scale});

            // Buttons
            const float button_width = BASE_BUTTON_WIDTH * scale;
            const float button_spacing = BASE_BUTTON_SPACING * scale;
            const float total_width = button_width * 3.0f + button_spacing * 2.0f;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x - total_width) * 0.5f);

            // Cancel button
            if (widgets::ColoredButton(LOC(DiskSpaceDialog::CANCEL), widgets::ButtonStyle::Secondary, {button_width, 0}) ||
                ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                open_ = false;
                if (on_cancel_)
                    on_cancel_();
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine(0.0f, button_spacing);

            // Change Location button
            if (widgets::ColoredButton(LOC(DiskSpaceDialog::CHANGE_LOCATION), widgets::ButtonStyle::Warning, {button_width, 0})) {
                // Open folder selection dialog
                std::filesystem::path new_location = SelectFolderDialog(LOC(DiskSpaceDialog::SELECT_OUTPUT_LOCATION), info_.path.parent_path());
                if (!new_location.empty()) {
                    open_ = false;
                    if (on_change_location_)
                        on_change_location_(new_location);
                    ImGui::CloseCurrentPopup();
                }
            }

            ImGui::SameLine(0.0f, button_spacing);

            // Retry button
            if (widgets::ColoredButton(LOC(DiskSpaceDialog::RETRY), widgets::ButtonStyle::Primary, {button_width, 0}) ||
                ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                open_ = false;
                if (on_retry_)
                    on_retry_();
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(5);
    }

} // namespace lfs::vis::gui
