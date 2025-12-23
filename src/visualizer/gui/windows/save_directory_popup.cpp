/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "save_directory_popup.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        // Fixed dimensions to prevent DPI-related resize feedback loop
        constexpr float POPUP_WIDTH = 500.0f;
        constexpr float POPUP_HEIGHT = 260.0f;
        constexpr float INPUT_WIDTH = 340.0f;
        constexpr float MAX_PATH_WIDTH = 380.0f;

        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 WINDOW_PADDING = {20.0f, 16.0f};
        constexpr ImVec2 BUTTON_SIZE = {100.0f, 0.0f};

        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoDocking |
                                                 ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoScrollbar |
                                                 ImGuiWindowFlags_NoScrollWithMouse;
        constexpr size_t PATH_BUFFER_SIZE = 1024;
        constexpr float DARKEN_TITLE = 0.1f;
        constexpr float DARKEN_TITLE_ACTIVE = 0.05f;
        constexpr float DARKEN_SUCCESS_BUTTON = 0.3f;
        constexpr float DARKEN_SUCCESS_HOVER = 0.15f;
        constexpr float DARKEN_SUCCESS_ACTIVE = 0.2f;
    } // namespace

    void SaveDirectoryPopup::show(const std::filesystem::path& dataset_path) {
        dataset_path_ = dataset_path;
        output_path_buffer_ = deriveDefaultOutputPath(dataset_path).string();
        output_path_buffer_.resize(PATH_BUFFER_SIZE);
        should_open_ = true;
    }

    std::filesystem::path SaveDirectoryPopup::deriveDefaultOutputPath(const std::filesystem::path& dataset_path) {
        return dataset_path / "output";
    }

    void SaveDirectoryPopup::render(const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        if (should_open_) {
            ImGui::OpenPopup("Select Output Directory");
            popup_open_ = true;
            should_open_ = false;
        }

        if (!popup_open_)
            return;

        const auto& t = theme();
        const ImVec4 popup_bg = {t.palette.surface.x, t.palette.surface.y, t.palette.surface.z, POPUP_ALPHA};
        const ImVec4 title_bg = darken(t.palette.surface, DARKEN_TITLE);
        const ImVec4 title_bg_active = darken(t.palette.surface, DARKEN_TITLE_ACTIVE);
        const ImVec4 frame_bg = darken(t.palette.surface, DARKEN_TITLE);

        // Center in viewport if provided
        const ImVec2 center = (viewport_size.x > 0 && viewport_size.y > 0)
                                  ? ImVec2{viewport_pos.x + viewport_size.x * 0.5f, viewport_pos.y + viewport_size.y * 0.5f}
                                  : ImGui::GetMainViewport()->GetCenter();

        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});
        ImGui::SetNextWindowSize({POPUP_WIDTH, POPUP_HEIGHT}, ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.info);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, frame_bg);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, t.palette.primary_dim);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, WINDOW_PADDING);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal("Select Output Directory", nullptr, POPUP_FLAGS)) {
            ImGui::TextColored(t.palette.info, "Dataset");
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "|");
            ImGui::SameLine();
            ImGui::TextUnformatted("Configure output location");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextColored(t.palette.text_dim, "Dataset:");
            ImGui::SameLine();
            const std::string dataset_str = dataset_path_.string();
            const bool is_clipped = ImGui::CalcTextSize(dataset_str.c_str()).x > MAX_PATH_WIDTH;
            ImGui::TextUnformatted(dataset_str.c_str());
            if (is_clipped && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", dataset_str.c_str());
            }

            ImGui::Spacing();

            ImGui::TextColored(t.palette.text_dim, "Output Directory:");
            ImGui::SetNextItemWidth(INPUT_WIDTH);
            ImGui::InputText("##output_path", output_path_buffer_.data(), PATH_BUFFER_SIZE);

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());
            if (ImGui::Button("Browse...")) {
                std::filesystem::path start_dir(output_path_buffer_.c_str());
                if (!std::filesystem::exists(start_dir)) {
                    start_dir = dataset_path_;
                }
                if (const auto selected = SelectFolderDialog("Select Output Directory", start_dir); !selected.empty()) {
                    output_path_buffer_ = selected.string();
                    output_path_buffer_.resize(PATH_BUFFER_SIZE);
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::Spacing();
            ImGui::Spacing();

            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 25.0f);
            ImGui::TextColored(t.palette.text_dim,
                               "Training checkpoints, logs, and exported models will be saved to this directory.");
            ImGui::PopTextWrapPos();

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            const float avail = ImGui::GetContentRegionAvail().x;
            const float total_width = BUTTON_SIZE.x * 2 + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - total_width);

            if (ImGui::Button("Cancel", BUTTON_SIZE) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                popup_open_ = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.success, DARKEN_SUCCESS_BUTTON));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.success, DARKEN_SUCCESS_HOVER));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.success, DARKEN_SUCCESS_ACTIVE));
            if (ImGui::Button("Load", BUTTON_SIZE) || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                popup_open_ = false;
                if (on_confirm_) {
                    on_confirm_(dataset_path_, std::filesystem::path(output_path_buffer_.c_str()));
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor(3);

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(8);
    }

} // namespace lfs::vis::gui
