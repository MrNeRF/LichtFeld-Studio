/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "save_directory_popup.hpp"
#include "core/path_utils.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    namespace {
        constexpr float BASE_POPUP_WIDTH = 560.0f;
        constexpr float BASE_POPUP_HEIGHT = 360.0f;
        constexpr float BASE_INPUT_WIDTH = 380.0f;
        constexpr float BASE_MAX_PATH_WIDTH = 420.0f;
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 BASE_WINDOW_PADDING = {20.0f, 16.0f};
        constexpr ImVec2 BASE_BUTTON_SIZE = {100.0f, 0.0f};
        constexpr size_t PATH_BUFFER_SIZE = 1024;
        constexpr float DARKEN_TITLE = 0.1f;
        constexpr float DARKEN_TITLE_ACTIVE = 0.05f;
        constexpr float DARKEN_SUCCESS_BUTTON = 0.3f;
        constexpr float DARKEN_SUCCESS_HOVER = 0.15f;
        constexpr float DARKEN_SUCCESS_ACTIVE = 0.2f;

        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoDocking |
                                                 ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoScrollbar |
                                                 ImGuiWindowFlags_NoScrollWithMouse |
                                                 ImGuiWindowFlags_NoSavedSettings;
    } // namespace

    void SaveDirectoryPopup::show(const std::filesystem::path& dataset_path) {
        dataset_info_ = lfs::io::detect_dataset_info(dataset_path);
        output_path_buffer_ = lfs::core::path_to_utf8(deriveDefaultOutputPath(dataset_path));
        output_path_buffer_.resize(PATH_BUFFER_SIZE);
        init_path_buffer_.clear();
        init_path_buffer_.resize(PATH_BUFFER_SIZE);
        use_custom_init_ = false;
        should_open_ = true;
    }

    std::filesystem::path SaveDirectoryPopup::deriveDefaultOutputPath(const std::filesystem::path& dataset_path) {
        return dataset_path / "output";
    }

    void SaveDirectoryPopup::renderPathRow(const char* label, const std::string& path, const float max_width) {
        const auto& t = theme();
        ImGui::TextColored(t.palette.text_dim, "%s", label);
        ImGui::SameLine();

        const bool is_clipped = ImGui::CalcTextSize(path.c_str()).x > max_width;
        ImGui::TextUnformatted(path.c_str());
        if (is_clipped && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", path.c_str());
        }
    }

    void SaveDirectoryPopup::render(const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        const char* popup_title = LOC(LoadDatasetPopup::TITLE);

        if (should_open_) {
            ImGui::OpenPopup(popup_title);
            popup_open_ = true;
            should_open_ = false;
        }

        if (!popup_open_)
            return;

        const auto& t = theme();
        const float scale = getDpiScale();
        const ImVec4 popup_bg = {t.palette.surface.x, t.palette.surface.y, t.palette.surface.z, POPUP_ALPHA};
        const ImVec4 title_bg = darken(t.palette.surface, DARKEN_TITLE);
        const ImVec4 title_bg_active = darken(t.palette.surface, DARKEN_TITLE_ACTIVE);
        const ImVec4 frame_bg = darken(t.palette.surface, DARKEN_TITLE);
        const float max_path_width = BASE_MAX_PATH_WIDTH * scale;

        const ImVec2 center = (viewport_size.x > 0 && viewport_size.y > 0)
                                  ? ImVec2{viewport_pos.x + viewport_size.x * 0.5f, viewport_pos.y + viewport_size.y * 0.5f}
                                  : ImGui::GetMainViewport()->GetCenter();

        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});
        ImGui::SetNextWindowSize({BASE_POPUP_WIDTH * scale, BASE_POPUP_HEIGHT * scale}, ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.info);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, frame_bg);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, t.palette.primary_dim);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(BASE_WINDOW_PADDING.x * scale, BASE_WINDOW_PADDING.y * scale));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal(popup_title, nullptr, POPUP_FLAGS)) {
            ImGui::TextColored(t.palette.info, "%s", LOC(Training::Section::DATASET));
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "|");
            ImGui::SameLine();
            ImGui::TextUnformatted(LOC(LoadDatasetPopup::CONFIGURE_PATHS));

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            const std::string images_str = lfs::core::path_to_utf8(dataset_info_.images_path);
            renderPathRow(LOC(LoadDatasetPopup::IMAGES_DIR), images_str, max_path_width);
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "(%d images)", dataset_info_.image_count);

            const std::string sparse_str = lfs::core::path_to_utf8(dataset_info_.sparse_path);
            renderPathRow(LOC(LoadDatasetPopup::SPARSE_DIR), sparse_str, max_path_width);

            if (dataset_info_.has_masks) {
                const std::string masks_str = lfs::core::path_to_utf8(dataset_info_.masks_path);
                renderPathRow(LOC(LoadDatasetPopup::MASKS_DIR), masks_str, max_path_width);
                ImGui::SameLine();
                ImGui::TextColored(t.palette.text_dim, "(%d masks)", dataset_info_.mask_count);
            }

            ImGui::Spacing();

            ImGui::Checkbox("Custom init file (replaces points3D)", &use_custom_init_);
            if (use_custom_init_) {
                ImGui::SetNextItemWidth(BASE_INPUT_WIDTH * scale);
                ImGui::InputText("##init_path", init_path_buffer_.data(), PATH_BUFFER_SIZE);
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());
                if (ImGui::Button((std::string(LOC(Common::BROWSE)) + "##init").c_str())) {
                    if (const auto selected = OpenPlyFileDialogNative(dataset_info_.base_path); !selected.empty()) {
                        init_path_buffer_ = lfs::core::path_to_utf8(selected);
                        init_path_buffer_.resize(PATH_BUFFER_SIZE);
                    }
                }
                ImGui::PopStyleColor(3);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextColored(t.palette.text_dim, "%s", LOC(LoadDatasetPopup::OUTPUT_DIR));
            ImGui::SetNextItemWidth(BASE_INPUT_WIDTH * scale);
            ImGui::InputText("##output_path", output_path_buffer_.data(), PATH_BUFFER_SIZE);

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());
            if (ImGui::Button((std::string(LOC(Common::BROWSE)) + "##output").c_str())) {
                std::filesystem::path start_dir = lfs::core::utf8_to_path(output_path_buffer_);
                if (!std::filesystem::exists(start_dir)) {
                    start_dir = dataset_info_.base_path;
                }
                if (const auto selected = SelectFolderDialog(LOC(LoadDatasetPopup::TITLE), start_dir); !selected.empty()) {
                    output_path_buffer_ = lfs::core::path_to_utf8(selected);
                    output_path_buffer_.resize(PATH_BUFFER_SIZE);
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::Spacing();

            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(LoadDatasetPopup::HELP_TEXT));
            ImGui::PopTextWrapPos();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            const float avail = ImGui::GetContentRegionAvail().x;
            const ImVec2 button_size = {BASE_BUTTON_SIZE.x * scale, BASE_BUTTON_SIZE.y};
            const float total_width = button_size.x * 2 + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - total_width);

            if (ImGui::Button(LOC(Common::CANCEL), button_size) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                popup_open_ = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.success, DARKEN_SUCCESS_BUTTON));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.success, DARKEN_SUCCESS_HOVER));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.success, DARKEN_SUCCESS_ACTIVE));
            if (ImGui::Button(LOC(Common::LOAD), button_size) || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                popup_open_ = false;
                if (on_confirm_) {
                    DatasetLoadParams params;
                    params.dataset_path = dataset_info_.base_path;
                    params.output_path = lfs::core::utf8_to_path(output_path_buffer_);
                    if (use_custom_init_ && !init_path_buffer_.empty() && init_path_buffer_[0] != '\0') {
                        params.init_path = lfs::core::utf8_to_path(init_path_buffer_);
                    }
                    on_confirm_(params);
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
