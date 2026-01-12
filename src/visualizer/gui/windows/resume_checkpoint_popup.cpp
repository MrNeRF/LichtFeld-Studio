/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "resume_checkpoint_popup.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/windows_utils.hpp"
#include "io/loader.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    namespace {
        constexpr float BASE_POPUP_WIDTH = 580.0f;
        constexpr float BASE_POPUP_HEIGHT = 340.0f;
        constexpr float BASE_INPUT_WIDTH = 400.0f;
        constexpr float BASE_MAX_PATH_WIDTH = 440.0f;
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

    void ResumeCheckpointPopup::show(const std::filesystem::path& checkpoint_path) {
        checkpoint_path_ = checkpoint_path;

        // Load checkpoint header and params
        const auto header_result = lfs::training::load_checkpoint_header(checkpoint_path);
        if (!header_result) {
            LOG_ERROR("Failed to read checkpoint header: {}", header_result.error());
            return;
        }
        header_ = *header_result;

        const auto params_result = lfs::training::load_checkpoint_params(checkpoint_path);
        if (!params_result) {
            LOG_ERROR("Failed to read checkpoint params: {}", params_result.error());
            return;
        }

        stored_dataset_path_ = lfs::core::path_to_utf8(params_result->dataset.data_path);
        dataset_path_buffer_ = stored_dataset_path_;
        dataset_path_buffer_.resize(PATH_BUFFER_SIZE);

        output_path_buffer_ = lfs::core::path_to_utf8(params_result->dataset.output_path);
        output_path_buffer_.resize(PATH_BUFFER_SIZE);

        // Check if stored dataset path is valid
        const auto stored_path = lfs::core::utf8_to_path(stored_dataset_path_);
        dataset_valid_ = lfs::io::Loader::isDatasetPath(stored_path);

        should_open_ = true;
    }

    void ResumeCheckpointPopup::renderPathRow(const char* label, const std::string& path, const float max_width) {
        const auto& t = theme();
        ImGui::TextColored(t.palette.text_dim, "%s", label);
        ImGui::SameLine();

        const bool is_clipped = ImGui::CalcTextSize(path.c_str()).x > max_width;
        ImGui::TextUnformatted(path.c_str());
        if (is_clipped && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", path.c_str());
        }
    }

    void ResumeCheckpointPopup::render(const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        const char* popup_title = LOC(ResumeCheckpointPopup_::TITLE);

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
            ImGui::TextColored(t.palette.info, "%s", LOC(ResumeCheckpointPopup_::CHECKPOINT));
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "|");
            ImGui::SameLine();
            ImGui::TextUnformatted(LOC(ResumeCheckpointPopup_::CONFIGURE_PATHS));

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Checkpoint info
            const std::string checkpoint_str = lfs::core::path_to_utf8(checkpoint_path_.filename());
            renderPathRow(LOC(ResumeCheckpointPopup_::FILE), checkpoint_str, max_path_width);
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "(iter %d, %u gaussians)", header_.iteration, header_.num_gaussians);

            ImGui::Spacing();

            // Stored path (read-only, for reference)
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(ResumeCheckpointPopup_::STORED_PATH));
            ImGui::SameLine();
            if (!dataset_valid_) {
                ImGui::TextColored(t.palette.error, "%s", stored_dataset_path_.c_str());
                ImGui::SameLine();
                ImGui::TextColored(t.palette.error, "(%s)", LOC(ResumeCheckpointPopup_::NOT_FOUND));
            } else {
                ImGui::TextUnformatted(stored_dataset_path_.c_str());
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Dataset path (editable)
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(ResumeCheckpointPopup_::DATASET_PATH));
            ImGui::SetNextItemWidth(BASE_INPUT_WIDTH * scale);
            if (ImGui::InputText("##dataset_path", dataset_path_buffer_.data(), PATH_BUFFER_SIZE)) {
                const auto new_path = lfs::core::utf8_to_path(dataset_path_buffer_);
                dataset_valid_ = lfs::io::Loader::isDatasetPath(new_path);
            }

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());
            if (ImGui::Button((std::string(LOC(Common::BROWSE)) + "##dataset").c_str())) {
                std::filesystem::path start_dir = std::filesystem::absolute(lfs::core::utf8_to_path(dataset_path_buffer_));
                if (!std::filesystem::exists(start_dir)) {
                    start_dir = start_dir.parent_path();
                }
                if (const auto selected = SelectFolderDialog(LOC(ResumeCheckpointPopup_::DATASET_PATH), start_dir); !selected.empty()) {
                    dataset_path_buffer_ = lfs::core::path_to_utf8(selected);
                    dataset_path_buffer_.resize(PATH_BUFFER_SIZE);
                    dataset_valid_ = lfs::io::Loader::isDatasetPath(selected);
                }
            }
            ImGui::PopStyleColor(3);

            // Validation indicator
            ImGui::SameLine();
            if (dataset_valid_) {
                ImGui::TextColored(t.palette.success, "[OK]");
            } else {
                ImGui::TextColored(t.palette.error, "[%s]", LOC(ResumeCheckpointPopup_::INVALID));
            }

            ImGui::Spacing();

            // Output path
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(LoadDatasetPopup::OUTPUT_DIR));
            ImGui::SetNextItemWidth(BASE_INPUT_WIDTH * scale);
            ImGui::InputText("##output_path", output_path_buffer_.data(), PATH_BUFFER_SIZE);

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());
            if (ImGui::Button((std::string(LOC(Common::BROWSE)) + "##output").c_str())) {
                std::filesystem::path start_dir = std::filesystem::absolute(lfs::core::utf8_to_path(output_path_buffer_));
                if (!std::filesystem::exists(start_dir)) {
                    start_dir = std::filesystem::absolute(lfs::core::utf8_to_path(dataset_path_buffer_));
                }
                if (const auto selected = SelectFolderDialog(LOC(LoadDatasetPopup::OUTPUT_DIR), start_dir); !selected.empty()) {
                    output_path_buffer_ = lfs::core::path_to_utf8(selected);
                    output_path_buffer_.resize(PATH_BUFFER_SIZE);
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::Spacing();

            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(ResumeCheckpointPopup_::HELP_TEXT));
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

            ImGui::BeginDisabled(!dataset_valid_);
            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.success, DARKEN_SUCCESS_BUTTON));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.success, DARKEN_SUCCESS_HOVER));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.success, DARKEN_SUCCESS_ACTIVE));
            if (ImGui::Button(LOC(Common::LOAD), button_size) || (dataset_valid_ && ImGui::IsKeyPressed(ImGuiKey_Enter))) {
                popup_open_ = false;
                if (on_confirm_) {
                    CheckpointLoadParams params;
                    params.checkpoint_path = checkpoint_path_;
                    params.dataset_path = lfs::core::utf8_to_path(dataset_path_buffer_);
                    params.output_path = lfs::core::utf8_to_path(output_path_buffer_);
                    on_confirm_(params);
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor(3);
            ImGui::EndDisabled();

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(8);
    }

} // namespace lfs::vis::gui
