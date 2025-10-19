/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/windows/save_project_browser.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/utils/windows_utils.hpp"
#include <algorithm>
#include <imgui.h>
#include <print>

namespace gs::gui {

#ifdef WIN32
    bool SaveProjectBrowser::SaveProjectFileDialog(bool* p_open) {
        // show native windows file dialog for project directory selection
        PWSTR filePath = nullptr;
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path project_path(filePath);
            events::cmd::SaveProject{project_path}.emit();
            LOG_INFO("Saving project file into : {}", std::filesystem::path(project_path).string());
            *p_open = false;
            return true;
        }
        *p_open = false;
        return false;
    }
#endif

    SaveProjectBrowser::SaveProjectBrowser() {
        current_path_ = std::filesystem::current_path().string();
        project_dir_name_ = "";
    }

    // returns true if project was saved
    bool SaveProjectBrowser::render(bool* p_open) {
        ImGui::SetNextWindowSize(ImVec2(650, 400), ImGuiCond_FirstUseEver);

        // Add NoDocking flag and make it modal
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.17f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));

        bool was_project_saved = false;

        if (!ImGui::Begin("Save Project", p_open, window_flags)) {
            ImGui::End();
            ImGui::PopStyleColor(2);
            return was_project_saved;
        }

        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Quick Access")) {
                if (ImGui::MenuItem("Current Directory")) {
                    current_path_ = std::filesystem::current_path().string();
                    selected_directory_.clear();
                }
                if (ImGui::MenuItem("Home")) {
                    current_path_ = std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : "/").string();
                    selected_directory_.clear();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        ImGui::Text("Select directory to save project:");
        ImGui::Text("Current Path: %s", current_path_.c_str());
        ImGui::Separator();

        // Directory listing
        if (ImGui::BeginChild("DirectoryList", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 4), true)) {
            std::filesystem::path current_path(current_path_);

            // Parent directory option
            if (current_path.has_parent_path()) {
                if (ImGui::Selectable("../", false, ImGuiSelectableFlags_DontClosePopups)) {
                    current_path_ = current_path.parent_path().string();
                    selected_directory_.clear();
                }
            }

            // Current directory option
            bool current_selected = (selected_directory_ == current_path_);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 0.4f, 1.0f));
            if (ImGui::Selectable("./", current_selected, ImGuiSelectableFlags_DontClosePopups)) {
                selected_directory_ = current_path_;
            }
            ImGui::PopStyleColor();

            if (current_selected) {
                ImGui::SameLine();
                ImGui::TextDisabled("[Selected]");
            }

            std::vector<std::filesystem::directory_entry> dirs;

            try {
                for (const auto& entry : std::filesystem::directory_iterator(current_path)) {
                    if (entry.is_directory()) {
                        dirs.push_back(entry);
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", e.what());
            }

            std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
                return a.path().filename() < b.path().filename();
            });

            for (const auto& dir : dirs) {
                std::string dirname = "[DIR] " + dir.path().filename().string();
                bool is_selected = (selected_directory_ == dir.path().string());

                if (ImGui::Selectable(dirname.c_str(), is_selected,
                                      ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_DontClosePopups)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        // Navigate into directory
                        current_path_ = dir.path().string();
                        selected_directory_ = current_path_;
                    } else {
                        // Select directory
                        selected_directory_ = dir.path().string();
                    }
                }

                if (is_selected) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("[Selected]");
                }
            }
        }
        ImGui::EndChild();

        // Project name input
        ImGui::Separator();

        // Make the label white (or light gray)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
        ImGui::Text("Project dir name:");
        ImGui::PopStyleColor();

        ImGui::SetNextItemWidth(-1);
        char name_buffer[256];

        std::string temp = project_dir_name_.substr(0, sizeof(name_buffer) - 1);
        std::copy(temp.begin(), temp.end(), name_buffer);
        name_buffer[temp.size()] = '\0';

        // Keep input text black (on white background)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0, 0, 0, 1));
        if (ImGui::InputText("##project_dir_name", name_buffer, sizeof(name_buffer))) {
            project_dir_name_ = name_buffer;
        }
        ImGui::PopStyleColor();

        // Selected path display
        if (!selected_directory_.empty()) {
            std::filesystem::path final_path = std::filesystem::path(selected_directory_) / project_dir_name_;
            ImGui::Text("Will save to: %s", final_path.string().c_str());
        } else {
            ImGui::TextDisabled("No directory selected");
        }

        ImGui::Separator();

        // Action buttons
        bool can_save = !selected_directory_.empty();

        if (!can_save) {
            ImGui::BeginDisabled();
        }

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.3f, 0.9f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.4f, 1.0f, 1.0f));
        if (ImGui::Button("Save Project", ImVec2(120, 0))) {
            std::filesystem::path project_dir = std::filesystem::path(selected_directory_) / project_dir_name_;
            // Emit the SaveProject event
            events::cmd::SaveProject{project_dir}.emit();
            // Call the callback if set
            *p_open = false;
            was_project_saved = true;
        }
        ImGui::PopStyleColor(2);

        if (!can_save) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            *p_open = false;
            selected_directory_.clear();
        }

        ImGui::End();
        ImGui::PopStyleColor(2);

        return was_project_saved;
    }

    void SaveProjectBrowser::setCurrentPath(const std::filesystem::path& path) {
        current_path_ = path.string();
        selected_directory_.clear();
    }

} // namespace gs::gui
