/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/windows/file_browser.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    FileBrowser::FileBrowser() {
        current_path_ = std::filesystem::current_path().string();
    }

    void FileBrowser::render(bool* p_open) {
        const float scale = getDpiScale();
        ImGui::SetNextWindowSize(ImVec2(700 * scale, 450 * scale), ImGuiCond_FirstUseEver);

        // Add NoDocking flag
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.15f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));

        if (!ImGui::Begin(LOC(lichtfeld::Strings::FileBrowser::TITLE), p_open, window_flags)) {
            ImGui::End();
            ImGui::PopStyleColor(2);
            return;
        }

        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu(LOC(lichtfeld::Strings::FileBrowser::QUICK_ACCESS))) {
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::FileBrowser::CURRENT_DIR))) {
                    current_path_ = std::filesystem::current_path().string();
                }
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::FileBrowser::HOME))) {
                    current_path_ = std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : "/").string();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        ImGui::Text(LOC(lichtfeld::Strings::FileBrowser::CURRENT_PATH), current_path_.c_str());
        ImGui::Separator();

        if (ImGui::BeginChild("FileList", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 2), true)) {
            std::filesystem::path current_path(current_path_);

            if (current_path.has_parent_path()) {
                if (ImGui::Selectable("../", false, ImGuiSelectableFlags_DontClosePopups)) {
                    current_path_ = current_path.parent_path().string();
                    selected_file_.clear();
                }
            }

            std::vector<std::filesystem::directory_entry> dirs;
            std::vector<std::filesystem::directory_entry> files;

            try {
                for (const auto& entry : std::filesystem::directory_iterator(current_path)) {
                    if (entry.is_directory()) {
                        dirs.push_back(entry);
                    } else if (entry.is_regular_file()) {
                        auto ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                        // Supported file extensions
                        if (ext == ".ply" || ext == ".sog" || ext == ".json" ||
                            entry.path().filename() == "cameras.bin" ||
                            entry.path().filename() == "cameras.txt" ||
                            entry.path().filename() == "images.bin" ||
                            entry.path().filename() == "images.txt" ||
                            entry.path().filename() == "points3D.bin" ||
                            entry.path().filename() == "points3D.txt" ||
                            entry.path().filename() == "meta.json" ||
                            entry.path().filename() == "transforms.json" ||
                            entry.path().filename() == "transforms_train.json") {
                            files.push_back(entry);
                        }
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", LOC(FileBrowserExt::ERROR_MSG), e.what());
            }

            std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
                return a.path().filename() < b.path().filename();
            });
            std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
                return a.path().filename() < b.path().filename();
            });

            // Create a Loader instance to check if paths can be loaded
            auto loader = lfs::io::Loader::create();

            for (const auto& dir : dirs) {
                std::string dirname = std::string(LOC(lichtfeld::Strings::FileBrowser::DIRECTORY)) + " " + dir.path().filename().string();
                bool is_selected = (selected_file_ == dir.path().string());

                bool is_dataset = lfs::io::Loader::isDatasetPath(dir.path());
                bool is_sog_dir = false;

                // Check if it's a SOG directory (has meta.json and WebP files)
                if (!is_dataset && std::filesystem::exists(dir.path() / "meta.json")) {
                    // Check for SOG-specific files
                    if (std::filesystem::exists(dir.path() / "means_l.webp") ||
                        std::filesystem::exists(dir.path() / "means_u.webp") ||
                        std::filesystem::exists(dir.path() / "quats.webp") ||
                        std::filesystem::exists(dir.path() / "scales.webp") ||
                        std::filesystem::exists(dir.path() / "sh0.webp")) {
                        is_sog_dir = true;
                    }
                }

                if (is_dataset) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.5f, 0.9f, 1.0f));
                    dirname += std::string(" ") + LOC(FileBrowserExt::DATASET);
                } else if (is_sog_dir) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.6f, 0.2f, 1.0f)); // Orange for SOG
                    dirname += std::string(" ") + LOC(FileBrowserExt::SOG);
                }

                if (ImGui::Selectable(dirname.c_str(), is_selected,
                                      ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_DontClosePopups)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        if (!is_sog_dir) {
                            current_path_ = dir.path().string();
                            selected_file_.clear();
                        } else {
                            // For SOG directories, select them instead of entering
                            selected_file_ = dir.path().string();
                        }
                    } else {
                        selected_file_ = dir.path().string();
                    }
                }

                if (is_dataset || is_sog_dir) {
                    ImGui::PopStyleColor();
                }
            }

            for (const auto& file : files) {
                std::string filename = file.path().filename().string();
                bool is_selected = (selected_file_ == file.path().string());

                ImVec4 color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                if (file.path().extension() == ".ply") {
                    color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f); // Green for PLY
                } else if (file.path().extension() == ".sog") {
                    color = ImVec4(0.9f, 0.6f, 0.2f, 1.0f); // Orange for SOG
                } else if (filename == "cameras.bin" || filename == "cameras.txt" ||
                           filename == "images.bin" || filename == "images.txt" ||
                           filename == "transforms.json" || filename == "transforms_train.json") {
                    color = ImVec4(0.3f, 0.5f, 0.9f, 1.0f); // Blue for dataset files
                } else if (filename == "meta.json") {
                    // Check if it's a SOG meta.json by looking for SOG files in the same directory
                    bool is_sog_meta = false;
                    auto parent = file.path().parent_path();
                    if (std::filesystem::exists(parent / "means_l.webp") ||
                        std::filesystem::exists(parent / "means_u.webp") ||
                        std::filesystem::exists(parent / "quats.webp") ||
                        std::filesystem::exists(parent / "scales.webp") ||
                        std::filesystem::exists(parent / "sh0.webp")) {
                        is_sog_meta = true;
                    }
                    color = is_sog_meta ? ImVec4(0.9f, 0.6f, 0.2f, 1.0f) : ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                }

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                if (ImGui::Selectable(filename.c_str(), is_selected, ImGuiSelectableFlags_DontClosePopups)) {
                    selected_file_ = file.path().string();
                }
                ImGui::PopStyleColor();
            }
        }
        ImGui::EndChild();

        if (!selected_file_.empty()) {
            ImGui::Text(LOC(lichtfeld::Strings::FileBrowser::SELECTED), std::filesystem::path(selected_file_).filename().string().c_str());
        } else {
            ImGui::TextDisabled("%s", LOC(FileBrowserExt::NO_FILE_SELECTED));
        }

        ImGui::Separator();

        bool can_load = !selected_file_.empty();

        if (!can_load) {
            ImGui::BeginDisabled();
        }

        if (can_load) {
            std::filesystem::path selected_path(selected_file_);

            if (std::filesystem::is_directory(selected_path)) {
                bool is_dataset = lfs::io::Loader::isDatasetPath(selected_path);

                // Check if it's a SOG directory
                bool is_sog_dir = false;
                if (std::filesystem::exists(selected_path / "meta.json")) {
                    if (std::filesystem::exists(selected_path / "means_l.webp") ||
                        std::filesystem::exists(selected_path / "means_u.webp") ||
                        std::filesystem::exists(selected_path / "quats.webp") ||
                        std::filesystem::exists(selected_path / "scales.webp") ||
                        std::filesystem::exists(selected_path / "sh0.webp")) {
                        is_sog_dir = true;
                    }
                }

                if (is_dataset) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
                    if (ImGui::Button(LOC(lichtfeld::Strings::FileBrowser::LOAD_DATASET), ImVec2(120 * scale, 0))) {
                        if (on_file_selected_) {
                            on_file_selected_(selected_path, true);
                            *p_open = false;
                        }
                    }
                    ImGui::PopStyleColor();

                    ImGui::SameLine();

                    auto dataset_type = lfs::io::Loader::getDatasetType(selected_path);
                    const char* type_str = (dataset_type == lfs::io::DatasetType::COLMAP) ? "(COLMAP)" : (dataset_type == lfs::io::DatasetType::Transforms) ? "(Transforms)"
                                                                                                                                                            : "(Dataset)";
                    ImGui::TextDisabled("%s", type_str);
                } else if (is_sog_dir) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.5f, 0.1f, 1.0f)); // Orange button
                    if (ImGui::Button(LOC(lichtfeld::Strings::FileBrowser::LOAD_SOG), ImVec2(120 * scale, 0))) {
                        if (on_file_selected_) {
                            on_file_selected_(selected_path, false);
                            *p_open = false;
                        }
                    }
                    ImGui::PopStyleColor();

                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", LOC(FileBrowserExt::SOG_DIRECTORY));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                    if (ImGui::Button(LOC(lichtfeld::Strings::FileBrowser::ENTER_DIR), ImVec2(120 * scale, 0))) {
                        current_path_ = selected_path.string();
                        selected_file_.clear();
                    }
                    ImGui::PopStyleColor();

                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", LOC(FileBrowserExt::NOT_A_DATASET));
                }
            } else {
                auto ext = selected_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".ply") {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                    if (ImGui::Button(LOC(lichtfeld::Strings::FileBrowser::LOAD_PLY), ImVec2(120 * scale, 0))) {
                        if (on_file_selected_) {
                            on_file_selected_(selected_path, false);
                            *p_open = false;
                        }
                    }
                    ImGui::PopStyleColor();
                } else if (ext == ".sog") {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.5f, 0.1f, 1.0f)); // Orange button
                    if (ImGui::Button(LOC(lichtfeld::Strings::FileBrowser::LOAD_SOG), ImVec2(120 * scale, 0))) {
                        if (on_file_selected_) {
                            on_file_selected_(selected_path, false);
                            *p_open = false;
                        }
                    }
                    ImGui::PopStyleColor();
                } else if (selected_path.filename() == "meta.json") {
                    // Check if it's a SOG meta.json
                    bool is_sog_meta = false;
                    auto parent = selected_path.parent_path();
                    if (std::filesystem::exists(parent / "means_l.webp") ||
                        std::filesystem::exists(parent / "means_u.webp") ||
                        std::filesystem::exists(parent / "quats.webp") ||
                        std::filesystem::exists(parent / "scales.webp") ||
                        std::filesystem::exists(parent / "sh0.webp")) {
                        is_sog_meta = true;
                    }

                    if (is_sog_meta) {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.5f, 0.1f, 1.0f)); // Orange button
                        if (ImGui::Button(LOC(lichtfeld::Strings::FileBrowser::LOAD_SOG), ImVec2(120 * scale, 0))) {
                            if (on_file_selected_) {
                                on_file_selected_(selected_path, false);
                                *p_open = false;
                            }
                        }
                        ImGui::PopStyleColor();

                        ImGui::SameLine();
                        ImGui::TextDisabled("%s", LOC(FileBrowserExt::SOG_META));
                    }
                }
            }
        }

        if (!can_load) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();

        if (ImGui::Button(LOC(Common::CANCEL), ImVec2(120 * scale, 0))) {
            *p_open = false;
            selected_file_.clear();
        }

        ImGui::End();
        ImGui::PopStyleColor(2);
    }

    void FileBrowser::setOnFileSelected(std::function<void(const std::filesystem::path&, bool)> callback) {
        on_file_selected_ = callback;
    }

    void FileBrowser::setCurrentPath(const std::filesystem::path& path) {
        current_path_ = path.string();
    }

    void FileBrowser::setSelectedPath(const std::filesystem::path& path) {
        selected_file_ = path.string();
        if (std::filesystem::is_directory(path)) {
            current_path_ = path.string();
        } else {
            current_path_ = path.parent_path().string();
        }
    }
} // namespace lfs::vis::gui