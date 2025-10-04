/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/menu_bar.hpp"
#include "core/logger.hpp"
#include <imgui.h>

#include <cstdlib> // for system()
#ifdef _WIN32
#include <windows.h> // for ShellExecuteA
#endif

namespace gs::gui {

    MenuBar::MenuBar() {
        LOG_DEBUG("MenuBar created");
    }

    MenuBar::~MenuBar() {
        // Cleanup handled automatically
    }

    void MenuBar::render() {
        // Modern color scheme
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.0f, 8.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12.0f, 6.0f));
        ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.15f, 0.15f, 0.17f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg,   ImVec4(0.12f, 0.12f, 0.14f, 1.0f)); // dark menus
        ImGui::PushStyleColor(ImGuiCol_Header,    ImVec4(0.35f, 0.65f, 1.0f, 0.25f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.45f, 0.75f, 1.0f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive,  ImVec4(0.45f, 0.75f, 1.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_Text,      ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TextDisabled, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));


        if (ImGui::BeginMainMenuBar()) {
            // File menu
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Open Project")) {
                    LOG_DEBUG("Open Project clicked");
                    if (on_open_project_) {
                        on_open_project_();
                    }
                }

                if (ImGui::MenuItem("Import Dataset")) {
                    LOG_DEBUG("Import Dataset clicked");
                    if (on_import_dataset_) {
                        on_import_dataset_();
                    }
                }

                if (ImGui::MenuItem("Import Ply")) {
                    LOG_DEBUG("Import Ply clicked");
                    if (on_import_ply_) {
                        on_import_ply_();
                    }
                }

                ImGui::Separator();

                if (ImGui::MenuItem("Save Project As...")) {
                    LOG_DEBUG("Save Project As clicked");
                    if (on_save_project_as_) {
                        on_save_project_as_();
                    }
                }

                // Disable "Save Project" if project is temporary
                if (ImGui::MenuItem("Save Project", nullptr, false, !is_project_temp_)) {
                    LOG_DEBUG("Save Project clicked");
                    if (on_save_project_) {
                        on_save_project_();
                    }
                }

                ImGui::Separator();

                if (ImGui::MenuItem("Exit")) {
                    LOG_DEBUG("Exit clicked");
                    if (on_exit_) {
                        on_exit_();
                    }
                }

                ImGui::EndMenu();
            }

            // Help menu
            if (ImGui::BeginMenu("Help")) {
                if (ImGui::MenuItem("Getting Started")) {
                    LOG_DEBUG("Getting Started clicked");
                    show_getting_started_ = true;
                }

                if (ImGui::MenuItem("Controls & Shortcuts")) {
                    LOG_DEBUG("Camera Controls and Shortcuts clicked");
                    show_controls_and_shortcuts_ = true;
                }

                if (ImGui::MenuItem("About LichtFeld Studio")) {
                    LOG_DEBUG("About clicked");
                    show_about_window_ = true;
                }

                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }

        ImGui::PopStyleColor(7);
        ImGui::PopStyleVar(2);

        renderGettingStartedWindow();
        renderAboutWindow();
        renderControlsAndShortcutsWindow();
    }

    void MenuBar::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#elif __APPLE__
        std::string cmd = "open " + std::string(url);
        system(cmd.c_str());
#else
        std::string cmd = "xdg-open " + std::string(url);
        system(cmd.c_str());
#endif
    }

    void MenuBar::renderGettingStartedWindow() {
        if (!show_getting_started_) {
            return;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;

        ImGui::SetNextWindowSize(ImVec2(700, 0), ImGuiCond_Once);

        // Modern dark theme with subtle gradient effect
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 12.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.13f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.26f, 0.59f, 0.98f, 0.3f));

        if (ImGui::Begin("Getting Started", &show_getting_started_, window_flags)) {
            // Header with gradient-like color
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // Assumes default font
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "QUICK START GUIDE");
            ImGui::PopFont();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextWrapped("Learn how to prepare datasets and get started with LichtFeld Studio:");
            ImGui::Spacing();
            ImGui::Spacing();

            // Reality Scan video with modern styling
            const char* reality_scan_url = "http://www.youtube.com/watch?v=JWmkhTlbDvg";
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.26f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "*");
            ImGui::SameLine();
            ImGui::TextWrapped("Using Reality Scan to create a dataset");

            ImGui::Indent(25.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", reality_scan_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
                ImGui::SetTooltip("Click to open youtube tutorial");
                ImGui::PopStyleColor();
                if (ImGui::IsItemClicked()) {
                    openURL(reality_scan_url);
                }
            }
            ImGui::Unindent(25.0f);
            ImGui::PopStyleColor(3);

            ImGui::Spacing();

            // Colmap tutorial video
            const char* colmap_tutorial_url = "https://www.youtube.com/watch?v=-3TBbukYN00";
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.26f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "*");
            ImGui::SameLine();
            ImGui::TextWrapped("Beginner Tutorial - Using COLMAP to create a dataset");

            ImGui::Indent(25.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", colmap_tutorial_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
                ImGui::SetTooltip("Click to open youtube tutorial");
                ImGui::PopStyleColor();
                if (ImGui::IsItemClicked()) {
                    openURL(colmap_tutorial_url);
                }
            }
            ImGui::Unindent(25.0f);
            ImGui::PopStyleColor(3);

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // FAQ link
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "FREQUENTLY ASKED QUESTIONS");
            ImGui::Spacing();

            const char* faq_url = "https://github.com/MrNeRF/LichtFeld-Studio/blob/master/docs/docs/faq.md";
            ImGui::Indent(25.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", faq_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
                ImGui::SetTooltip("Click to open FAQ");
                ImGui::PopStyleColor();
                if (ImGui::IsItemClicked()) {
                    openURL(faq_url);
                }
            }
            ImGui::Unindent(25.0f);
        }
        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(3);
    }

    void MenuBar::renderAboutWindow() {
        if (!show_about_window_) {
            return;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;

        ImGui::SetNextWindowSize(ImVec2(700, 0), ImGuiCond_Once);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 12.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.13f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.26f, 0.59f, 0.98f, 0.3f));

        std::string version = "v0.1.3";
        const char* repo_url = "https://github.com/MrNeRF/LichtFeld-Studio";

        if (ImGui::Begin("About LichtFeld Studio", &show_about_window_, window_flags)) {
            // Modern header
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "LICHTFELD STUDIO");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", version.c_str());
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextWrapped(
                "A high-performance C++ and CUDA implementation of 3D Gaussian Splatting, "
                "designed to fuse real and digital content seamlessly. Empowering research and development in "
                "neural rendering with real-time visualization, training, and inspection of neural radiance fields.");

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "RESOURCES");
            ImGui::Spacing();
            ImGui::TextWrapped("Source code, documentation, and project details:");

            ImGui::Indent(25.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", repo_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
                ImGui::SetTooltip("Click to open GitHub repository");
                ImGui::PopStyleColor();
                if (ImGui::IsItemClicked()) {
                    openURL(repo_url);
                }
            }
            ImGui::Unindent(25.0f);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Created by LichtFeld Studio Authors");
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Licensed under GPLv3");
        }
        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(3);
    }

    void MenuBar::renderControlsAndShortcutsWindow() {
        if (!show_controls_and_shortcuts_) {
            return;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 8.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 6.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.13f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.26f, 0.59f, 0.98f, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0.18f, 0.18f, 0.22f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.26f, 0.59f, 0.98f, 0.4f));
        ImGui::PushStyleColor(ImGuiCol_TabActive, ImVec4(0.26f, 0.59f, 0.98f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

        if (ImGui::Begin("Controls & Shortcuts", &show_controls_and_shortcuts_, window_flags)) {
            if (ImGui::BeginTabBar("ControlsTabBar", ImGuiTabBarFlags_None)) {
                // Camera Controls Tab
                if (ImGui::BeginTabItem("Camera Controls")) {
                    ImGui::Spacing();

                    if (ImGui::BeginTable("camera_controls_table", 2,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {

                        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 350.0f);
                        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, 200.0f);
                        ImGui::TableHeadersRow();

                        const ImVec4 actionColor = ImVec4(0.9f, 0.9f, 0.9f, 1.0f);
                        const ImVec4 controlColor = ImVec4(0.4f, 0.7f, 1.0f, 1.0f);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "* Local Translate Camera");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(controlColor, "Left Mouse + Drag");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "* Local Rotate Camera (Pitch/Yaw)");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(controlColor, "Right Mouse + Drag");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "* Rotate Around Scene Center");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(controlColor, "Middle Mouse + Drag");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "* Zoom");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(controlColor, "Mouse Scroll");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "* Roll Camera");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(controlColor, "R + Mouse Scroll");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "* Navigate Scene (Forward/Back/Left/Right)");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(controlColor, "W, S, A, D");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "* Adjust Movement Speed");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(controlColor, "Ctrl + +/-");

                        ImGui::EndTable();
                    }

                    ImGui::Spacing();
                    ImGui::EndTabItem();
                }

                // Shortcuts Tab
                if (ImGui::BeginTabItem("Shortcuts")) {
                    ImGui::Spacing();

                    if (ImGui::BeginTable("shortcuts_table", 2,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {

                        ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 430.0f);
                        ImGui::TableHeadersRow();

                        const ImVec4 keyColor = ImVec4(0.4f, 0.7f, 1.0f, 1.0f);
                        const ImVec4 actionColor = ImVec4(0.9f, 0.9f, 0.9f, 1.0f);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(keyColor, "G");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "> Compare ground truth images");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(keyColor, "F2");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(actionColor, "> Rename ply file");

                        ImGui::EndTable();
                    }

                    ImGui::Spacing();
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
        ImGui::End();

        ImGui::PopStyleColor(10);
        ImGui::PopStyleVar(4);
    }

    void MenuBar::setOnImportDataset(std::function<void()> callback) {
        on_import_dataset_ = std::move(callback);
    }

    void MenuBar::setOnOpenProject(std::function<void()> callback) {
        on_open_project_ = std::move(callback);
    }

    void MenuBar::setOnImportPLY(std::function<void()> callback) {
        on_import_ply_ = std::move(callback);
    }

    void MenuBar::setOnSaveProjectAs(std::function<void()> callback) {
        on_save_project_as_ = std::move(callback);
    }

    void MenuBar::setOnSaveProject(std::function<void()> callback) {
        on_save_project_ = std::move(callback);
    }

    void MenuBar::setOnExit(std::function<void()> callback) {
        on_exit_ = std::move(callback);
    }

} // namespace gs::gui