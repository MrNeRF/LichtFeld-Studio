/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>

namespace gs::gui {

    class MenuBar {
    public:
        MenuBar();
        ~MenuBar();

        // Render the menu bar
        void render();

        // Callbacks for menu actions
        void setOnImportDataset(std::function<void()> callback);
        void setOnOpenProject(std::function<void()> callback);
        void setOnImportPLY(std::function<void()> callback);
        void setOnSaveProjectAs(std::function<void()> callback);
        void setOnSaveProject(std::function<void()> callback);
        void setOnExit(std::function<void()> callback);

        // Render separate windows (call these in your main render loop)
        void renderGettingStartedWindow();
        void renderAboutWindow();
        void renderControlsAndShortcutsWindow();

        void setIsProjectTemp(bool is_temp) { is_project_temp_ = is_temp; }
        [[nodiscard]] bool getIsProjectTemp() const { return is_project_temp_; }

    private:
        void openURL(const char* url);
        // Callbacks
        std::function<void()> on_import_dataset_;
        std::function<void()> on_open_project_;
        std::function<void()> on_import_ply_;
        std::function<void()> on_save_project_as_;
        std::function<void()> on_save_project_;
        std::function<void()> on_exit_;

        // Window states
        bool show_about_window_ = false;
        bool show_controls_and_shortcuts_ = false;
        bool show_getting_started_ = false;

        bool is_project_temp_ = true;
    };

} // namespace gs::gui