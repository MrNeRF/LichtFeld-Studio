/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

namespace lfs::vis::gui {

    struct LayoutState {
        float right_panel_width = 300.0f;
        float scene_panel_ratio = 0.4f;
        float python_console_width = -1.0f;
        bool show_sequencer = false;
        std::unordered_map<std::string, bool> window_visibility;

        void save() const;
        void load();

    private:
        static std::filesystem::path getConfigPath();
    };

} // namespace lfs::vis::gui
