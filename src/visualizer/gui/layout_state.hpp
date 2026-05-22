/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {

    struct LayoutState {
        float right_panel_width = 300.0f;
        float scene_panel_ratio = 0.4f;
        float python_console_width = -1.0f;
        float bottom_dock_height = 320.0f;
        bool show_sequencer = false;
        std::string file_association;
        std::unordered_map<std::string, bool> window_visibility;

        float vram_hud_x = -1.0f;
        float vram_hud_y = -1.0f;
        float vram_hud_width = -1.0f;
        float vram_hud_height = -1.0f;
        std::string vram_hud_active_tab;
        std::vector<std::string> vram_hud_collapsed_paths;

        void save() const;
        void load();
        static std::filesystem::path getConfigDir();

    private:
        static std::filesystem::path getConfigPath();
    };

} // namespace lfs::vis::gui
