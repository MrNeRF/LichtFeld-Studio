/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

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
        float left_dock_width = 320.0f;
        bool show_sequencer = false;
        std::string file_association;
        std::unordered_map<std::string, bool> window_visibility;

        float vram_hud_x = -1.0f;
        float vram_hud_y = -1.0f;
        float vram_hud_width = -1.0f;
        float vram_hud_height = -1.0f;
        std::string vram_hud_active_tab;
        std::vector<std::string> vram_hud_collapsed_paths;
        float vram_hud_opacity = 1.0f;
        int vram_hud_snap_corner = 0;
        int vram_hud_window_seconds = 300;
        bool vram_hud_iteration_axis = false;
        bool vram_hud_device_scale = false;
        unsigned int vram_hud_visible_categories = 1023;
        bool vram_hud_movers_collapsed = false;
        bool vram_hud_peak_collapsed = false;
        bool perf_hud_visible = false;
        bool perf_hud_expanded = true;

        // Writes user-global UI preferences only. Project layout is persisted
        // exclusively in the .licht GUIL chapter.
        void saveUserPreferences() const;
        [[nodiscard]] lfs::Status saveUserPreferencesChecked() const;
        void load();
        static std::filesystem::path getConfigDir();
        LFS_VIS_API static void setPersistenceEnabled(bool enabled) noexcept;

    private:
        static std::filesystem::path getLegacyConfigPath();
        static std::filesystem::path getUserPreferencesPath();
    };

} // namespace lfs::vis::gui
