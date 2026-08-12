/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace lfs::vis::gui {

    struct LayoutState {
        float right_panel_width = 360.0f;
        float scene_panel_ratio = 0.4f;
        float python_console_width = -1.0f;
        float bottom_dock_height = 320.0f;
        float left_dock_width = 320.0f;
        bool show_sequencer = false;
        std::string active_main_tab;
        std::string file_association;
        std::unordered_map<std::string, bool> window_visibility;

        void load(bool log_success = true);
        static std::filesystem::path getConfigDir();
        LFS_VIS_API static void setPersistenceEnabled(bool enabled) noexcept;

    private:
        static std::filesystem::path getConfigPath();
    };

} // namespace lfs::vis::gui
