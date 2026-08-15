/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/layout_state.hpp"
#include "core/config_paths.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::vis::gui {

    std::filesystem::path LayoutState::getConfigDir() {
        return lfs::core::user_config_dir();
    }

    std::filesystem::path LayoutState::getLegacyConfigPath() {
        return getConfigDir() / "layout.json";
    }

    std::filesystem::path
    LayoutState::getUserPreferencesPath() {
        return getConfigDir() /
               "ui_preferences.json";
    }

    void LayoutState::saveUserPreferences() const {
        try {
            const auto path =
                getUserPreferencesPath();
            std::filesystem::create_directories(path.parent_path());

            nlohmann::json j;
            if (!file_association.empty())
                j["file_association"] = file_association;

            nlohmann::json vram_hud;
            vram_hud["x"] = vram_hud_x;
            vram_hud["y"] = vram_hud_y;
            vram_hud["width"] = vram_hud_width;
            vram_hud["height"] = vram_hud_height;
            vram_hud["active_tab"] = vram_hud_active_tab;
            vram_hud["collapsed"] = vram_hud_collapsed_paths;
            j["vram_hud"] = vram_hud;

            nlohmann::json perf_hud;
            perf_hud["visible"] = perf_hud_visible;
            perf_hud["expanded"] = perf_hud_expanded;
            j["perf_hud"] = perf_hud;

            std::ofstream file(path);
            if (file) {
                file << j.dump(2);
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save user UI preferences: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save user UI preferences: unknown error");
        }
    }

    void LayoutState::load() {
        try {
            const auto load_file =
                [this](const std::filesystem::path& path,
                       const bool legacy_layout) {
                    if (!std::filesystem::exists(path))
                        return;
                    std::ifstream file(path);
                    if (!file)
                        return;
                    const auto j =
                        nlohmann::json::parse(file);

                    if (legacy_layout) {
                        right_panel_width = j.value(
                            "right_panel_width",
                            right_panel_width);
                        scene_panel_ratio = j.value(
                            "scene_panel_ratio",
                            scene_panel_ratio);
                        python_console_width = j.value(
                            "python_console_width",
                            python_console_width);
                        bottom_dock_height = j.value(
                            "bottom_dock_height",
                            bottom_dock_height);
                        left_dock_width = j.value(
                            "left_dock_width",
                            left_dock_width);
                        show_sequencer = j.value(
                            "show_sequencer",
                            show_sequencer);
                        if (j.contains("windows") &&
                            j["windows"].is_object()) {
                            for (const auto& [key, val] :
                                 j["windows"].items()) {
                                if (val.is_boolean()) {
                                    window_visibility[key] =
                                        val.get<bool>();
                                }
                            }
                        }
                    }

                    file_association = j.value(
                        "file_association",
                        file_association);
                    if (j.contains("vram_hud") &&
                        j["vram_hud"].is_object()) {
                        const auto& vh =
                            j["vram_hud"];
                        vram_hud_x =
                            vh.value("x", vram_hud_x);
                        vram_hud_y =
                            vh.value("y", vram_hud_y);
                        vram_hud_width = vh.value(
                            "width", vram_hud_width);
                        vram_hud_height = vh.value(
                            "height", vram_hud_height);
                        vram_hud_active_tab = vh.value(
                            "active_tab",
                            vram_hud_active_tab);
                        if (vh.contains("collapsed") &&
                            vh["collapsed"].is_array()) {
                            vram_hud_collapsed_paths.clear();
                            for (const auto& entry :
                                 vh["collapsed"]) {
                                if (entry.is_string()) {
                                    vram_hud_collapsed_paths.push_back(
                                        entry.get<std::string>());
                                }
                            }
                        }
                    }

                    if (j.contains("perf_hud") &&
                        j["perf_hud"].is_object()) {
                        const auto& ph = j["perf_hud"];
                        perf_hud_visible = ph.value(
                            "visible", perf_hud_visible);
                        perf_hud_expanded = ph.value(
                            "expanded", perf_hud_expanded);
                    }
                };

            // layout.json is import-only. New writes contain only user-global
            // fields and go to ui_preferences.json.
            load_file(getLegacyConfigPath(), true);
            load_file(getUserPreferencesPath(), false);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load UI state: {}", e.what());
        }
    }

} // namespace lfs::vis::gui
