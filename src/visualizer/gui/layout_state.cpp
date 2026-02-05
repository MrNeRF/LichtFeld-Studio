/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/layout_state.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <cstdlib>
#else
#include <pwd.h>
#include <unistd.h>
#endif

namespace lfs::vis::gui {

    std::filesystem::path LayoutState::getConfigPath() {
        std::filesystem::path config_dir;
#ifdef _WIN32
        const char* appdata = std::getenv("APPDATA");
        if (appdata) {
            config_dir = std::filesystem::path(appdata) / "LichtFeldStudio";
        } else {
            config_dir = std::filesystem::current_path() / "config";
        }
#else
        const char* xdg = std::getenv("XDG_CONFIG_HOME");
        if (xdg) {
            config_dir = std::filesystem::path(xdg) / "LichtFeldStudio";
        } else {
            const char* home = std::getenv("HOME");
            if (!home) {
                struct passwd* pw = getpwuid(getuid());
                if (pw)
                    home = pw->pw_dir;
            }
            if (home) {
                config_dir = std::filesystem::path(home) / ".config" / "LichtFeldStudio";
            } else {
                config_dir = std::filesystem::current_path() / "config";
            }
        }
#endif
        return config_dir / "layout.json";
    }

    void LayoutState::save() const {
        try {
            const auto path = getConfigPath();
            std::filesystem::create_directories(path.parent_path());

            nlohmann::json j;
            j["right_panel_width"] = right_panel_width;
            j["scene_panel_ratio"] = scene_panel_ratio;
            j["python_console_width"] = python_console_width;
            j["show_sequencer"] = show_sequencer;

            nlohmann::json windows;
            for (const auto& [name, visible] : window_visibility) {
                windows[name] = visible;
            }
            j["windows"] = windows;

            std::ofstream file(path);
            if (file) {
                file << j.dump(2);
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save layout state: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save layout state: unknown error");
        }
    }

    void LayoutState::load() {
        try {
            const auto path = getConfigPath();
            if (!std::filesystem::exists(path))
                return;

            std::ifstream file(path);
            if (!file)
                return;

            const auto j = nlohmann::json::parse(file);
            right_panel_width = j.value("right_panel_width", right_panel_width);
            scene_panel_ratio = j.value("scene_panel_ratio", scene_panel_ratio);
            python_console_width = j.value("python_console_width", python_console_width);
            show_sequencer = j.value("show_sequencer", show_sequencer);

            if (j.contains("windows") && j["windows"].is_object()) {
                for (const auto& [key, val] : j["windows"].items()) {
                    if (val.is_boolean()) {
                        window_visibility[key] = val.get<bool>();
                    }
                }
            }

            LOG_INFO("Layout state loaded from {}", path.string());
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load layout state: {}", e.what());
        }
    }

} // namespace lfs::vis::gui
