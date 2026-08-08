/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/layout_state.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>

#ifdef _WIN32
#include <cstdlib>
#include <shlobj.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

namespace lfs::vis::gui {

    namespace {
        constexpr int LAYOUT_STATE_VERSION = 1;

        bool migrateLayoutJson(nlohmann::json& layout, int version) {
            while (version < LAYOUT_STATE_VERSION) {
                switch (version) {
                case 0:
                    // Version 0 is the original unversioned layout format. Its
                    // field names already match version 1, so only stamp the
                    // schema version before applying future migration steps.
                    layout["version"] = 1;
                    version = 1;
                    break;
                default:
                    LOG_WARN("No layout state migration available from version {}", version);
                    return false;
                }
            }
            return true;
        }
    } // namespace

    std::filesystem::path LayoutState::getConfigDir() {
        std::filesystem::path config_dir;
#ifdef _WIN32
        wchar_t path[MAX_PATH];
        if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
            config_dir = std::filesystem::path(path) / "LichtFeldStudio";
        } else {
            const char* appdata = std::getenv("APPDATA");
            if (appdata) {
                config_dir = std::filesystem::path(appdata) / "LichtFeldStudio";
            } else {
                config_dir = std::filesystem::current_path() / "config";
            }
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
        return config_dir;
    }

    std::filesystem::path LayoutState::getConfigPath() {
        return getConfigDir() / "layout.json";
    }

    void LayoutState::save() const {
        (void)saveTo(getConfigPath());
    }

    bool LayoutState::saveTo(const std::filesystem::path& path) const {
        try {
            std::filesystem::create_directories(path.parent_path());

            nlohmann::json j;
            j["version"] = LAYOUT_STATE_VERSION;
            j["right_panel_width"] = right_panel_width;
            j["scene_panel_ratio"] = scene_panel_ratio;
            j["python_console_width"] = python_console_width;
            j["bottom_dock_height"] = bottom_dock_height;
            j["show_sequencer"] = show_sequencer;

            if (!file_association.empty())
                j["file_association"] = file_association;

            nlohmann::json windows;
            for (const auto& [name, visible] : window_visibility) {
                windows[name] = visible;
            }
            j["windows"] = windows;

            nlohmann::json vram_hud;
            vram_hud["x"] = vram_hud_x;
            vram_hud["y"] = vram_hud_y;
            vram_hud["width"] = vram_hud_width;
            vram_hud["height"] = vram_hud_height;
            vram_hud["active_tab"] = vram_hud_active_tab;
            vram_hud["collapsed"] = vram_hud_collapsed_paths;
            j["vram_hud"] = vram_hud;

            std::ofstream file(path);
            if (!file)
                return false;
            file << j.dump(2);
            return static_cast<bool>(file);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save layout state: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save layout state: unknown error");
        }
        return false;
    }

    bool LayoutState::load() {
        return loadFrom(getConfigPath());
    }

    bool LayoutState::loadFrom(const std::filesystem::path& path) {
        try {
            if (!std::filesystem::exists(path))
                return true;

            std::ifstream file(path);
            if (!file)
                return false;

            auto j = nlohmann::json::parse(file);
            if (!j.is_object()) {
                LOG_WARN("Failed to load layout state: root must be a JSON object");
                return false;
            }

            int version = 0;
            if (j.contains("version")) {
                if (!j["version"].is_number_integer()) {
                    LOG_WARN("Failed to load layout state: version must be an integer");
                    return false;
                }
                version = j["version"].get<int>();
            }
            if (version < 0 || version > LAYOUT_STATE_VERSION) {
                LOG_WARN("Unsupported layout state version {} (current version is {})",
                         version,
                         LAYOUT_STATE_VERSION);
                return false;
            }
            const int loaded_version = version;
            if (!migrateLayoutJson(j, version))
                return false;

            LayoutState loaded = *this;
            loaded.right_panel_width = j.value("right_panel_width", loaded.right_panel_width);
            loaded.scene_panel_ratio = j.value("scene_panel_ratio", loaded.scene_panel_ratio);
            loaded.python_console_width = j.value("python_console_width", loaded.python_console_width);
            loaded.bottom_dock_height = j.value("bottom_dock_height", loaded.bottom_dock_height);
            loaded.show_sequencer = j.value("show_sequencer", loaded.show_sequencer);
            loaded.file_association = j.value("file_association", loaded.file_association);

            if (j.contains("windows") && j["windows"].is_object()) {
                for (const auto& [key, val] : j["windows"].items()) {
                    if (val.is_boolean()) {
                        loaded.window_visibility[key] = val.get<bool>();
                    }
                }
            }

            if (j.contains("vram_hud") && j["vram_hud"].is_object()) {
                const auto& vh = j["vram_hud"];
                loaded.vram_hud_x = vh.value("x", loaded.vram_hud_x);
                loaded.vram_hud_y = vh.value("y", loaded.vram_hud_y);
                loaded.vram_hud_width = vh.value("width", loaded.vram_hud_width);
                loaded.vram_hud_height = vh.value("height", loaded.vram_hud_height);
                loaded.vram_hud_active_tab = vh.value("active_tab", loaded.vram_hud_active_tab);
                if (vh.contains("collapsed") && vh["collapsed"].is_array()) {
                    loaded.vram_hud_collapsed_paths.clear();
                    for (const auto& entry : vh["collapsed"]) {
                        if (entry.is_string())
                            loaded.vram_hud_collapsed_paths.push_back(entry.get<std::string>());
                    }
                }
            }

            *this = std::move(loaded);
            if (loaded_version < LAYOUT_STATE_VERSION) {
                LOG_INFO("Migrated layout state from version {} to version {}",
                         loaded_version,
                         LAYOUT_STATE_VERSION);
            }
            LOG_INFO("Layout state loaded from {}", path.string());
            return true;
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load layout state: {}", e.what());
        }
        return false;
    }

} // namespace lfs::vis::gui
