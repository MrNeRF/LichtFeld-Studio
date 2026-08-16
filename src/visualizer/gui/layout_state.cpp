/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/layout_state.hpp"
#include "core/config_paths.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>

namespace lfs::vis::gui {

    namespace {
        constexpr int LAYOUT_STATE_VERSION = 1;

        bool migrateLayoutJson(nlohmann::json& layout,
                               int version,
                               const std::filesystem::path& path) {
            while (version < LAYOUT_STATE_VERSION) {
                switch (version) {
                case 0:
                    // Version 0 is the original unversioned format. Its field
                    // names already match version 1, so only stamp the schema
                    // before applying any future migration steps.
                    layout["version"] = 1;
                    version = 1;
                    break;
                default:
                    LOG_WARN("No UI state migration available from version {} for {}",
                             version,
                             path.string());
                    return false;
                }
            }
            return true;
        }

        bool validateAndMigrateLayoutJson(nlohmann::json& layout,
                                          const std::filesystem::path& path) {
            if (!layout.is_object()) {
                LOG_WARN("Failed to load UI state from {}: root must be a JSON object",
                         path.string());
                return false;
            }

            int version = 0;
            if (layout.contains("version")) {
                if (!layout["version"].is_number_integer()) {
                    LOG_WARN("Failed to load UI state from {}: version must be an integer",
                             path.string());
                    return false;
                }
                version = layout["version"].get<int>();
            }
            if (version < 0 || version > LAYOUT_STATE_VERSION) {
                LOG_WARN("Unsupported UI state version {} in {} (current version is {})",
                         version,
                         path.string(),
                         LAYOUT_STATE_VERSION);
                return false;
            }

            const int loaded_version = version;
            if (!migrateLayoutJson(layout, version, path))
                return false;
            if (loaded_version < LAYOUT_STATE_VERSION) {
                LOG_INFO("Migrated UI state from version {} to version {} while reading {}",
                         loaded_version,
                         LAYOUT_STATE_VERSION,
                         path.string());
            }
            return true;
        }

        void applyUserPreferences(LayoutState& state, const nlohmann::json& layout) {
            state.file_association = layout.value("file_association", state.file_association);

            if (layout.contains("vram_hud") && layout["vram_hud"].is_object()) {
                const auto& vram_hud = layout["vram_hud"];
                state.vram_hud_x = vram_hud.value("x", state.vram_hud_x);
                state.vram_hud_y = vram_hud.value("y", state.vram_hud_y);
                state.vram_hud_width = vram_hud.value("width", state.vram_hud_width);
                state.vram_hud_height = vram_hud.value("height", state.vram_hud_height);
                state.vram_hud_active_tab =
                    vram_hud.value("active_tab", state.vram_hud_active_tab);
                if (vram_hud.contains("collapsed") && vram_hud["collapsed"].is_array()) {
                    state.vram_hud_collapsed_paths.clear();
                    for (const auto& entry : vram_hud["collapsed"]) {
                        if (entry.is_string())
                            state.vram_hud_collapsed_paths.push_back(entry.get<std::string>());
                    }
                }
            }

            if (layout.contains("perf_hud") && layout["perf_hud"].is_object()) {
                const auto& perf_hud = layout["perf_hud"];
                state.perf_hud_visible = perf_hud.value("visible", state.perf_hud_visible);
                state.perf_hud_expanded = perf_hud.value("expanded", state.perf_hud_expanded);
            }
        }

        void applyLegacyLayout(LayoutState& state, const nlohmann::json& layout) {
            state.right_panel_width = layout.value("right_panel_width", state.right_panel_width);
            state.scene_panel_ratio = layout.value("scene_panel_ratio", state.scene_panel_ratio);
            state.python_console_width =
                layout.value("python_console_width", state.python_console_width);
            state.bottom_dock_height =
                layout.value("bottom_dock_height", state.bottom_dock_height);
            state.left_dock_width = layout.value("left_dock_width", state.left_dock_width);
            state.show_sequencer = layout.value("show_sequencer", state.show_sequencer);

            if (layout.contains("windows") && layout["windows"].is_object()) {
                for (const auto& [key, value] : layout["windows"].items()) {
                    if (value.is_boolean())
                        state.window_visibility[key] = value.get<bool>();
                }
            }
        }

        bool loadLayoutFile(LayoutState& state,
                            const std::filesystem::path& path,
                            const bool legacy_layout) {
            if (!std::filesystem::exists(path))
                return true;

            std::ifstream file(path);
            if (!file) {
                LOG_WARN("Failed to open UI state file {}", path.string());
                return false;
            }

            auto layout = nlohmann::json::parse(file);
            if (!validateAndMigrateLayoutJson(layout, path))
                return false;

            if (legacy_layout)
                applyLegacyLayout(state, layout);
            applyUserPreferences(state, layout);
            return true;
        }
    } // namespace

    std::filesystem::path LayoutState::getConfigDir() {
        return lfs::core::user_config_dir();
    }

    std::filesystem::path LayoutState::getLegacyConfigPath() {
        return getConfigDir() / "layout.json";
    }

    std::filesystem::path LayoutState::getUserPreferencesPath() {
        return getConfigDir() / "ui_preferences.json";
    }

    bool LayoutState::saveUserPreferences() const {
        return saveUserPreferencesTo(getUserPreferencesPath());
    }

    bool LayoutState::saveUserPreferencesTo(const std::filesystem::path& path) const {
        try {
            if (!path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path());

            nlohmann::json layout;
            layout["version"] = LAYOUT_STATE_VERSION;
            if (!file_association.empty())
                layout["file_association"] = file_association;

            nlohmann::json vram_hud;
            vram_hud["x"] = vram_hud_x;
            vram_hud["y"] = vram_hud_y;
            vram_hud["width"] = vram_hud_width;
            vram_hud["height"] = vram_hud_height;
            vram_hud["active_tab"] = vram_hud_active_tab;
            vram_hud["collapsed"] = vram_hud_collapsed_paths;
            layout["vram_hud"] = std::move(vram_hud);

            nlohmann::json perf_hud;
            perf_hud["visible"] = perf_hud_visible;
            perf_hud["expanded"] = perf_hud_expanded;
            layout["perf_hud"] = std::move(perf_hud);

            std::ofstream file(path);
            if (!file)
                return false;
            file << layout.dump(2);
            return static_cast<bool>(file);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save user UI preferences: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save user UI preferences: unknown error");
        }
        return false;
    }

    bool LayoutState::load() {
        return loadFrom(getLegacyConfigPath(), getUserPreferencesPath());
    }

    bool LayoutState::loadFrom(const std::filesystem::path& legacy_path,
                               const std::filesystem::path& user_preferences_path) {
        try {
            LayoutState loaded = *this;
            if (!loadLayoutFile(loaded, legacy_path, true) ||
                !loadLayoutFile(loaded, user_preferences_path, false)) {
                return false;
            }

            *this = std::move(loaded);
            return true;
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load UI state: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to load UI state: unknown error");
        }
        return false;
    }

} // namespace lfs::vis::gui
