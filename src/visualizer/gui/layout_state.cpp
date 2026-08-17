/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/layout_state.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/user_paths.hpp"
#include <atomic>
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>

namespace lfs::vis::gui {

    namespace {
        constexpr int LAYOUT_STATE_VERSION = 1;
        std::atomic<bool> g_persistence_enabled{true};

        [[nodiscard]] lfs::Error uiPreferencesError(std::string detail = {}) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = std::string(LOC("preferences.ui_preferences_save_failed")),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

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

        [[nodiscard]] nlohmann::json serializeUserPreferences(const LayoutState& state) {
            nlohmann::json layout;
            layout["version"] = LAYOUT_STATE_VERSION;
            if (!state.file_association.empty())
                layout["file_association"] = state.file_association;

            layout["vram_hud"] = {
                {"x", state.vram_hud_x},
                {"y", state.vram_hud_y},
                {"width", state.vram_hud_width},
                {"height", state.vram_hud_height},
                {"active_tab", state.vram_hud_active_tab},
                {"collapsed", state.vram_hud_collapsed_paths},
            };
            layout["perf_hud"] = {
                {"visible", state.perf_hud_visible},
                {"expanded", state.perf_hud_expanded},
            };
            return layout;
        }
    } // namespace

    std::filesystem::path LayoutState::getConfigDir() {
        const auto paths = lfs::core::UserPaths::resolve();
        if (!paths) {
            LOG_WARN("Unable to resolve UI preference directory: {}",
                     lfs::format_for_developer(paths.error()));
            return {};
        }
        return paths->configDir();
    }

    std::filesystem::path LayoutState::getLegacyConfigPath() {
        const auto directory = getConfigDir();
        return directory.empty() ? std::filesystem::path{}
                                 : directory / "layout.json";
    }

    std::filesystem::path LayoutState::getUserPreferencesPath() {
        const auto paths = lfs::core::UserPaths::resolve();
        if (!paths) {
            LOG_WARN("Unable to resolve UI preference path: {}",
                     lfs::format_for_developer(paths.error()));
            return {};
        }
        return paths->uiPreferencesFile();
    }

    bool LayoutState::saveUserPreferences() const {
        const auto saved = saveUserPreferencesChecked();
        if (!saved) {
            LOG_WARN("Failed to save user UI preferences: {}",
                     lfs::format_for_developer(saved.error()));
        }
        return saved.has_value();
    }

    lfs::Status LayoutState::saveUserPreferencesChecked() const {
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return {};

        try {
            const auto paths = lfs::core::UserPaths::resolve();
            if (!paths)
                return lfs::Status::failure(paths.error());

            return paths->writeUiPreferencesAtomically(
                serializeUserPreferences(*this).dump(2) + '\n');
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): convert JSON serialization failures into a typed status.
            return lfs::Status::failure(uiPreferencesError(e.what()));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): contain non-standard serialization failures at the persistence boundary.
            return lfs::Status::failure(uiPreferencesError());
        }
    }

    bool LayoutState::saveUserPreferencesTo(const std::filesystem::path& path) const {
        try {
            const auto saved = lfs::core::writeTextFileAtomically(
                path, serializeUserPreferences(*this).dump(2) + '\n');
            if (!saved) {
                LOG_WARN("Failed to save user UI preferences to {}: {}",
                         path.string(),
                         lfs::format_for_developer(saved.error()));
            }
            return saved.has_value();
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save user UI preferences to {}: {}", path.string(), e.what());
        } catch (...) {
            LOG_WARN("Failed to save user UI preferences to {}: unknown error", path.string());
        }
        return false;
    }

    bool LayoutState::load() {
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return true;

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

    void LayoutState::setPersistenceEnabled(const bool enabled) noexcept {
        g_persistence_enabled.store(enabled, std::memory_order_release);
    }

} // namespace lfs::vis::gui
