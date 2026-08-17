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

namespace lfs::vis::gui {

    namespace {
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

    std::filesystem::path
    LayoutState::getUserPreferencesPath() {
        const auto paths = lfs::core::UserPaths::resolve();
        if (!paths) {
            LOG_WARN("Unable to resolve UI preference path: {}",
                     lfs::format_for_developer(paths.error()));
            return {};
        }
        return paths->uiPreferencesFile();
    }

    void LayoutState::saveUserPreferences() const {
        if (const auto saved = saveUserPreferencesChecked(); !saved)
            LOG_WARN("Failed to save user UI preferences: {}",
                     lfs::format_for_developer(saved.error()));
    }

    lfs::Status LayoutState::saveUserPreferencesChecked() const {
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return {};

        try {
            const auto paths = lfs::core::UserPaths::resolve();
            if (!paths)
                return lfs::Status::failure(paths.error());

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

            return paths->writeUiPreferencesAtomically(j.dump(2) + '\n');
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): convert JSON serialization failures into a typed status.
            return lfs::Status::failure(uiPreferencesError(e.what()));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): contain non-standard serialization failures at the persistence boundary.
            return lfs::Status::failure(uiPreferencesError());
        }
    }

    void LayoutState::load() {
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return;
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
            const auto legacy_path = getLegacyConfigPath();
            const auto preferences_path = getUserPreferencesPath();
            if (!legacy_path.empty())
                load_file(legacy_path, true);
            if (!preferences_path.empty())
                load_file(preferences_path, false);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load UI state: {}", e.what());
        }
    }

    void LayoutState::setPersistenceEnabled(const bool enabled) noexcept {
        g_persistence_enabled.store(enabled, std::memory_order_release);
    }

} // namespace lfs::vis::gui
