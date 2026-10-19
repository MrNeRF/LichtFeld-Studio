/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/layout_state.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/persisted_settings.hpp"
#include "core/user_paths.hpp"
#include <algorithm>
#include <atomic>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace lfs::vis::gui {

    namespace {
        std::atomic<bool> g_persistence_enabled{true};
        std::atomic<bool> g_legacy_layout_backup_notice{false};
        std::mutex g_legacy_layout_backup_mutex;
        std::unordered_map<std::string,
                           std::pair<std::filesystem::file_time_type, std::uintmax_t>>
            g_legacy_layout_backups;

        void backUpDamagedLegacyLayout(const std::filesystem::path& path) {
            std::error_code error;
            const auto modified = lfs::core::native_fs::last_write_time(path, error);
            if (error)
                throw std::runtime_error("legacy layout timestamp could not be read: " + error.message());
            const auto size = lfs::core::native_fs::file_size(path, error);
            if (error)
                throw std::runtime_error("legacy layout size could not be read: " + error.message());
            std::scoped_lock lock(g_legacy_layout_backup_mutex);
            const auto key = lfs::core::path_to_utf8(path);
            const auto signature = std::pair{modified, size};
            if (const auto found = g_legacy_layout_backups.find(key);
                found != g_legacy_layout_backups.end() && found->second == signature)
                return;
            lfs::core::persisted_settings::backupDamaged(path);
            g_legacy_layout_backups[key] = signature;
            g_legacy_layout_backup_notice.store(true, std::memory_order_release);
        }

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

    void LayoutState::saveUserPreferences(bool* backed_up) const {
        if (const auto saved = saveUserPreferencesChecked(backed_up); !saved)
            LOG_WARN("Failed to save user UI preferences: {}",
                     lfs::format_for_developer(saved.error()));
    }

    lfs::Status LayoutState::saveUserPreferencesChecked(bool* backed_up) const {
        if (backed_up)
            *backed_up = false;
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
            vram_hud["opacity"] = vram_hud_opacity;
            vram_hud["snap_corner"] = vram_hud_snap_corner;
            vram_hud["window_seconds"] = vram_hud_window_seconds;
            vram_hud["iteration_axis"] = vram_hud_iteration_axis;
            vram_hud["device_scale"] = vram_hud_device_scale;
            vram_hud["visible_categories"] = vram_hud_visible_categories;
            vram_hud["movers_collapsed"] = vram_hud_movers_collapsed;
            vram_hud["peak_collapsed"] = vram_hud_peak_collapsed;
            j["vram_hud"] = vram_hud;

            nlohmann::json perf_hud;
            perf_hud["visible"] = perf_hud_visible;
            perf_hud["expanded"] = perf_hud_expanded;
            j["perf_hud"] = perf_hud;

            j = lfs::core::persisted_settings::mergeForWrite(
                paths->uiPreferencesFile(), j, "schema_version", 1, backed_up, {}, paths->backupDir());
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
                    if (!lfs::core::native_fs::exists(path))
                        return;
                    std::ifstream file(lfs::core::native_fs::native_path(path));
                    if (!file)
                        return;
                    nlohmann::json j;
                    try {
                        j = nlohmann::json::parse(file);
                        if (!j.is_object())
                            throw std::runtime_error("settings root must be an object");
                    } catch (const std::exception& error) {
                        LOG_WARN("Failed to load UI settings from '{}': {}", lfs::core::path_to_utf8(path), error.what());
                        if (legacy_layout) {
                            try {
                                backUpDamagedLegacyLayout(path);
                            } catch (const std::exception& backup_error) {
                                LOG_WARN("Failed to back up damaged legacy UI layout: {}", backup_error.what());
                            }
                        }
                        return;
                    }
                    const auto read = [&j](const char* key, auto fallback) {
                        return lfs::core::persisted_settings::field(j, key, fallback);
                    };

                    if (legacy_layout) {
                        right_panel_width = read(
                            "right_panel_width",
                            right_panel_width);
                        scene_panel_ratio = read(
                            "scene_panel_ratio",
                            scene_panel_ratio);
                        python_console_width = read(
                            "python_console_width",
                            python_console_width);
                        bottom_dock_height = read(
                            "bottom_dock_height",
                            bottom_dock_height);
                        left_dock_width = read(
                            "left_dock_width",
                            left_dock_width);
                        show_sequencer = read(
                            "show_sequencer",
                            show_sequencer);
                        if (lfs::core::persisted_settings::present(j, "windows") &&
                            j["windows"].is_object()) {
                            for (const auto& [key, val] :
                                 j["windows"].items()) {
                                if (val.is_boolean()) {
                                    window_visibility[key] =
                                        val.get<bool>();
                                } else {
                                    LOG_WARN("Invalid window visibility in UI settings for '{}'", key);
                                }
                            }
                        } else if (lfs::core::persisted_settings::present(j, "windows")) {
                            LOG_WARN("Invalid window visibility map in UI settings");
                        }
                    }

                    file_association = read(
                        "file_association",
                        file_association);
                    if (lfs::core::persisted_settings::present(j, "vram_hud") &&
                        j["vram_hud"].is_object()) {
                        const auto& vh =
                            j["vram_hud"];
                        const auto read_hud = [&vh](const char* key, auto fallback) {
                            return lfs::core::persisted_settings::field(vh, key, fallback);
                        };
                        vram_hud_x =
                            read_hud("x", vram_hud_x);
                        vram_hud_y =
                            read_hud("y", vram_hud_y);
                        vram_hud_width = read_hud(
                            "width", vram_hud_width);
                        vram_hud_height = read_hud(
                            "height", vram_hud_height);
                        vram_hud_active_tab = read_hud(
                            "active_tab",
                            vram_hud_active_tab);
                        vram_hud_opacity = std::clamp(read_hud("opacity", vram_hud_opacity), 0.4f, 1.0f);
                        vram_hud_snap_corner = std::clamp(read_hud("snap_corner", vram_hud_snap_corner), 0, 4);
                        vram_hud_window_seconds = read_hud("window_seconds", vram_hud_window_seconds);
                        vram_hud_iteration_axis = read_hud("iteration_axis", vram_hud_iteration_axis);
                        vram_hud_device_scale = read_hud("device_scale", vram_hud_device_scale);
                        vram_hud_visible_categories = read_hud("visible_categories", vram_hud_visible_categories) & 1023u;
                        vram_hud_movers_collapsed = read_hud("movers_collapsed", vram_hud_movers_collapsed);
                        vram_hud_peak_collapsed = read_hud("peak_collapsed", vram_hud_peak_collapsed);
                        if (lfs::core::persisted_settings::present(vh, "collapsed") &&
                            vh["collapsed"].is_array()) {
                            vram_hud_collapsed_paths.clear();
                            for (const auto& entry :
                                 vh["collapsed"]) {
                                if (entry.is_string()) {
                                    vram_hud_collapsed_paths.push_back(
                                        entry.get<std::string>());
                                } else {
                                    LOG_WARN("Invalid collapsed HUD path in UI settings; skipping it");
                                }
                            }
                        } else if (lfs::core::persisted_settings::present(vh, "collapsed")) {
                            LOG_WARN("Invalid collapsed HUD paths in UI settings");
                        }
                    } else if (lfs::core::persisted_settings::present(j, "vram_hud")) {
                        LOG_WARN("Invalid HUD object in UI settings");
                    }

                    if (lfs::core::persisted_settings::present(j, "perf_hud") &&
                        j["perf_hud"].is_object()) {
                        const auto& ph = j["perf_hud"];
                        const auto read_perf = [&ph](const char* key, auto fallback) {
                            return lfs::core::persisted_settings::field(ph, key, fallback);
                        };
                        perf_hud_visible = read_perf(
                            "visible", perf_hud_visible);
                        perf_hud_expanded = read_perf(
                            "expanded", perf_hud_expanded);
                    } else if (lfs::core::persisted_settings::present(j, "perf_hud")) {
                        LOG_WARN("Invalid performance HUD object in UI settings");
                    }
                };

            // layout.json is import-only. New writes contain only user-global
            // fields and go to ui_preferences.json.
            const auto legacy_path = getLegacyConfigPath();
            const auto preferences_path = getUserPreferencesPath();
            const auto load_independently = [&load_file](const std::filesystem::path& path,
                                                         bool legacy_layout) {
                if (path.empty())
                    return;
                try {
                    load_file(path, legacy_layout);
                } catch (const std::exception& error) {
                    LOG_WARN("Failed to load UI settings from '{}': {}", lfs::core::path_to_utf8(path), error.what());
                }
            };
            load_independently(legacy_path, true);
            load_independently(preferences_path, false);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load UI state: {}", e.what());
        }
    }

    void LayoutState::setPersistenceEnabled(const bool enabled) noexcept {
        g_persistence_enabled.store(enabled, std::memory_order_release);
    }

    bool LayoutState::takeLegacyLayoutBackupNotice() noexcept {
        return g_legacy_layout_backup_notice.exchange(false, std::memory_order_acq_rel);
    }

} // namespace lfs::vis::gui
