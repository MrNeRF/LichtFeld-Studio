/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {

    /**
     * Per-invocation overrides for the user-owned storage tree.
     *
     * `explicit_root` is intended for --user-dir and automated tests. `portable`
     * uses `.lichtfeld` next to the executable directory supplied by the caller.
     * Neither option reads or mutates the process environment.
     */
    struct UserPathOptions {
        std::optional<std::filesystem::path> explicit_root;
        std::optional<std::filesystem::path> executable_dir;
        // Test-only override for deterministic legacy migration coverage.
        // Production callers leave this empty and use the platform candidates.
        std::vector<std::filesystem::path> legacy_config_dirs;
        // Test-only override. Production callers use the profile policy.
        std::optional<bool> automatic_legacy_migration;
        bool portable = false;
    };

    enum class LegacyGuiSettingStatus {
        Available,
        Migrated,
        AlreadyMigrated,
        Invalid,
        DestinationInvalid,
    };

    struct LegacyGuiSetting {
        std::string kind;
        std::filesystem::path source;
        std::filesystem::path destination;
        LegacyGuiSettingStatus status = LegacyGuiSettingStatus::Available;
        std::string detail;
    };

    /**
     * Resolved locations for user configuration, durable data, and disposable
     * cache. The default follows the platform policy:
     *
     * - Windows: `%USERPROFILE%/.lichtfeld/{config,data,cache,logs}`.
     * - Linux: XDG config/data/cache/state locations, with plugins kept under
     *   `~/.lichtfeld` for first-generation compatibility.
     * - `LFS_HOME`, --user-dir, and --portable use one unified root on every OS.
     */
    class LFS_CORE_API UserPaths {
    public:
        [[nodiscard]] static std::expected<UserPaths, std::string> resolve(const UserPathOptions& options = {});

        /** Create all primary directories. This never creates legacy paths. */
        [[nodiscard]] std::expected<void, std::string> ensureDirectories() const;

        /**
         * Copy recognized legacy GUI settings into the resolved tree.
         *
         * Existing destination files are never replaced and legacy files are
         * never renamed or deleted. The returned entries report the outcome
         * for every recognized legacy artifact.
         */
        [[nodiscard]] std::expected<std::vector<LegacyGuiSetting>, std::string>
        inspectLegacyGuiSettings() const;

        /**
         * Migrate recognized legacy GUI settings once. The automatic startup
         * path uses force=false; a completed manifest prevents repeat scans.
         */
        [[nodiscard]] std::expected<std::vector<LegacyGuiSetting>, std::string>
        migrateLegacyGuiSettings(bool force = false) const;

        /**
         * Copy recognized legacy files into a timestamped archive, verify the
         * copies, then remove only sources already represented by a valid
         * current setting. This never touches current settings, plugin, or
         * virtual-environment directories.
         */
        [[nodiscard]] std::expected<std::vector<std::filesystem::path>, std::string>
        archiveLegacyGuiSettings() const;

        /** Back up preferences.json, if it exists, then write built-in defaults. */
        [[nodiscard]] std::expected<std::optional<std::filesystem::path>, std::string>
        resetPreferences() const;

        /** Move layout.json to a timestamped backup, if it exists. */
        [[nodiscard]] std::expected<std::optional<std::filesystem::path>, std::string>
        resetLayout() const;

        [[nodiscard]] const std::filesystem::path& configDir() const noexcept { return config_dir_; }
        [[nodiscard]] const std::filesystem::path& dataDir() const noexcept { return data_dir_; }
        [[nodiscard]] const std::filesystem::path& cacheDir() const noexcept { return cache_dir_; }
        [[nodiscard]] const std::filesystem::path& logDir() const noexcept { return log_dir_; }
        [[nodiscard]] const std::filesystem::path& pluginDir() const noexcept { return plugin_dir_; }
        [[nodiscard]] const std::filesystem::path& venvDir() const noexcept { return venv_dir_; }

        [[nodiscard]] std::filesystem::path preferencesFile() const;
        [[nodiscard]] std::filesystem::path layoutFile() const;
        [[nodiscard]] std::filesystem::path keymapDir() const;
        [[nodiscard]] std::filesystem::path presetDir() const;
        [[nodiscard]] std::filesystem::path assetLibraryDir() const;
        [[nodiscard]] std::filesystem::path backupDir() const;
        [[nodiscard]] std::filesystem::path migrationDir() const;
        [[nodiscard]] std::filesystem::path migrationManifestFile() const;

        /**
         * Legacy locations which may be read by a dedicated migration. Callers
         * must never write to these locations through this API.
         */
        [[nodiscard]] std::vector<std::filesystem::path> legacyConfigDirs() const;

        [[nodiscard]] bool usesUnifiedRoot() const noexcept { return unified_root_; }
        [[nodiscard]] bool allowsAutomaticLegacyMigration() const noexcept {
            return automatic_legacy_migration_;
        }

    private:
        UserPaths(std::filesystem::path config_dir,
                  std::filesystem::path data_dir,
                  std::filesystem::path cache_dir,
                  std::filesystem::path log_dir,
                  std::filesystem::path plugin_dir,
                  std::filesystem::path venv_dir,
                  std::vector<std::filesystem::path> legacy_config_dirs,
                  bool unified_root,
                  bool automatic_legacy_migration);

        [[nodiscard]] static UserPaths fromUnifiedRoot(
            const std::filesystem::path& root,
            std::vector<std::filesystem::path> legacy_config_dirs = {},
            bool automatic_legacy_migration = false);

        std::filesystem::path config_dir_;
        std::filesystem::path data_dir_;
        std::filesystem::path cache_dir_;
        std::filesystem::path log_dir_;
        std::filesystem::path plugin_dir_;
        std::filesystem::path venv_dir_;
        std::vector<std::filesystem::path> legacy_config_dirs_;
        bool unified_root_ = false;
        bool automatic_legacy_migration_ = false;
    };

} // namespace lfs::core
