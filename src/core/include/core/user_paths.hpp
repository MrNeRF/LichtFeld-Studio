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
        bool portable = false;
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

        /**
         * Legacy locations which may be read by a dedicated migration. Callers
         * must never write to these locations through this API.
         */
        [[nodiscard]] std::vector<std::filesystem::path> legacyConfigDirs() const;

        [[nodiscard]] bool usesUnifiedRoot() const noexcept { return unified_root_; }

    private:
        UserPaths(std::filesystem::path config_dir,
                  std::filesystem::path data_dir,
                  std::filesystem::path cache_dir,
                  std::filesystem::path log_dir,
                  std::filesystem::path plugin_dir,
                  std::filesystem::path venv_dir,
                  bool unified_root);

        [[nodiscard]] static UserPaths fromUnifiedRoot(const std::filesystem::path& root);

        std::filesystem::path config_dir_;
        std::filesystem::path data_dir_;
        std::filesystem::path cache_dir_;
        std::filesystem::path log_dir_;
        std::filesystem::path plugin_dir_;
        std::filesystem::path venv_dir_;
        bool unified_root_ = false;
    };

} // namespace lfs::core
