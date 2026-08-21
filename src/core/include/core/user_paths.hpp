/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {

    /** Durably replace a user-owned text file without exposing a partial write. */
    [[nodiscard]] LFS_CORE_API lfs::Status
    writeTextFileAtomically(const std::filesystem::path& destination,
                            const std::string& contents);

    /**
     * Per-invocation overrides for the user-owned storage tree.
     *
     * `explicit_root` is intended for application callers and automated tests.
     * The internal `portable` option uses `.lichtfeld` next to the executable
     * directory supplied by the caller.
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
     * - Linux: `~/.lichtfeld/{config,data,cache,logs,plugins,venv}`.
     * - `LFS_HOME` and explicit roots use one unified root on every OS.
     */
    class LFS_CORE_API UserPaths {
    public:
        [[nodiscard]] static lfs::Result<UserPaths> resolve(const UserPathOptions& options = {});

        /** Create all primary directories. This never creates legacy paths. */
        [[nodiscard]] lfs::Status ensureDirectories() const;

        /** Back up preferences.json, if it exists, then write built-in defaults. */
        [[nodiscard]] lfs::Result<std::optional<std::filesystem::path>>
        resetPreferences() const;

        /** Back up a corrupt preferences.json without writing replacement data. */
        [[nodiscard]] lfs::Result<std::optional<std::filesystem::path>>
        backupCorruptPreferences() const;

        /** Move layout.json to a timestamped backup, if it exists. */
        [[nodiscard]] lfs::Result<std::optional<std::filesystem::path>>
        resetLayout() const;
        [[nodiscard]] lfs::Result<std::optional<std::filesystem::path>>
        resetUiPreferences() const;
        [[nodiscard]] lfs::Result<std::optional<std::filesystem::path>>
        resetWindowState() const;
        [[nodiscard]] lfs::Result<std::optional<std::filesystem::path>>
        resetProjectLifecycle() const;

        [[nodiscard]] const std::filesystem::path& configDir() const noexcept { return config_dir_; }
        [[nodiscard]] const std::filesystem::path& dataDir() const noexcept { return data_dir_; }
        [[nodiscard]] const std::filesystem::path& cacheDir() const noexcept { return cache_dir_; }
        [[nodiscard]] const std::filesystem::path& logDir() const noexcept { return log_dir_; }
        [[nodiscard]] const std::filesystem::path& pluginDir() const noexcept { return plugin_dir_; }
        [[nodiscard]] const std::filesystem::path& venvDir() const noexcept { return venv_dir_; }

        [[nodiscard]] std::filesystem::path preferencesFile() const;
        /** Atomically replace preferences.json with already-serialized JSON. */
        [[nodiscard]] lfs::Status
        writePreferencesAtomically(const std::string& serialized_json) const;
        [[nodiscard]] std::filesystem::path layoutFile() const;
        [[nodiscard]] std::filesystem::path uiPreferencesFile() const;
        [[nodiscard]] lfs::Status
        writeUiPreferencesAtomically(const std::string& serialized_json) const;
        [[nodiscard]] std::filesystem::path projectLifecycleFile() const;
        [[nodiscard]] std::filesystem::path windowStateFile() const;
        [[nodiscard]] lfs::Status
        writeWindowStateAtomically(const std::string& serialized_json) const;
        [[nodiscard]] std::filesystem::path keymapDir() const;
        [[nodiscard]] std::filesystem::path presetDir() const;
        [[nodiscard]] std::filesystem::path assetLibraryDir() const;
        [[nodiscard]] std::filesystem::path backupDir() const;
        /** App-private crash-recovery files: `<root>/recovery`. */
        [[nodiscard]] std::filesystem::path recoveryDir() const;
        [[nodiscard]] std::filesystem::path mcpLogDir() const;
        /** Append one complete JSONL record to an MCP session log below logs/mcp. */
        [[nodiscard]] lfs::Status
        appendMcpLogLine(const std::filesystem::path& filename,
                         const std::string& line) const;
        [[nodiscard]] bool usesUnifiedRoot() const noexcept { return unified_root_; }

    private:
        UserPaths(std::filesystem::path config_dir,
                  std::filesystem::path data_dir,
                  std::filesystem::path cache_dir,
                  std::filesystem::path log_dir,
                  std::filesystem::path plugin_dir,
                  std::filesystem::path venv_dir,
                  bool unified_root);

        [[nodiscard]] static UserPaths fromUnifiedRoot(
            const std::filesystem::path& root);

        std::filesystem::path config_dir_;
        std::filesystem::path data_dir_;
        std::filesystem::path cache_dir_;
        std::filesystem::path log_dir_;
        std::filesystem::path plugin_dir_;
        std::filesystem::path venv_dir_;
        bool unified_root_ = false;
    };

} // namespace lfs::core
