/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/user_paths.hpp"

#include "core/environment.hpp"
#include "core/executable_path.hpp"
#include "path_utils.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <mutex>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace lfs::core {

    namespace {

        std::atomic<std::uint64_t> g_temporary_file_sequence{0};
        std::mutex g_atomic_write_mutex;
        std::mutex g_mcp_log_append_mutex;

        [[nodiscard]] std::uint64_t currentProcessId() noexcept {
#ifdef _WIN32
            return static_cast<std::uint64_t>(::GetCurrentProcessId());
#else
            return static_cast<std::uint64_t>(::getpid());
#endif
        }

        [[nodiscard]] std::optional<std::filesystem::path> environmentPath(const char* const name) {
            if (const auto value = environment::value(name))
                return utf8_to_path(std::string(*value));
            return std::nullopt;
        }

        [[nodiscard]] std::optional<std::filesystem::path> userHomeDirectory() {
#ifdef _WIN32
            if (const auto path = environmentPath("USERPROFILE"))
                return path;
            if (const auto path = environmentPath("HOME"))
                return path;
#else
            if (const auto path = environmentPath("HOME"))
                return path;
#endif
            return std::nullopt;
        }

        using json = nlohmann::json;

        [[nodiscard]] lfs::Error userPathError(
            const lfs::ErrorCode code,
            std::string user_message,
            std::string detail,
            const std::filesystem::path& path = {}) {
            lfs::SmallFields fields;
            if (!path.empty())
                fields.add("path", path_to_utf8(path));
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = std::move(user_message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
            });
        }

        [[nodiscard]] lfs::Status failStatus(
            const lfs::ErrorCode code,
            std::string user_message,
            std::string detail,
            const std::filesystem::path& path = {}) {
            return lfs::Status::failure(userPathError(
                code, std::move(user_message), std::move(detail), path));
        }

        [[nodiscard]] lfs::Status writeTextAtomicallyImpl(
            const std::filesystem::path& destination, const std::string& contents) {
            const std::lock_guard write_lock(g_atomic_write_mutex);
            std::error_code error;
            auto directory = destination.parent_path();
            if (directory.empty()) {
                directory = std::filesystem::current_path(error);
                if (error)
                    return failStatus(
                        lfs::ErrorCode::Unavailable,
                        "The user settings location is unavailable.",
                        std::format("Unable to resolve the current directory: {}", error.message()),
                        destination);
            }
            std::filesystem::create_directories(directory, error);
            if (error)
                return failStatus(
                    lfs::ErrorCode::PermissionDenied,
                    "The user settings directory could not be created.",
                    std::format("Unable to create directory '{}': {}",
                                path_to_utf8(directory), error.message()),
                    directory);

            const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto sequence = g_temporary_file_sequence.fetch_add(1, std::memory_order_relaxed);
            const auto temporary = directory /
                                   std::format("{}.tmp-{}-{}-{}", path_to_utf8(destination.filename()),
                                               currentProcessId(), ticks, sequence);
            {
                // Atomic publication must preserve the caller's bytes exactly.
                // Text mode rewrites LF to CRLF on Windows, which corrupts
                // byte-oriented payloads such as captured log tails.
                std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
                if (!file)
                    return failStatus(
                        lfs::ErrorCode::PermissionDenied,
                        "The user settings file could not be saved.",
                        std::format("Unable to write temporary file '{}'", path_to_utf8(temporary)),
                        temporary);
                file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
                file.close();
                if (!file) {
                    std::filesystem::remove(temporary, error);
                    return failStatus(
                        lfs::ErrorCode::DataLoss,
                        "The user settings file could not be saved.",
                        std::format("Unable to finish temporary file '{}'", path_to_utf8(temporary)),
                        temporary);
                }
            }

#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                const auto message = std::system_category().message(static_cast<int>(GetLastError()));
                std::filesystem::remove(temporary, error);
                return failStatus(
                    lfs::ErrorCode::PermissionDenied,
                    "The user settings file could not be replaced.",
                    std::format("Unable to replace '{}' atomically: {}",
                                path_to_utf8(destination), message),
                    destination);
            }
#else
            const int temporary_fd = ::open(temporary.c_str(), O_RDONLY);
            if (temporary_fd < 0 || ::fsync(temporary_fd) != 0) {
                const int sync_error = errno;
                if (temporary_fd >= 0)
                    ::close(temporary_fd);
                std::filesystem::remove(temporary, error);
                return failStatus(
                    lfs::ErrorCode::DataLoss,
                    "The user settings file could not be synchronized.",
                    std::format("Unable to flush temporary file '{}': {}",
                                path_to_utf8(temporary),
                                std::system_category().message(sync_error)),
                    temporary);
            }
            ::close(temporary_fd);

            std::filesystem::rename(temporary, destination, error);
            if (error) {
                std::filesystem::remove(temporary, error);
                return failStatus(
                    lfs::ErrorCode::PermissionDenied,
                    "The user settings file could not be replaced.",
                    std::format("Unable to replace '{}' atomically: {}",
                                path_to_utf8(destination), error.message()),
                    destination);
            }

            const int directory_fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
            if (directory_fd < 0 || ::fsync(directory_fd) != 0) {
                const int sync_error = errno;
                if (directory_fd >= 0)
                    ::close(directory_fd);
                return failStatus(
                    lfs::ErrorCode::DataLoss,
                    "The user settings directory could not be synchronized.",
                    std::format("Unable to flush directory '{}': {}",
                                path_to_utf8(directory),
                                std::system_category().message(sync_error)),
                    directory);
            }
            ::close(directory_fd);
#endif
            return {};
        }

        [[nodiscard]] lfs::Status writeJsonAtomically(
            const std::filesystem::path& destination, const json& value) {
            return writeTextAtomicallyImpl(destination, value.dump(2) + '\n');
        }

        [[nodiscard]] lfs::Result<std::optional<std::filesystem::path>>
        backupAndRemoveFile(const std::filesystem::path& source,
                            const std::filesystem::path& backup_root,
                            const std::string_view category) {
            std::error_code error;
            if (!std::filesystem::exists(source, error)) {
                if (error)
                    return userPathError(
                        lfs::ErrorCode::Unavailable,
                        "The user settings file could not be inspected.",
                        std::format("Unable to inspect settings file '{}': {}",
                                    path_to_utf8(source), error.message()),
                        source);
                return std::optional<std::filesystem::path>{};
            }
            if (!std::filesystem::is_regular_file(source, error) || error)
                return userPathError(
                    lfs::ErrorCode::InvalidArgument,
                    "The user settings file cannot be reset.",
                    std::format("Settings reset requires a regular file '{}': {}",
                                path_to_utf8(source), error.message()),
                    source);

            const auto now = std::chrono::system_clock::now().time_since_epoch();
            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
            const auto sequence = g_temporary_file_sequence.fetch_add(1, std::memory_order_relaxed);
            const auto destination = backup_root /
                                     std::format("reset-{}-{}-{}-{}", category, millis,
                                                 currentProcessId(), sequence) /
                                     source.filename();
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error)
                return userPathError(
                    lfs::ErrorCode::PermissionDenied,
                    "The settings backup directory could not be created.",
                    std::format("Unable to create reset backup directory '{}': {}",
                                path_to_utf8(destination.parent_path()), error.message()),
                    destination.parent_path());
            if (!std::filesystem::copy_file(source, destination, std::filesystem::copy_options::none, error))
                return userPathError(
                    lfs::ErrorCode::DataLoss,
                    "The user settings backup could not be created.",
                    std::format("Unable to back up settings file '{}' to '{}': {}",
                                path_to_utf8(source), path_to_utf8(destination), error.message()),
                    source);
            if (!std::filesystem::exists(destination, error) || error)
                return userPathError(
                    lfs::ErrorCode::DataLoss,
                    "The user settings backup could not be verified.",
                    std::format("Unable to verify reset backup '{}': {}",
                                path_to_utf8(destination), error.message()),
                    destination);
            if (!std::filesystem::remove(source, error) || error)
                return userPathError(
                    lfs::ErrorCode::PermissionDenied,
                    "The user settings file was backed up but could not be reset.",
                    std::format("Backed up '{}' but could not reset it: {}",
                                path_to_utf8(source), error.message()),
                    source);
            return std::optional<std::filesystem::path>{destination};
        }

        [[nodiscard]] lfs::Status
        writeDefaultPreferences(const std::filesystem::path& destination) {
            return writeJsonAtomically(destination, {
                                                        {"schema_version", 1},
                                                        {"language", "en"},
                                                        {"theme", "dark"},
                                                        {"ui_scale", "auto"},
                                                        {"mcp", {
                                                                    {"enabled", true},
                                                                    {"expose_network", false},
                                                                    {"port", 45677},
                                                                    {"request_logging", false},
                                                                }},
                                                    });
        }

    } // namespace

    lfs::Status writeTextFileAtomically(
        const std::filesystem::path& destination, const std::string& contents) {
        return writeTextAtomicallyImpl(destination, contents);
    }

    UserPaths::UserPaths(std::filesystem::path config_dir,
                         std::filesystem::path data_dir,
                         std::filesystem::path cache_dir,
                         std::filesystem::path log_dir,
                         std::filesystem::path plugin_dir,
                         std::filesystem::path venv_dir,
                         const bool unified_root)
        : config_dir_(std::move(config_dir)),
          data_dir_(std::move(data_dir)),
          cache_dir_(std::move(cache_dir)),
          log_dir_(std::move(log_dir)),
          plugin_dir_(std::move(plugin_dir)),
          venv_dir_(std::move(venv_dir)),
          unified_root_(unified_root) {}

    UserPaths UserPaths::fromUnifiedRoot(const std::filesystem::path& root) {
        return UserPaths(
            root / "config",
            root / "data",
            root / "cache",
            root / "logs",
            root / "plugins",
            root / "venv",
            true);
    }

    lfs::Result<UserPaths> UserPaths::resolve(const UserPathOptions& options) {
        if (options.explicit_root && !options.explicit_root->empty())
            return fromUnifiedRoot(*options.explicit_root);

        if (const auto root = environmentPath("LFS_HOME"))
            return fromUnifiedRoot(*root);

#ifdef LFS_BUILD_PORTABLE
        if (!options.portable) {
            try {
                return fromUnifiedRoot(getExecutableDir() / ".lichtfeld");
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): convert executable path failures into a typed result.
                return userPathError(
                    lfs::ErrorCode::Unavailable,
                    "Portable user storage could not be resolved.",
                    std::format("Unable to resolve portable executable directory: {}", error.what()));
            }
        }
#endif

        if (options.portable) {
            if (!options.executable_dir || options.executable_dir->empty())
                return userPathError(
                    lfs::ErrorCode::InvalidArgument,
                    "Portable user storage requires an executable directory.",
                    "UserPathOptions.portable was set without executable_dir");
            return fromUnifiedRoot(*options.executable_dir / ".lichtfeld");
        }

        const auto home = userHomeDirectory();
        if (!home)
            return userPathError(
                lfs::ErrorCode::Unavailable,
                "The current user's home directory is unavailable.",
                "Neither the platform home variable nor a user-path override is available");

        return fromUnifiedRoot(*home / ".lichtfeld");
    }

    lfs::Status UserPaths::ensureDirectories() const {
        const std::filesystem::path directories[] = {
            config_dir_, data_dir_, cache_dir_, log_dir_, plugin_dir_, venv_dir_,
            keymapDir(), presetDir(), assetLibraryDir(), backupDir()};
        for (const auto& directory : directories) {
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            if (error)
                return failStatus(
                    lfs::ErrorCode::PermissionDenied,
                    "A required user directory could not be created.",
                    std::format("Unable to create user directory '{}': {}",
                                path_to_utf8(directory), error.message()),
                    directory);
        }
        return {};
    }

    lfs::Result<std::optional<std::filesystem::path>> UserPaths::resetPreferences() const {
        auto backup = backupAndRemoveFile(preferencesFile(), backupDir(), "preferences");
        if (!backup)
            return std::move(backup).error();
        if (const auto defaults = writeDefaultPreferences(preferencesFile()); !defaults)
            return defaults.error();
        return *backup;
    }

    lfs::Result<std::optional<std::filesystem::path>> UserPaths::backupCorruptPreferences() const {
        return backupAndRemoveFile(preferencesFile(), backupDir(), "corrupt-preferences");
    }

    lfs::Result<std::optional<std::filesystem::path>> UserPaths::resetLayout() const {
        return backupAndRemoveFile(layoutFile(), backupDir(), "layout");
    }

    lfs::Result<std::optional<std::filesystem::path>> UserPaths::resetUiPreferences() const {
        return backupAndRemoveFile(uiPreferencesFile(), backupDir(), "ui-preferences");
    }

    lfs::Result<std::optional<std::filesystem::path>> UserPaths::resetWindowState() const {
        return backupAndRemoveFile(windowStateFile(), backupDir(), "window");
    }

    lfs::Result<std::optional<std::filesystem::path>> UserPaths::resetProjectLifecycle() const {
        return backupAndRemoveFile(projectLifecycleFile(), backupDir(), "project-lifecycle");
    }

    std::filesystem::path UserPaths::preferencesFile() const { return config_dir_ / "preferences.json"; }
    lfs::Status
    UserPaths::writePreferencesAtomically(const std::string& serialized_json) const {
        return writeTextFileAtomically(preferencesFile(), serialized_json);
    }
    lfs::Status
    UserPaths::writeWindowStateAtomically(const std::string& serialized_json) const {
        return writeTextFileAtomically(windowStateFile(), serialized_json);
    }
    std::filesystem::path UserPaths::layoutFile() const { return config_dir_ / "layout.json"; }
    std::filesystem::path UserPaths::uiPreferencesFile() const { return config_dir_ / "ui_preferences.json"; }
    lfs::Status
    UserPaths::writeUiPreferencesAtomically(const std::string& serialized_json) const {
        return writeTextFileAtomically(uiPreferencesFile(), serialized_json);
    }
    std::filesystem::path UserPaths::projectLifecycleFile() const {
        return config_dir_ / "project_lifecycle.json";
    }
    std::filesystem::path UserPaths::windowStateFile() const { return config_dir_ / "window.json"; }
    std::filesystem::path UserPaths::keymapDir() const { return config_dir_ / "keymaps"; }
    std::filesystem::path UserPaths::presetDir() const { return data_dir_ / "presets"; }
    std::filesystem::path UserPaths::assetLibraryDir() const { return data_dir_ / "asset_library"; }
    std::filesystem::path UserPaths::backupDir() const { return data_dir_ / "backups"; }
    std::filesystem::path UserPaths::mcpLogDir() const { return log_dir_ / "mcp"; }
    lfs::Status UserPaths::appendMcpLogLine(
        const std::filesystem::path& filename, const std::string& line) const {
        if (filename.empty() || filename != filename.filename() ||
            filename == "." || filename == "..")
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = "The MCP log filename is invalid.",
                .detail = "MCP log filenames must not contain a directory",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));

        const std::lock_guard append_lock(g_mcp_log_append_mutex);
        const auto directory = mcpLogDir();
        const auto destination = directory / filename;
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error)
            return failStatus(
                lfs::ErrorCode::PermissionDenied,
                "The MCP log directory could not be created.",
                std::format("Unable to create MCP log directory '{}': {}",
                            path_to_utf8(directory), error.message()),
                directory);

        std::ofstream output(destination, std::ios::binary | std::ios::app);
        if (!output.is_open())
            return failStatus(
                lfs::ErrorCode::PermissionDenied,
                "The MCP session log could not be opened.",
                std::format("Unable to open MCP log '{}' for append",
                            path_to_utf8(destination)),
                destination);
        output.write(line.data(), static_cast<std::streamsize>(line.size()));
        output.put('\n');
        output.flush();
        if (!output)
            return failStatus(
                lfs::ErrorCode::DataLoss,
                "The MCP session log could not be written.",
                std::format("Unable to append a complete record to MCP log '{}'",
                            path_to_utf8(destination)),
                destination);
        return {};
    }
} // namespace lfs::core
