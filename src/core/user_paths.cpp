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

        [[nodiscard]] std::filesystem::path xdgOrHome(const char* const variable,
                                                      const std::filesystem::path& home,
                                                      const char* const fallback) {
            if (const auto path = environmentPath(variable))
                return *path;
            return home / fallback;
        }

        using json = nlohmann::json;

        [[nodiscard]] std::expected<void, std::string> writeTextAtomically(
            const std::filesystem::path& destination, const std::string& contents) {
            const std::lock_guard write_lock(g_atomic_write_mutex);
            std::error_code error;
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error)
                return std::unexpected(std::format("Unable to create directory '{}': {}",
                                                   path_to_utf8(destination.parent_path()), error.message()));

            const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto sequence = g_temporary_file_sequence.fetch_add(1, std::memory_order_relaxed);
            const auto temporary = destination.parent_path() /
                                   std::format("{}.tmp-{}-{}-{}", path_to_utf8(destination.filename()),
                                               currentProcessId(), ticks, sequence);
            {
                std::ofstream file(temporary, std::ios::trunc);
                if (!file)
                    return std::unexpected(std::format("Unable to write temporary file '{}'", path_to_utf8(temporary)));
                file << contents;
                file.close();
                if (!file) {
                    std::filesystem::remove(temporary, error);
                    return std::unexpected(std::format("Unable to finish temporary file '{}'", path_to_utf8(temporary)));
                }
            }

#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                const auto message = std::system_category().message(static_cast<int>(GetLastError()));
                std::filesystem::remove(temporary, error);
                return std::unexpected(std::format("Unable to replace '{}' atomically: {}",
                                                   path_to_utf8(destination), message));
            }
#else
            const int temporary_fd = ::open(temporary.c_str(), O_RDONLY);
            if (temporary_fd < 0 || ::fsync(temporary_fd) != 0) {
                const int sync_error = errno;
                if (temporary_fd >= 0)
                    ::close(temporary_fd);
                std::filesystem::remove(temporary, error);
                return std::unexpected(std::format("Unable to flush temporary file '{}': {}",
                                                   path_to_utf8(temporary),
                                                   std::system_category().message(sync_error)));
            }
            ::close(temporary_fd);

            std::filesystem::rename(temporary, destination, error);
            if (error) {
                std::filesystem::remove(temporary, error);
                return std::unexpected(std::format("Unable to replace '{}' atomically: {}",
                                                   path_to_utf8(destination), error.message()));
            }

            const int directory_fd = ::open(destination.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
            if (directory_fd < 0 || ::fsync(directory_fd) != 0) {
                const int sync_error = errno;
                if (directory_fd >= 0)
                    ::close(directory_fd);
                return std::unexpected(std::format("Unable to flush directory '{}': {}",
                                                   path_to_utf8(destination.parent_path()),
                                                   std::system_category().message(sync_error)));
            }
            ::close(directory_fd);
#endif
            return {};
        }

        [[nodiscard]] std::expected<void, std::string> writeJsonAtomically(
            const std::filesystem::path& destination, const json& value) {
            return writeTextAtomically(destination, value.dump(2) + '\n');
        }

        [[nodiscard]] std::expected<std::optional<std::filesystem::path>, std::string>
        backupAndRemoveFile(const std::filesystem::path& source,
                            const std::filesystem::path& backup_root,
                            const std::string_view category) {
            std::error_code error;
            if (!std::filesystem::exists(source, error)) {
                if (error)
                    return std::unexpected(std::format("Unable to inspect settings file '{}': {}",
                                                       path_to_utf8(source), error.message()));
                return std::nullopt;
            }
            if (!std::filesystem::is_regular_file(source, error) || error)
                return std::unexpected(std::format("Settings reset requires a regular file '{}': {}",
                                                   path_to_utf8(source), error.message()));

            const auto now = std::chrono::system_clock::now().time_since_epoch();
            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
            const auto sequence = g_temporary_file_sequence.fetch_add(1, std::memory_order_relaxed);
            const auto destination = backup_root /
                                     std::format("reset-{}-{}-{}-{}", category, millis,
                                                 currentProcessId(), sequence) /
                                     source.filename();
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error)
                return std::unexpected(std::format("Unable to create reset backup directory '{}': {}",
                                                   path_to_utf8(destination.parent_path()), error.message()));
            if (!std::filesystem::copy_file(source, destination, std::filesystem::copy_options::none, error))
                return std::unexpected(std::format("Unable to back up settings file '{}' to '{}': {}",
                                                   path_to_utf8(source), path_to_utf8(destination), error.message()));
            if (!std::filesystem::exists(destination, error) || error)
                return std::unexpected(std::format("Unable to verify reset backup '{}': {}",
                                                   path_to_utf8(destination), error.message()));
            if (!std::filesystem::remove(source, error) || error)
                return std::unexpected(std::format("Backed up '{}' but could not reset it: {}",
                                                   path_to_utf8(source), error.message()));
            return destination;
        }

        [[nodiscard]] std::expected<void, std::string>
        writeDefaultPreferences(const std::filesystem::path& destination) {
            return writeJsonAtomically(destination, {
                                                        {"schema_version", 1},
                                                        {"language", "en"},
                                                        {"theme", "dark"},
                                                        {"ui_scale", "auto"},
                                                        {"scene_render_scale", 1.0},
                                                        {"mcp", {
                                                                    {"enabled", true},
                                                                    {"expose_network", false},
                                                                    {"port", 45677},
                                                                    {"request_logging", false},
                                                                }},
                                                    });
        }

    } // namespace

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

    std::expected<UserPaths, std::string> UserPaths::resolve(const UserPathOptions& options) {
        if (options.explicit_root && !options.explicit_root->empty())
            return fromUnifiedRoot(*options.explicit_root);

        if (const auto root = environmentPath("LFS_HOME"))
            return fromUnifiedRoot(*root);

#ifdef LFS_BUILD_PORTABLE
        if (!options.portable) {
            try {
                return fromUnifiedRoot(getExecutableDir() / ".lichtfeld");
            } catch (const std::exception& error) {
                return std::unexpected(std::format(
                    "Unable to resolve portable executable directory: {}", error.what()));
            }
        }
#endif

        if (options.portable) {
            if (!options.executable_dir || options.executable_dir->empty())
                return std::unexpected("Portable user storage requires an executable directory");
            return fromUnifiedRoot(*options.executable_dir / ".lichtfeld");
        }

        const auto home = userHomeDirectory();
        if (!home)
            return std::unexpected("Unable to resolve the current user's home directory");

#ifdef _WIN32
        return fromUnifiedRoot(*home / ".lichtfeld");
#else
        const auto config_home = xdgOrHome("XDG_CONFIG_HOME", *home, ".config");
        const auto data_home = xdgOrHome("XDG_DATA_HOME", *home, ".local/share");
        const auto cache_home = xdgOrHome("XDG_CACHE_HOME", *home, ".cache");
        const auto state_home = xdgOrHome("XDG_STATE_HOME", *home, ".local/state");
        const auto plugin_home = *home / ".lichtfeld";
        return UserPaths(
            config_home / "lichtfeld-studio",
            data_home / "lichtfeld-studio",
            cache_home / "lichtfeld-studio",
            state_home / "lichtfeld-studio",
            plugin_home / "plugins",
            plugin_home / "venv",
            false);
#endif
    }

    std::expected<void, std::string> UserPaths::ensureDirectories() const {
        const std::filesystem::path directories[] = {
            config_dir_, data_dir_, cache_dir_, log_dir_, mcpLogDir(), plugin_dir_, venv_dir_,
            keymapDir(), presetDir(), assetLibraryDir(), backupDir()};
        for (const auto& directory : directories) {
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            if (error)
                return std::unexpected(std::format("Unable to create user directory '{}': {}",
                                                   path_to_utf8(directory), error.message()));
        }
        return {};
    }

    std::expected<std::optional<std::filesystem::path>, std::string> UserPaths::resetPreferences() const {
        auto backup = backupAndRemoveFile(preferencesFile(), backupDir(), "preferences");
        if (!backup)
            return std::unexpected(backup.error());
        if (const auto defaults = writeDefaultPreferences(preferencesFile()); !defaults)
            return std::unexpected(defaults.error());
        return *backup;
    }

    std::expected<std::optional<std::filesystem::path>, std::string> UserPaths::resetLayout() const {
        return backupAndRemoveFile(layoutFile(), backupDir(), "layout");
    }

    std::expected<std::optional<std::filesystem::path>, std::string> UserPaths::resetWindowState() const {
        return backupAndRemoveFile(windowStateFile(), backupDir(), "window");
    }

    std::filesystem::path UserPaths::preferencesFile() const { return config_dir_ / "preferences.json"; }
    std::expected<void, std::string>
    UserPaths::writePreferencesAtomically(const std::string& serialized_json) const {
        return writeTextAtomically(preferencesFile(), serialized_json);
    }
    std::expected<void, std::string>
    UserPaths::writeWindowStateAtomically(const std::string& serialized_json) const {
        return writeTextAtomically(windowStateFile(), serialized_json);
    }
    std::filesystem::path UserPaths::layoutFile() const { return config_dir_ / "layout.json"; }
    std::filesystem::path UserPaths::windowStateFile() const { return config_dir_ / "window.json"; }
    std::filesystem::path UserPaths::keymapDir() const { return config_dir_ / "keymaps"; }
    std::filesystem::path UserPaths::presetDir() const { return data_dir_ / "presets"; }
    std::filesystem::path UserPaths::assetLibraryDir() const { return data_dir_ / "asset_library"; }
    std::filesystem::path UserPaths::backupDir() const { return data_dir_ / "backups"; }
    std::filesystem::path UserPaths::mcpLogDir() const { return log_dir_ / "mcp"; }
    std::expected<void, std::string> UserPaths::writeMcpLogAtomically(
        const std::filesystem::path& filename, const std::string& contents) const {
        if (filename.empty() || filename.has_parent_path())
            return std::unexpected("MCP log filename must not contain a directory");
        return writeTextAtomically(mcpLogDir() / filename, contents);
    }
} // namespace lfs::core
