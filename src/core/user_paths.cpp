/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/user_paths.hpp"

#include "core/environment.hpp"
#include "path_utils.hpp"

#include <chrono>
#include <format>
#include <fstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace lfs::core {

    namespace {

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

        [[nodiscard]] bool isMissingPathError(const std::error_code& error) {
            return error == std::errc::no_such_file_or_directory ||
                   error == std::errc::not_a_directory;
        }

        [[nodiscard]] std::expected<bool, std::string> copyFileIfMissing(
            const std::filesystem::path& source,
            const std::filesystem::path& destination) {
            std::error_code error;
            if (!std::filesystem::is_regular_file(source, error)) {
                if (error && !isMissingPathError(error))
                    return std::unexpected(std::format("Unable to inspect legacy settings file '{}': {}",
                                                        path_to_utf8(source), error.message()));
                return false;
            }
            if (std::filesystem::exists(destination, error)) {
                if (error)
                    return std::unexpected(std::format("Unable to inspect destination settings file '{}': {}",
                                                        path_to_utf8(destination), error.message()));
                return false;
            }
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error)
                return std::unexpected(std::format("Unable to create settings directory '{}': {}",
                                                    path_to_utf8(destination.parent_path()), error.message()));
            if (!std::filesystem::copy_file(source, destination, std::filesystem::copy_options::none, error)) {
                return std::unexpected(std::format("Unable to copy legacy settings file '{}' to '{}': {}",
                                                    path_to_utf8(source), path_to_utf8(destination), error.message()));
            }
            return true;
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
            const auto destination = backup_root / std::format("reset-{}-{}", category, millis) / source.filename();
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
            std::error_code error;
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error)
                return std::unexpected(std::format("Unable to create preferences directory '{}': {}",
                                                    path_to_utf8(destination.parent_path()), error.message()));

            std::ofstream file(destination, std::ios::trunc);
            if (!file)
                return std::unexpected(std::format("Unable to write default preferences '{}'",
                                                    path_to_utf8(destination)));
            file << "{\n  \"schema_version\": 1,\n  \"theme\": \"dark\",\n  \"ui_scale\": \"auto\"\n}\n";
            file.close();
            if (!file)
                return std::unexpected(std::format("Unable to finish writing default preferences '{}'",
                                                    path_to_utf8(destination)));
            return {};
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
            config_dir_, data_dir_, cache_dir_, log_dir_, plugin_dir_, venv_dir_,
            keymapDir(), presetDir(), assetLibraryDir(), backupDir(), migrationDir()};
        for (const auto& directory : directories) {
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            if (error)
                return std::unexpected(std::format("Unable to create user directory '{}': {}",
                                                    path_to_utf8(directory), error.message()));
        }
        return {};
    }

    std::expected<std::vector<std::filesystem::path>, std::string>
    UserPaths::migrateLegacyGuiSettings() const {
        std::vector<std::filesystem::path> copied;
        for (const auto& legacy_dir : legacyConfigDirs()) {
            const auto layout_result = copyFileIfMissing(legacy_dir / "layout.json", layoutFile());
            if (!layout_result)
                return std::unexpected(layout_result.error());
            if (*layout_result)
                copied.push_back(layoutFile());

            const auto legacy_profiles = legacy_dir / "input_profiles";
            std::error_code error;
            if (!std::filesystem::is_directory(legacy_profiles, error)) {
                if (error && !isMissingPathError(error))
                    return std::unexpected(std::format("Unable to inspect legacy input profiles '{}': {}",
                                                        path_to_utf8(legacy_profiles), error.message()));
                continue;
            }

            for (std::filesystem::recursive_directory_iterator it(legacy_profiles, error), end;
                 it != end && !error;
                 it.increment(error)) {
                if (!it->is_regular_file(error)) {
                    if (error)
                        break;
                    continue;
                }
                const auto relative = std::filesystem::relative(it->path(), legacy_profiles, error);
                if (error)
                    break;
                const auto destination = keymapDir() / relative;
                const auto profile_result = copyFileIfMissing(it->path(), destination);
                if (!profile_result)
                    return std::unexpected(profile_result.error());
                if (*profile_result)
                    copied.push_back(destination);
            }
            if (error) {
                return std::unexpected(std::format("Unable to migrate legacy input profiles '{}': {}",
                                                    path_to_utf8(legacy_profiles), error.message()));
            }
        }
        return copied;
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

    std::filesystem::path UserPaths::preferencesFile() const { return config_dir_ / "preferences.json"; }
    std::filesystem::path UserPaths::layoutFile() const { return config_dir_ / "layout.json"; }
    std::filesystem::path UserPaths::keymapDir() const { return config_dir_ / "keymaps"; }
    std::filesystem::path UserPaths::presetDir() const { return data_dir_ / "presets"; }
    std::filesystem::path UserPaths::assetLibraryDir() const { return data_dir_ / "asset_library"; }
    std::filesystem::path UserPaths::backupDir() const { return data_dir_ / "backups"; }
    std::filesystem::path UserPaths::migrationDir() const { return data_dir_ / "migrations"; }

    std::vector<std::filesystem::path> UserPaths::legacyConfigDirs() const {
        std::vector<std::filesystem::path> result;
        const auto home = userHomeDirectory();
#ifdef _WIN32
        if (const auto appdata = environmentPath("APPDATA"))
            result.push_back(*appdata / "LichtFeldStudio");
        if (const auto local_appdata = environmentPath("LOCALAPPDATA"))
            result.push_back(*local_appdata / "LichtFeldStudio");
        if (home)
            result.push_back(*home / ".lichtfeld");
#else
        if (home) {
            if (const auto config_home = environmentPath("XDG_CONFIG_HOME"))
                result.push_back(*config_home / "LichtFeldStudio");
            result.push_back(*home / ".config" / "LichtFeldStudio");
            result.push_back(*home / ".lichtfeld");
        }
#endif
        return result;
    }

} // namespace lfs::core
