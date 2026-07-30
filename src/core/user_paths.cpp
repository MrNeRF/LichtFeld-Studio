/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/user_paths.hpp"

#include "core/environment.hpp"
#include "path_utils.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

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

        using json = nlohmann::json;

        [[nodiscard]] std::string trim(std::string value) {
            const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char c) {
                return std::isspace(c) != 0;
            });
            const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char c) {
                                  return std::isspace(c) != 0;
                              }).base();
            return first < last ? std::string(first, last) : std::string{};
        }

        [[nodiscard]] std::expected<json, std::string> readJsonObject(const std::filesystem::path& path) {
            std::ifstream file(path);
            if (!file)
                return std::unexpected(std::format("Unable to open JSON file '{}'", path_to_utf8(path)));
            try {
                const auto value = json::parse(file);
                if (!value.is_object())
                    return std::unexpected(std::format("JSON file '{}' must contain an object", path_to_utf8(path)));
                return value;
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Invalid JSON file '{}': {}", path_to_utf8(path), e.what()));
            }
        }

        [[nodiscard]] std::expected<void, std::string> writeJsonAtomically(
            const std::filesystem::path& destination, const json& value) {
            std::error_code error;
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error)
                return std::unexpected(std::format("Unable to create directory '{}': {}",
                                                   path_to_utf8(destination.parent_path()), error.message()));

            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
            const auto temporary = destination.parent_path() /
                                   std::format("{}.tmp-{}", path_to_utf8(destination.filename()), millis);
            {
                std::ofstream file(temporary, std::ios::trunc);
                if (!file)
                    return std::unexpected(std::format("Unable to write temporary file '{}'", path_to_utf8(temporary)));
                file << value.dump(2) << '\n';
                file.close();
                if (!file)
                    return std::unexpected(std::format("Unable to finish temporary file '{}'", path_to_utf8(temporary)));
            }

#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                const auto message = std::system_category().message(static_cast<int>(GetLastError()));
                std::filesystem::remove(temporary, error);
                return std::unexpected(std::format("Unable to replace '{}' atomically: {}",
                                                   path_to_utf8(destination), message));
            }
#else
            std::filesystem::rename(temporary, destination, error);
            if (error) {
                std::filesystem::remove(temporary, error);
                return std::unexpected(std::format("Unable to replace '{}' atomically: {}",
                                                   path_to_utf8(destination), error.message()));
            }
#endif
            return {};
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
            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
            const auto temporary = destination.parent_path() /
                                   std::format("{}.migration-{}", path_to_utf8(destination.filename()), millis);
            if (!std::filesystem::copy_file(source, temporary, std::filesystem::copy_options::none, error) || error)
                return std::unexpected(std::format("Unable to copy legacy settings file '{}' to '{}': {}",
                                                   path_to_utf8(source), path_to_utf8(temporary), error.message()));
            if (std::filesystem::file_size(source, error) != std::filesystem::file_size(temporary, error) || error) {
                std::filesystem::remove(temporary, error);
                return std::unexpected(std::format("Unable to verify copied legacy settings file '{}'", path_to_utf8(source)));
            }
#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH)) {
                const auto message = std::system_category().message(static_cast<int>(GetLastError()));
                std::filesystem::remove(temporary, error);
                return std::unexpected(std::format("Unable to finalize migrated settings file '{}': {}",
                                                   path_to_utf8(destination), message));
            }
#else
            std::filesystem::create_hard_link(temporary, destination, error);
            if (error) {
                const auto link_error = error;
                error.clear();
                const bool destination_exists = std::filesystem::exists(destination, error);
                std::filesystem::remove(temporary, error);
                if (!error && destination_exists)
                    return false;
                return std::unexpected(std::format("Unable to finalize migrated settings file '{}': {}",
                                                   path_to_utf8(destination), link_error.message()));
            }
            std::filesystem::remove(temporary, error);
            if (error)
                return std::unexpected(std::format("Unable to remove temporary migration file '{}': {}",
                                                   path_to_utf8(temporary), error.message()));
#endif
            return true;
        }

        [[nodiscard]] bool isValidThemeId(const std::string& value) {
            return !value.empty() && value.size() <= 64 &&
                   std::all_of(value.begin(), value.end(), [](const unsigned char c) {
                       return std::isalnum(c) != 0 || c == '_' || c == '-';
                   });
        }

        [[nodiscard]] bool isValidUiScale(const std::string& value) {
            try {
                size_t consumed = 0;
                const float scale = std::stof(value, &consumed);
                return consumed == value.size() && std::isfinite(scale) && scale >= 0.5f && scale <= 4.0f;
            } catch (...) {
                return false;
            }
        }

        [[nodiscard]] std::optional<std::string> readLegacyTextValue(const std::filesystem::path& path) {
            std::ifstream file(path);
            std::string value;
            if (!file || !std::getline(file, value))
                return std::nullopt;
            value = trim(std::move(value));
            return value.empty() ? std::nullopt : std::optional<std::string>(std::move(value));
        }

        [[nodiscard]] const char* legacyStatusName(const LegacyGuiSettingStatus status) {
            switch (status) {
            case LegacyGuiSettingStatus::Available:
                return "available";
            case LegacyGuiSettingStatus::Migrated:
                return "migrated";
            case LegacyGuiSettingStatus::AlreadyMigrated:
                return "already_migrated";
            case LegacyGuiSettingStatus::Invalid:
                return "invalid";
            case LegacyGuiSettingStatus::DestinationInvalid:
                return "destination_invalid";
            }
            return "unknown";
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
            return writeJsonAtomically(destination, {
                                                        {"schema_version", 1},
                                                        {"theme", "dark"},
                                                        {"ui_scale", "auto"},
                                                    });
        }

    } // namespace

    UserPaths::UserPaths(std::filesystem::path config_dir,
                         std::filesystem::path data_dir,
                         std::filesystem::path cache_dir,
                         std::filesystem::path log_dir,
                         std::filesystem::path plugin_dir,
                         std::filesystem::path venv_dir,
                         std::vector<std::filesystem::path> legacy_config_dirs,
                         const bool unified_root,
                         const bool automatic_legacy_migration)
        : config_dir_(std::move(config_dir)),
          data_dir_(std::move(data_dir)),
          cache_dir_(std::move(cache_dir)),
          log_dir_(std::move(log_dir)),
          plugin_dir_(std::move(plugin_dir)),
          venv_dir_(std::move(venv_dir)),
          legacy_config_dirs_(std::move(legacy_config_dirs)),
          unified_root_(unified_root),
          automatic_legacy_migration_(automatic_legacy_migration) {}

    UserPaths UserPaths::fromUnifiedRoot(const std::filesystem::path& root,
                                         std::vector<std::filesystem::path> legacy_config_dirs,
                                         const bool automatic_legacy_migration) {
        return UserPaths(
            root / "config",
            root / "data",
            root / "cache",
            root / "logs",
            root / "plugins",
            root / "venv",
            std::move(legacy_config_dirs),
            true,
            automatic_legacy_migration);
    }

    std::expected<UserPaths, std::string> UserPaths::resolve(const UserPathOptions& options) {
        if (options.explicit_root && !options.explicit_root->empty())
            return fromUnifiedRoot(*options.explicit_root, options.legacy_config_dirs,
                                   options.automatic_legacy_migration.value_or(false));

        if (const auto root = environmentPath("LFS_HOME"))
            return fromUnifiedRoot(*root, options.legacy_config_dirs,
                                   options.automatic_legacy_migration.value_or(false));

        if (options.portable) {
            if (!options.executable_dir || options.executable_dir->empty())
                return std::unexpected("Portable user storage requires an executable directory");
            return fromUnifiedRoot(*options.executable_dir / ".lichtfeld", options.legacy_config_dirs,
                                   options.automatic_legacy_migration.value_or(false));
        }

        const auto home = userHomeDirectory();
        if (!home)
            return std::unexpected("Unable to resolve the current user's home directory");

#ifdef _WIN32
        std::vector<std::filesystem::path> legacy_config_dirs = options.legacy_config_dirs;
        if (legacy_config_dirs.empty()) {
            if (const auto appdata = environmentPath("APPDATA"))
                legacy_config_dirs.push_back(*appdata / "LichtFeldStudio");
            if (const auto local_appdata = environmentPath("LOCALAPPDATA"))
                legacy_config_dirs.push_back(*local_appdata / "LichtFeldStudio");
            legacy_config_dirs.push_back(*home / ".lichtfeld");
        }
        return fromUnifiedRoot(*home / ".lichtfeld", std::move(legacy_config_dirs),
                               options.automatic_legacy_migration.value_or(true));
#else
        const auto config_home = xdgOrHome("XDG_CONFIG_HOME", *home, ".config");
        const auto data_home = xdgOrHome("XDG_DATA_HOME", *home, ".local/share");
        const auto cache_home = xdgOrHome("XDG_CACHE_HOME", *home, ".cache");
        const auto state_home = xdgOrHome("XDG_STATE_HOME", *home, ".local/state");
        const auto plugin_home = *home / ".lichtfeld";
        std::vector<std::filesystem::path> legacy_config_dirs = options.legacy_config_dirs;
        if (legacy_config_dirs.empty()) {
            if (const auto legacy_config_home = environmentPath("XDG_CONFIG_HOME"))
                legacy_config_dirs.push_back(*legacy_config_home / "LichtFeldStudio");
            legacy_config_dirs.push_back(*home / ".config" / "LichtFeldStudio");
            legacy_config_dirs.push_back(*home / ".lichtfeld");
        }
        return UserPaths(
            config_home / "lichtfeld-studio",
            data_home / "lichtfeld-studio",
            cache_home / "lichtfeld-studio",
            state_home / "lichtfeld-studio",
            plugin_home / "plugins",
            plugin_home / "venv",
            std::move(legacy_config_dirs),
            false,
            options.automatic_legacy_migration.value_or(true));
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

    std::expected<std::vector<LegacyGuiSetting>, std::string>
    UserPaths::inspectLegacyGuiSettings() const {
        std::vector<LegacyGuiSetting> entries;

        const auto add_json_entry = [&entries](std::string kind,
                                               const std::filesystem::path& source,
                                               const std::filesystem::path& destination) {
            std::error_code error;
            if (!std::filesystem::is_regular_file(source, error))
                return;

            LegacyGuiSetting entry{
                .kind = std::move(kind),
                .source = source,
                .destination = destination,
            };
            if (const auto source_json = readJsonObject(source); !source_json) {
                entry.status = LegacyGuiSettingStatus::Invalid;
                entry.detail = source_json.error();
            } else if (std::filesystem::exists(destination, error)) {
                if (const auto destination_json = readJsonObject(destination); !destination_json) {
                    entry.status = LegacyGuiSettingStatus::DestinationInvalid;
                    entry.detail = destination_json.error();
                } else {
                    entry.status = LegacyGuiSettingStatus::AlreadyMigrated;
                }
            }
            entries.push_back(std::move(entry));
        };

        const auto add_preference_entry = [&entries, this](std::string kind,
                                                           const std::filesystem::path& source,
                                                           const char* const key,
                                                           const auto& validator) {
            std::error_code error;
            if (!std::filesystem::is_regular_file(source, error))
                return;

            LegacyGuiSetting entry{
                .kind = std::move(kind),
                .source = source,
                .destination = preferencesFile(),
            };
            const auto value = readLegacyTextValue(source);
            if (!value || !validator(*value)) {
                entry.status = LegacyGuiSettingStatus::Invalid;
                entry.detail = "Legacy value is missing or invalid";
            } else if (std::filesystem::exists(entry.destination, error)) {
                if (const auto preferences = readJsonObject(entry.destination); !preferences) {
                    entry.status = LegacyGuiSettingStatus::DestinationInvalid;
                    entry.detail = preferences.error();
                } else if (preferences->contains(key)) {
                    const auto& current_value = preferences->at(key);
                    const bool valid_destination =
                        (current_value.is_string() &&
                         ((std::string_view(key) == "ui_scale" && current_value.get<std::string>() == "auto") ||
                          validator(current_value.get<std::string>()))) ||
                        (std::string_view(key) == "ui_scale" && current_value.is_number() &&
                         validator(std::to_string(current_value.get<float>())));
                    if (valid_destination) {
                        entry.status = LegacyGuiSettingStatus::AlreadyMigrated;
                    } else {
                        entry.status = LegacyGuiSettingStatus::DestinationInvalid;
                        entry.detail = "Current preferences value is invalid";
                    }
                }
            }
            entries.push_back(std::move(entry));
        };

        for (const auto& legacy_dir : legacyConfigDirs()) {
            add_json_entry("layout", legacy_dir / "layout.json", layoutFile());
            add_preference_entry("theme", legacy_dir / "theme_preference", "theme", isValidThemeId);
            add_preference_entry("ui_scale", legacy_dir / "ui_scale", "ui_scale", isValidUiScale);

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
                if (it->path().extension() != ".json")
                    continue;
                const auto relative = std::filesystem::relative(it->path(), legacy_profiles, error);
                if (error)
                    break;
                add_json_entry("keymap", it->path(), keymapDir() / relative);
            }
            if (error) {
                return std::unexpected(std::format("Unable to inspect legacy input profiles '{}': {}",
                                                   path_to_utf8(legacy_profiles), error.message()));
            }
        }
        return entries;
    }

    std::expected<std::vector<LegacyGuiSetting>, std::string>
    UserPaths::migrateLegacyGuiSettings(const bool force) const {
        std::error_code error;
        if (!force && (!allowsAutomaticLegacyMigration() || std::filesystem::exists(migrationManifestFile(), error)))
            return std::vector<LegacyGuiSetting>{};
        if (error)
            return std::unexpected(std::format("Unable to inspect migration manifest '{}': {}",
                                               path_to_utf8(migrationManifestFile()), error.message()));

        auto inspected = inspectLegacyGuiSettings();
        if (!inspected)
            return std::unexpected(inspected.error());

        json preferences = json::object();
        bool preferences_valid = true;
        if (std::filesystem::exists(preferencesFile(), error)) {
            if (const auto loaded = readJsonObject(preferencesFile()); loaded) {
                preferences = *loaded;
            } else {
                preferences_valid = false;
            }
        }
        if (error)
            return std::unexpected(std::format("Unable to inspect preferences '{}': {}",
                                               path_to_utf8(preferencesFile()), error.message()));

        bool preferences_changed = false;
        for (auto& entry : *inspected) {
            if (entry.status != LegacyGuiSettingStatus::Available)
                continue;
            if ((entry.kind == "theme" || entry.kind == "ui_scale") && !preferences_valid) {
                entry.status = LegacyGuiSettingStatus::DestinationInvalid;
                entry.detail = "Current preferences.json is invalid and was left untouched";
                continue;
            }
            const auto legacy_value = readLegacyTextValue(entry.source);
            if ((entry.kind == "theme" || entry.kind == "ui_scale") && !legacy_value) {
                entry.status = LegacyGuiSettingStatus::Invalid;
                entry.detail = "Legacy value disappeared or became invalid during migration";
                continue;
            }
            if (entry.kind == "theme") {
                preferences["theme"] = *legacy_value;
                preferences_changed = true;
                entry.status = LegacyGuiSettingStatus::Migrated;
            } else if (entry.kind == "ui_scale") {
                preferences["ui_scale"] = std::stof(*legacy_value);
                preferences_changed = true;
                entry.status = LegacyGuiSettingStatus::Migrated;
            }
        }
        if (preferences_changed) {
            preferences["schema_version"] = 1;
            if (const auto result = writeJsonAtomically(preferencesFile(), preferences); !result)
                return std::unexpected(result.error());
        }

        for (auto& entry : *inspected) {
            if (entry.status != LegacyGuiSettingStatus::Available)
                continue;
            if (entry.kind != "layout" && entry.kind != "keymap")
                continue;
            const auto copied = copyFileIfMissing(entry.source, entry.destination);
            if (!copied)
                return std::unexpected(copied.error());
            entry.status = *copied ? LegacyGuiSettingStatus::Migrated
                                   : LegacyGuiSettingStatus::AlreadyMigrated;
        }

        json manifest;
        manifest["schema_version"] = 1;
        manifest["completed"] = true;
        manifest["entries"] = json::array();
        for (const auto& entry : *inspected) {
            manifest["entries"].push_back({
                {"kind", entry.kind},
                {"source", path_to_utf8(entry.source)},
                {"destination", path_to_utf8(entry.destination)},
                {"status", legacyStatusName(entry.status)},
                {"detail", entry.detail},
            });
        }
        if (const auto result = writeJsonAtomically(migrationManifestFile(), manifest); !result)
            return std::unexpected(result.error());
        return *inspected;
    }

    std::expected<std::vector<std::filesystem::path>, std::string>
    UserPaths::archiveLegacyGuiSettings() const {
        auto inspected = inspectLegacyGuiSettings();
        if (!inspected)
            return std::unexpected(inspected.error());

        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
        const auto archive_root = backupDir() / std::format("legacy-archive-{}", millis);
        std::vector<std::filesystem::path> archived;
        size_t index = 0;
        for (const auto& entry : *inspected) {
            std::error_code error;
            if (entry.status != LegacyGuiSettingStatus::Migrated &&
                entry.status != LegacyGuiSettingStatus::AlreadyMigrated)
                continue;
            if (!std::filesystem::is_regular_file(entry.source, error) || entry.source == entry.destination)
                continue;

            const auto destination = archive_root /
                                     std::format("{:03}-{}", ++index, path_to_utf8(entry.source.filename()));
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error)
                return std::unexpected(std::format("Unable to create legacy archive '{}': {}",
                                                   path_to_utf8(destination.parent_path()), error.message()));
            if (!std::filesystem::copy_file(entry.source, destination, std::filesystem::copy_options::none, error) || error)
                return std::unexpected(std::format("Unable to archive legacy file '{}'", path_to_utf8(entry.source)));
            if (std::filesystem::file_size(entry.source, error) != std::filesystem::file_size(destination, error) || error)
                return std::unexpected(std::format("Unable to verify legacy archive '{}'", path_to_utf8(destination)));
            if (!std::filesystem::remove(entry.source, error) || error)
                return std::unexpected(std::format("Archived legacy file but could not remove '{}'", path_to_utf8(entry.source)));
            archived.push_back(destination);
        }
        return archived;
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
    std::filesystem::path UserPaths::migrationManifestFile() const {
        return migrationDir() / "legacy-gui-settings-v1.json";
    }

    std::vector<std::filesystem::path> UserPaths::legacyConfigDirs() const {
        return legacy_config_dirs_;
    }

} // namespace lfs::core
