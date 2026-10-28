/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/legacy_settings_migration.hpp"
#include "core/native_fs.hpp"

#include "core/environment.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/persisted_settings.hpp"
#include "core/user_paths.hpp"

#include <atomic>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace lfs::core {
    namespace {
        using json = nlohmann::json;
        std::atomic<bool> g_migration_notice_pending{false};

        [[nodiscard]] std::optional<std::filesystem::path> environmentPath(const char* const name) {
            if (const auto value = environment::value(name))
                return utf8_to_path(std::string(*value));
            return std::nullopt;
        }

        [[nodiscard]] std::optional<std::filesystem::path> homeDirectory() {
#ifdef _WIN32
            if (const auto path = environmentPath("USERPROFILE"))
                return path;
            return environmentPath("HOME");
#else
            return environmentPath("HOME");
#endif
        }

        [[nodiscard]] std::optional<std::filesystem::path> legacyConfigDirectory() {
#ifdef _WIN32
            if (const auto path = environmentPath("APPDATA"))
                return *path / "LichtFeldStudio";
#else
            if (const auto path = environmentPath("XDG_CONFIG_HOME"))
                return *path / "LichtFeldStudio";
            if (const auto home = homeDirectory())
                return *home / ".config" / "LichtFeldStudio";
#endif
            return std::nullopt;
        }

        [[nodiscard]] lfs::Status migrationFailure(
            std::string message, const std::filesystem::path& path) {
            lfs::SmallFields fields;
            fields.add("path", path_to_utf8(path));
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::PermissionDenied,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = "Legacy user settings could not be migrated.",
                .detail = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
            }));
        }

        [[nodiscard]] lfs::Status copyIfMissing(
            const std::filesystem::path& source,
            const std::filesystem::path& destination,
            bool& conflict) {
            std::error_code error;
            if (!lfs::core::native_fs::is_regular_file(source, error))
                return {};
            if (lfs::core::native_fs::exists(destination, error)) {
                conflict = true;
                return {};
            }
            std::ifstream input(lfs::core::native_fs::native_path(source), std::ios::binary);
            if (!input)
                return migrationFailure(
                    std::format("Unable to read legacy settings file '{}'", path_to_utf8(source)), source);
            const std::string contents((std::istreambuf_iterator<char>(input)), {});
            return writeTextFileAtomically(destination, contents);
        }

        [[nodiscard]] std::vector<std::filesystem::path> legacyFiles(
            const std::filesystem::path& legacy) {
            std::vector<std::filesystem::path> files;
            for (const char* name : {"layout.json", "ui_preferences.json", "theme_preference",
                                     "language_preference", "ui_scale"}) {
                const auto path = legacy / name;
                std::error_code error;
                if (lfs::core::native_fs::is_regular_file(path, error))
                    files.push_back(path);
            }
            const auto keymaps = legacy / "input_profiles";
            std::error_code error;
            if (lfs::core::native_fs::is_directory(keymaps, error))
                for (std::filesystem::directory_iterator entry(lfs::core::native_fs::traversal_path(keymaps), error), end;
                     !error && entry != end; entry.increment(error)) {
                    if (entry->is_regular_file(error) && entry->path().extension() == ".json")
                        files.push_back(lfs::core::native_fs::entry_path(keymaps, entry->path()));
                }
            if (error)
                LOG_WARN("Unable to inspect older release settings: {}", error.message());
            return files;
        }

        [[nodiscard]] bool changedAfterMigration(
            const std::vector<std::filesystem::path>& files,
            const std::filesystem::path& marker) {
            std::error_code error;
            const auto migrated_at = lfs::core::native_fs::last_write_time(marker, error);
            if (error)
                return false;
            for (const auto& file : files) {
                const auto changed_at = lfs::core::native_fs::last_write_time(file, error);
                if (!error && changed_at > migrated_at)
                    return true;
                error.clear();
            }
            return false;
        }
    } // namespace

    lfs::Status migrateLegacySettings(const UserPaths& paths) {
        const auto home = homeDirectory();
        if (!home || paths.configDir() != *home / ".lichtfeld" / "config")
            return {};

        const auto marker = paths.dataDir() / "migrations" / "legacy-user-settings-v1.done";
        std::error_code error;
        std::string marker_text;
        if (lfs::core::native_fs::exists(marker, error) && !error) {
            std::ifstream input(lfs::core::native_fs::native_path(marker));
            marker_text.assign(std::istreambuf_iterator<char>(input), {});
            if (marker_text == "legacy settings migration conflict announced\n")
                return {};
            if (marker_text == "legacy settings migration conflict pending\n") {
                g_migration_notice_pending.store(true, std::memory_order_release);
                return {};
            }
        }
        if (const auto ensured = paths.ensureDirectories(); !ensured)
            return ensured;

        const auto legacy = legacyConfigDirectory();
        if (!legacy || *legacy == paths.configDir() || !lfs::core::native_fs::is_directory(*legacy, error))
            return marker_text.empty() ? writeTextFileAtomically(marker, "no legacy settings found\n") : lfs::Status{};

        const auto files = legacyFiles(*legacy);
        if (files.empty())
            return marker_text.empty() ? writeTextFileAtomically(marker, "no legacy settings found\n") : lfs::Status{};
        if (marker_text == "legacy settings migration complete\n") {
            if (changedAfterMigration(files, marker)) {
                LOG_WARN("Older release settings changed after import; current settings were retained");
                g_migration_notice_pending.store(true, std::memory_order_release);
                return writeTextFileAtomically(marker, "legacy settings migration conflict pending\n");
            }
            return {};
        }

        bool conflict = false;

        if (const auto copied = copyIfMissing(*legacy / "layout.json", paths.layoutFile(), conflict); !copied)
            return copied;
        if (const auto copied = copyIfMissing(*legacy / "ui_preferences.json", paths.uiPreferencesFile(), conflict); !copied)
            return copied;

        const auto old_keymaps = *legacy / "input_profiles";
        if (lfs::core::native_fs::is_directory(old_keymaps, error) && !error) {
            for (std::filesystem::directory_iterator entry(lfs::core::native_fs::traversal_path(old_keymaps), error), end;
                 !error && entry != end; entry.increment(error)) {
                if (!entry->is_regular_file(error) || entry->path().extension() != ".json")
                    continue;
                if (const auto copied = copyIfMissing(
                        lfs::core::native_fs::entry_path(old_keymaps, entry->path()), paths.keymapDir() / entry->path().filename(), conflict);
                    !copied)
                    return copied;
            }
            if (error)
                return migrationFailure(error.message(), old_keymaps);
        }

        const bool current_preferences_exist = lfs::core::native_fs::exists(paths.preferencesFile(), error);
        if (error)
            return migrationFailure(error.message(), paths.preferencesFile());
        json preferences = json::object();
        bool preferences_readable = true;
        if (current_preferences_exist) {
            try {
                preferences = persisted_settings::readObject(paths.preferencesFile());
            } catch (const std::exception& exception) {
                LOG_WARN("Unable to import older preference fields into '{}': {}",
                         path_to_utf8(paths.preferencesFile()), exception.what());
                preferences_readable = false;
                conflict = true;
            }
        }
        if (preferences_readable) {
            bool imported = false;
            const auto import_field = [&](const char* key, const json& value) {
                if (!preferences.contains(key)) {
                    preferences[key] = value;
                    imported = true;
                } else if (preferences[key] != value) {
                    conflict = true;
                    LOG_WARN("Older preference '{}' differs from current settings; current value was retained", key);
                }
            };
            for (const auto [filename, key] : {
                     std::pair{"theme_preference", "theme"},
                     std::pair{"language_preference", "language"}}) {
                std::ifstream input(lfs::core::native_fs::native_path(*legacy / filename));
                std::string value;
                if (input >> value)
                    import_field(key, value);
            }
            std::ifstream scale_input(lfs::core::native_fs::native_path(*legacy / "ui_scale"));
            float scale = 0.0f;
            if (scale_input >> scale && std::isfinite(scale))
                import_field("ui_scale", scale <= 0.0f ? json("auto") : json(scale));
            if (imported) {
                if (!current_preferences_exist)
                    preferences["schema_version"] = 1;
                if (const auto written = paths.writePreferencesAtomically(preferences.dump(2) + '\n'); !written)
                    return written;
            }
        }

        if (conflict) {
            LOG_WARN("Older release settings were found after current settings; current files were retained");
            g_migration_notice_pending.store(true, std::memory_order_release);
        }
        return writeTextFileAtomically(marker, conflict
                                                   ? "legacy settings migration conflict pending\n"
                                                   : "legacy settings migration complete\n");
    }

    bool takeLegacySettingsMigrationNotice() noexcept {
        if (!g_migration_notice_pending.exchange(false, std::memory_order_acq_rel))
            return false;
        try {
            if (const auto paths = UserPaths::resolve()) {
                const auto marker = paths->dataDir() / "migrations" / "legacy-user-settings-v1.done";
                if (const auto acknowledged = writeTextFileAtomically(
                        marker, "legacy settings migration conflict announced\n");
                    !acknowledged)
                    LOG_WARN("Unable to acknowledge older release settings notice: {}",
                             lfs::format_for_developer(acknowledged.error()));
            }
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): a failed acknowledgement repeats the notice on next launch.
            LOG_WARN("Unable to acknowledge older release settings notice: {}", error.what());
        }
        return true;
    }
} // namespace lfs::core
