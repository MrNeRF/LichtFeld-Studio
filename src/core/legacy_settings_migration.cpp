/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/legacy_settings_migration.hpp"

#include "core/environment.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <optional>
#include <utility>

namespace lfs::core {
    namespace {
        using json = nlohmann::json;

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
            const std::filesystem::path& destination) {
            std::error_code error;
            if (std::filesystem::exists(destination, error) ||
                !std::filesystem::is_regular_file(source, error))
                return {};
            std::ifstream input(source, std::ios::binary);
            if (!input)
                return migrationFailure(
                    std::format("Unable to read legacy settings file '{}'", path_to_utf8(source)), source);
            const std::string contents((std::istreambuf_iterator<char>(input)), {});
            return writeTextFileAtomically(destination, contents);
        }
    } // namespace

    lfs::Status migrateLegacySettings(const UserPaths& paths) {
        const auto home = homeDirectory();
        if (!home || paths.configDir() != *home / ".lichtfeld" / "config")
            return {};

        const auto marker = paths.dataDir() / "migrations" / "legacy-user-settings-v1.done";
        std::error_code error;
        if (std::filesystem::exists(marker, error) && !error)
            return {};
        if (const auto ensured = paths.ensureDirectories(); !ensured)
            return ensured;

        const auto legacy = legacyConfigDirectory();
        if (!legacy || *legacy == paths.configDir() || !std::filesystem::is_directory(*legacy, error))
            return writeTextFileAtomically(marker, "no legacy settings found\n");

        if (const auto copied = copyIfMissing(*legacy / "layout.json", paths.layoutFile()); !copied)
            return copied;
        if (const auto copied = copyIfMissing(*legacy / "ui_preferences.json", paths.uiPreferencesFile()); !copied)
            return copied;

        const auto old_keymaps = *legacy / "input_profiles";
        if (std::filesystem::is_directory(old_keymaps, error) && !error) {
            for (const auto& entry : std::filesystem::directory_iterator(old_keymaps, error)) {
                if (error)
                    return migrationFailure(error.message(), old_keymaps);
                if (!entry.is_regular_file() || entry.path().extension() != ".json")
                    continue;
                if (const auto copied = copyIfMissing(
                        entry.path(), paths.keymapDir() / entry.path().filename());
                    !copied)
                    return copied;
            }
        }

        if (!std::filesystem::exists(paths.preferencesFile(), error)) {
            json preferences = json::object();
            const auto import_text = [&](const char* const filename, const char* const key) {
                std::ifstream input(*legacy / filename);
                std::string value;
                if (input >> value)
                    preferences[key] = value;
            };
            import_text("theme_preference", "theme");
            import_text("language_preference", "language");
            std::ifstream scale_input(*legacy / "ui_scale");
            float scale = 0.0f;
            if (scale_input >> scale)
                preferences["ui_scale"] = scale <= 0.0f ? json("auto") : json(scale);
            if (!preferences.empty()) {
                preferences["schema_version"] = 1;
                if (const auto written = paths.writePreferencesAtomically(preferences.dump(2) + '\n'); !written)
                    return written;
            }
        }

        return writeTextFileAtomically(marker, "legacy settings migration complete\n");
    }
} // namespace lfs::core
