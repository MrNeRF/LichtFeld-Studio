/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/user_paths.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>

#include <nlohmann/json.hpp>

namespace {

    std::filesystem::path uniqueTestRoot(const char* const name) {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               std::format("{}_{}", name, suffix);
    }

} // namespace

TEST(UserPathsTest, ExplicitRootUsesPortableCompatibleLayout) {
    const auto root = uniqueTestRoot("lfs_user_paths_explicit");
    const auto paths = lfs::core::UserPaths::resolve({.explicit_root = root});

    ASSERT_TRUE(paths.has_value()) << paths.error();
    EXPECT_TRUE(paths->usesUnifiedRoot());
    EXPECT_EQ(paths->configDir(), root / "config");
    EXPECT_EQ(paths->dataDir(), root / "data");
    EXPECT_EQ(paths->cacheDir(), root / "cache");
    EXPECT_EQ(paths->logDir(), root / "logs");
    EXPECT_EQ(paths->pluginDir(), root / "plugins");
    EXPECT_EQ(paths->venvDir(), root / "venv");
    EXPECT_EQ(paths->preferencesFile(), root / "config" / "preferences.json");
    EXPECT_EQ(paths->layoutFile(), root / "config" / "layout.json");
    EXPECT_EQ(paths->presetDir(), root / "data" / "presets");
}

TEST(UserPathsTest, PortableModeUsesHiddenDirectoryNextToExecutable) {
    const auto executable_dir = uniqueTestRoot("lfs_user_paths_portable");
    const auto paths = lfs::core::UserPaths::resolve({
        .executable_dir = executable_dir,
        .portable = true,
    });

    ASSERT_TRUE(paths.has_value()) << paths.error();
    EXPECT_TRUE(paths->usesUnifiedRoot());
    EXPECT_EQ(paths->configDir(), executable_dir / ".lichtfeld" / "config");
}

TEST(UserPathsTest, PortableModeRequiresExecutableDirectory) {
    const auto paths = lfs::core::UserPaths::resolve({.portable = true});

    ASSERT_FALSE(paths.has_value());
    EXPECT_EQ(paths.error(), "Portable user storage requires an executable directory");
}

TEST(UserPathsTest, EnsureDirectoriesCreatesOnlyResolvedTree) {
    const auto root = uniqueTestRoot("lfs_user_paths_directories");
    const auto paths = lfs::core::UserPaths::resolve({.explicit_root = root});
    ASSERT_TRUE(paths.has_value()) << paths.error();

    const auto result = paths->ensureDirectories();
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(std::filesystem::is_directory(paths->configDir()));
    EXPECT_TRUE(std::filesystem::is_directory(paths->keymapDir()));
    EXPECT_TRUE(std::filesystem::is_directory(paths->presetDir()));
    EXPECT_TRUE(std::filesystem::is_directory(paths->migrationDir()));

    std::error_code error;
    std::filesystem::remove_all(root, error);
    EXPECT_FALSE(error) << error.message();
}

TEST(UserPathsTest, ForcedMigrationValidatesAndRecordsRecognizedLegacyGuiSettings) {
    const auto root = uniqueTestRoot("lfs_user_paths_migration");
    const auto legacy_root = uniqueTestRoot("lfs_legacy_gui_settings");
    std::filesystem::create_directories(legacy_root / "input_profiles");
    {
        std::ofstream file(legacy_root / "layout.json");
        file << R"({"right_panel_width": 420})";
    }
    {
        std::ofstream file(legacy_root / "theme_preference");
        file << "gruvbox\n";
    }
    {
        std::ofstream file(legacy_root / "ui_scale");
        file << "1.5\n";
    }
    {
        std::ofstream file(legacy_root / "input_profiles" / "Default.json");
        file << R"({"name":"Default","bindings":[]})";
    }

    const auto paths = lfs::core::UserPaths::resolve({
        .explicit_root = root,
        .legacy_config_dirs = {legacy_root},
    });
    ASSERT_TRUE(paths.has_value()) << paths.error();

    const auto migrated = paths->migrateLegacyGuiSettings(true);
    ASSERT_TRUE(migrated.has_value()) << migrated.error();
    EXPECT_TRUE(std::filesystem::exists(paths->layoutFile()));
    EXPECT_TRUE(std::filesystem::exists(paths->keymapDir() / "Default.json"));
    EXPECT_TRUE(std::filesystem::exists(paths->migrationManifestFile()));

    std::ifstream preferences_file(paths->preferencesFile());
    const auto preferences = nlohmann::json::parse(preferences_file);
    EXPECT_EQ(preferences.value("theme", ""), "gruvbox");
    EXPECT_FLOAT_EQ(preferences.value("ui_scale", 0.0f), 1.5f);

    std::error_code error;
    std::filesystem::remove_all(root, error);
    EXPECT_FALSE(error) << error.message();
    std::filesystem::remove_all(legacy_root, error);
    EXPECT_FALSE(error) << error.message();
}

TEST(UserPathsTest, InvalidLegacyCandidateDoesNotBlockLaterValidCandidate) {
    const auto root = uniqueTestRoot("lfs_user_paths_invalid_legacy");
    const auto invalid_root = uniqueTestRoot("lfs_legacy_invalid");
    const auto valid_root = uniqueTestRoot("lfs_legacy_valid");
    std::filesystem::create_directories(invalid_root);
    std::filesystem::create_directories(valid_root);
    {
        std::ofstream file(invalid_root / "layout.json");
        file << "not json";
    }
    {
        std::ofstream file(valid_root / "layout.json");
        file << R"({"left_dock_width": 240})";
    }

    const auto paths = lfs::core::UserPaths::resolve({
        .explicit_root = root,
        .legacy_config_dirs = {invalid_root, valid_root},
    });
    ASSERT_TRUE(paths.has_value()) << paths.error();
    const auto migrated = paths->migrateLegacyGuiSettings(true);
    ASSERT_TRUE(migrated.has_value()) << migrated.error();
    EXPECT_TRUE(std::filesystem::exists(paths->layoutFile()));

    std::ifstream layout_file(paths->layoutFile());
    const auto layout = nlohmann::json::parse(layout_file);
    EXPECT_EQ(layout.value("left_dock_width", 0), 240);

    std::error_code error;
    std::filesystem::remove_all(root, error);
    EXPECT_FALSE(error) << error.message();
    std::filesystem::remove_all(invalid_root, error);
    EXPECT_FALSE(error) << error.message();
    std::filesystem::remove_all(valid_root, error);
    EXPECT_FALSE(error) << error.message();
}

TEST(UserPathsTest, ArchiveOnlyRemovesLegacyFilesAlreadyMigratedToValidDestinations) {
    const auto root = uniqueTestRoot("lfs_user_paths_archive");
    const auto legacy_root = uniqueTestRoot("lfs_legacy_archive");
    std::filesystem::create_directories(legacy_root);
    {
        std::ofstream file(legacy_root / "layout.json");
        file << R"({"bottom_dock_height": 180})";
    }
    {
        std::ofstream file(legacy_root / "theme_preference");
        file << "dark\n";
    }
    {
        std::ofstream file(legacy_root / "ui_scale");
        file << "not-a-scale\n";
    }

    const auto paths = lfs::core::UserPaths::resolve({
        .explicit_root = root,
        .legacy_config_dirs = {legacy_root},
    });
    ASSERT_TRUE(paths.has_value()) << paths.error();
    ASSERT_TRUE(paths->migrateLegacyGuiSettings(true).has_value());

    const auto archived = paths->archiveLegacyGuiSettings();
    ASSERT_TRUE(archived.has_value()) << archived.error();
    EXPECT_EQ(archived->size(), 2);
    EXPECT_FALSE(std::filesystem::exists(legacy_root / "layout.json"));
    EXPECT_FALSE(std::filesystem::exists(legacy_root / "theme_preference"));
    EXPECT_TRUE(std::filesystem::exists(legacy_root / "ui_scale"));
    EXPECT_TRUE(std::filesystem::exists(paths->layoutFile()));
    EXPECT_TRUE(std::filesystem::exists(paths->preferencesFile()));
    for (const auto& archive_file : *archived)
        EXPECT_TRUE(std::filesystem::exists(archive_file));

    std::error_code error;
    std::filesystem::remove_all(root, error);
    EXPECT_FALSE(error) << error.message();
    std::filesystem::remove_all(legacy_root, error);
    EXPECT_FALSE(error) << error.message();
}

TEST(UserPathsTest, AutomaticMigrationManifestPreventsLegacyRescanAfterReset) {
    const auto root = uniqueTestRoot("lfs_user_paths_manifest");
    const auto legacy_root = uniqueTestRoot("lfs_legacy_manifest");
    std::filesystem::create_directories(legacy_root);
    {
        std::ofstream file(legacy_root / "layout.json");
        file << R"({"right_panel_width": 420})";
    }

    const auto paths = lfs::core::UserPaths::resolve({
        .explicit_root = root,
        .legacy_config_dirs = {legacy_root},
        .automatic_legacy_migration = true,
    });
    ASSERT_TRUE(paths.has_value()) << paths.error();
    ASSERT_TRUE(paths->migrateLegacyGuiSettings().has_value());
    ASSERT_TRUE(paths->resetLayout().has_value());
    EXPECT_FALSE(std::filesystem::exists(paths->layoutFile()));

    const auto repeat = paths->migrateLegacyGuiSettings();
    ASSERT_TRUE(repeat.has_value()) << repeat.error();
    EXPECT_TRUE(repeat->empty());
    EXPECT_FALSE(std::filesystem::exists(paths->layoutFile()));

    std::error_code error;
    std::filesystem::remove_all(root, error);
    EXPECT_FALSE(error) << error.message();
    std::filesystem::remove_all(legacy_root, error);
    EXPECT_FALSE(error) << error.message();
}
