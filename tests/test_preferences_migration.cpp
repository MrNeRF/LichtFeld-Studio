/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/path_utils.hpp"
#include "core/user_paths.hpp"
#include "visualizer/preferences.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace {
    using json = nlohmann::json;

    class ScopedLfsHome {
    public:
        explicit ScopedLfsHome(const std::filesystem::path& path) {
            if (const char* previous = std::getenv("LFS_HOME"))
                previous_ = previous;
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", path.string().c_str());
#else
            (void)setenv("LFS_HOME", path.string().c_str(), 1);
#endif
        }

        ~ScopedLfsHome() {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", previous_ ? previous_->c_str() : "");
#else
            if (previous_)
                (void)setenv("LFS_HOME", previous_->c_str(), 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
        }

    private:
        std::optional<std::string> previous_;
    };

    std::filesystem::path makeHome(const char* name) {
        const auto home = std::filesystem::temp_directory_path() / name;
        std::error_code error;
        std::filesystem::remove_all(home, error);
        std::filesystem::create_directories(home);
        return home;
    }

    void writePreferences(const lfs::core::UserPaths& paths, const json& values) {
        std::ofstream output(paths.preferencesFile());
        ASSERT_TRUE(output);
        output << values.dump(2) << '\n';
    }

    json readPreferences(const lfs::core::UserPaths& paths) {
        std::ifstream input(paths.preferencesFile());
        return json::parse(input);
    }
} // namespace

TEST(PreferencesMigration, AssetDirectoryWinsOverWorkingDirectoryAndErasesLegacyKeys) {
    const auto home = makeHome("lfs_preferences_migration_asset_wins");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto asset = home / "assets";
    const auto working = home / "working";
    writePreferences(*paths, {
                                 {"asset_manager_directory", asset.string()},
                                 {"working_directory", working.string()},
                             });

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), asset);
    const auto migrated = readPreferences(*paths);
    EXPECT_EQ(migrated.at("project_location"), asset.string());
    EXPECT_EQ(migrated.at("legacy_working_directory"), working.string());
    EXPECT_FALSE(migrated.contains("asset_manager_directory"));
    EXPECT_FALSE(migrated.contains("working_directory"));
}

TEST(PreferencesMigration, ExistingProjectLocationPreservesLegacyWorkingDirectory) {
    const auto home = makeHome("lfs_preferences_migration_existing_project");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto project = home / "projects";
    const auto working = home / "working";
    writePreferences(*paths, {
                                 {"project_location", project.string()},
                                 {"working_directory", working.string()},
                             });

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), project);
    const auto migrated = readPreferences(*paths);
    EXPECT_EQ(migrated.at("legacy_working_directory"), working.string());
    EXPECT_FALSE(migrated.contains("working_directory"));
}

TEST(PreferencesMigration, WorkingDirectoryIsUsedWhenAssetDirectoryIsEmpty) {
    const auto home = makeHome("lfs_preferences_migration_working");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto working = home / "working";
    writePreferences(*paths, {
                                 {"asset_manager_directory", ""},
                                 {"working_directory", working.string()},
                             });
    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), working);
}

TEST(PreferencesMigration, WorkingDirectoryEqualToRootFallsBackToDefault) {
    const auto home = makeHome("lfs_preferences_migration_root");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"working_directory", paths->rootDir().string()}});

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
}

TEST(PreferencesMigration, EmptyWorkingDirectoryDoesNotWriteLegacyTombstone) {
    const auto home = makeHome("lfs_preferences_migration_empty_working");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"working_directory", ""}});

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
    EXPECT_FALSE(readPreferences(*paths).contains("legacy_working_directory"));
}

TEST(PreferencesMigration, ExistingLegacyWorkingDirectoryIsNotOverwritten) {
    const auto home = makeHome("lfs_preferences_migration_existing_tombstone");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto preserved = home / "preserved-working";
    const auto current = home / "current-working";
    writePreferences(*paths, {
                                 {"legacy_working_directory", preserved.string()},
                                 {"working_directory", current.string()},
                             });

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), current);
    EXPECT_EQ(readPreferences(*paths).at("legacy_working_directory"), preserved.string());
}

TEST(PreferencesMigration, ExistingAssetsCatalogIsPinnedWhenLegacyLocationsUnset) {
    const auto home = makeHome("lfs_preferences_migration_assets_pin");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto assets = paths->rootDir() / "assets";
    std::filesystem::create_directories(assets / "nested");
    std::ofstream(assets / "nested" / "catalog.licht") << "catalog";
    writePreferences(*paths, json::object());

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), assets);
}

TEST(PreferencesMigration, FreshPreferencesUseProjectsDefaultAndSetClearRoundTrip) {
    const auto home = makeHome("lfs_preferences_migration_fresh");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
    const auto custom = home / "custom";
    ASSERT_TRUE(lfs::vis::setProjectLocationPreference(custom));
    EXPECT_EQ(lfs::vis::projectLocationPreferenceRaw(), custom);
    lfs::vis::clearProjectLocationPreference();
    EXPECT_TRUE(lfs::vis::projectLocationPreferenceRaw().empty());
    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
}

TEST(PreferencesMigration, InvalidProjectLocationIsRejectedWithoutPersisting) {
    const auto home = makeHome("lfs_preferences_migration_invalid");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto file = home / "not-a-directory";
    std::ofstream(file) << "file";

    ASSERT_TRUE(lfs::vis::setProjectLocationPreference(home / "valid"));
    const auto before = readPreferences(*paths);
    EXPECT_FALSE(lfs::vis::setProjectLocationPreference(file));
    EXPECT_EQ(readPreferences(*paths), before);
}

TEST(PreferencesMigration, CameraSpeedPreferencesPersistAndClamp) {
    const auto home = makeHome("lfs_preferences_camera_speed");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());

    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_FLOAT_EQ(preferences.zoomSpeed(), 11.0f);
    EXPECT_FLOAT_EQ(preferences.navigationSpeed(), 8.0f);

    preferences.setZoomSpeed(42.0f);
    preferences.setNavigationSpeed(73.0f);
    EXPECT_FLOAT_EQ(preferences.zoomSpeed(), 42.0f);
    EXPECT_FLOAT_EQ(preferences.navigationSpeed(), 73.0f);
    EXPECT_FLOAT_EQ(readPreferences(*paths).at("zoom_speed"), 42.0f);
    EXPECT_FLOAT_EQ(readPreferences(*paths).at("navigation_speed"), 73.0f);

    preferences.setZoomSpeed(0.0f);
    preferences.setNavigationSpeed(101.0f);
    EXPECT_FLOAT_EQ(preferences.zoomSpeed(), 1.0f);
    EXPECT_FLOAT_EQ(preferences.navigationSpeed(), 100.0f);
}
