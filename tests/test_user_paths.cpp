/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/user_paths.hpp"

#include <chrono>
#include <expected>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

    namespace fs = std::filesystem;

    class UserPathsContractTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            root_ = fs::temp_directory_path() / ("lfs_user_paths_" + std::to_string(nonce));
            legacy_dir_ = root_ / "legacy";
            std::error_code error;
            fs::remove_all(root_, error);
            ASSERT_TRUE(fs::create_directories(legacy_dir_));
        }

        void TearDown() override {
            std::error_code error;
            fs::remove_all(root_, error);
        }

        std::expected<lfs::core::UserPaths, std::string> resolvePaths() const {
            const lfs::core::UserPathOptions options{
                .explicit_root = root_ / "current",
                .legacy_config_dirs = {legacy_dir_},
                .automatic_legacy_migration = true,
            };
            return lfs::core::UserPaths::resolve(options);
        }

        static void writeText(const fs::path& path, const std::string& contents) {
            std::ofstream stream(path, std::ios::binary);
            ASSERT_TRUE(stream.is_open());
            stream << contents;
            ASSERT_TRUE(stream.good());
        }

        fs::path root_;
        fs::path legacy_dir_;
    };

    TEST_F(UserPathsContractTest, ExplicitRootUsesOnlyTheUnifiedStorageTree) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        EXPECT_TRUE(paths.usesUnifiedRoot());
        EXPECT_TRUE(fs::is_directory(paths.configDir()));
        EXPECT_TRUE(fs::is_directory(paths.dataDir()));
        EXPECT_TRUE(fs::is_directory(paths.cacheDir()));
        EXPECT_TRUE(fs::is_directory(paths.logDir()));
        EXPECT_TRUE(fs::is_directory(paths.pluginDir()));
        EXPECT_TRUE(fs::is_directory(paths.venvDir()));
        EXPECT_FALSE(fs::exists(legacy_dir_ / "LichtFeldStudio"));
    }

    TEST_F(UserPathsContractTest, MigrationCopiesValidSettingsAndNeverOverwritesTheirSources) {
        writeText(legacy_dir_ / "layout.json", "{}");
        writeText(legacy_dir_ / "theme_preference", "dark\n");
        writeText(legacy_dir_ / "ui_scale", "1.5\n");

        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        auto migrated = paths.migrateLegacyGuiSettings();
        ASSERT_TRUE(migrated.has_value()) << migrated.error();

        EXPECT_TRUE(fs::is_regular_file(paths.layoutFile()));
        EXPECT_TRUE(fs::is_regular_file(paths.preferencesFile()));
        EXPECT_TRUE(fs::is_regular_file(paths.migrationManifestFile()));
        EXPECT_TRUE(fs::is_regular_file(legacy_dir_ / "layout.json"));
        EXPECT_TRUE(fs::is_regular_file(legacy_dir_ / "theme_preference"));
        EXPECT_TRUE(fs::is_regular_file(legacy_dir_ / "ui_scale"));

        std::ifstream preferences(paths.preferencesFile(), std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(preferences)), {});
        EXPECT_NE(contents.find("\"theme\": \"dark\""), std::string::npos);
        EXPECT_NE(contents.find("\"ui_scale\": 1.5"), std::string::npos);

        auto repeated = paths.migrateLegacyGuiSettings();
        ASSERT_TRUE(repeated.has_value()) << repeated.error();
        EXPECT_TRUE(repeated->empty());
    }

} // namespace
