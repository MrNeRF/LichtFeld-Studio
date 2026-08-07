/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/user_paths.hpp"

#include <chrono>
#include <expected>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace {

    namespace fs = std::filesystem;

    class UserPathsContractTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            root_ = fs::temp_directory_path() / ("lfs_user_paths_" + std::to_string(nonce));
            std::error_code error;
            fs::remove_all(root_, error);
        }

        void TearDown() override {
            std::error_code error;
            fs::remove_all(root_, error);
        }

        std::expected<lfs::core::UserPaths, std::string> resolvePaths() const {
            const lfs::core::UserPathOptions options{
                .explicit_root = root_ / "current",
            };
            return lfs::core::UserPaths::resolve(options);
        }

        fs::path portableExecutableDir() const { return root_ / "portable-app"; }

        fs::path root_;
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
        EXPECT_FALSE(fs::exists(root_ / "LichtFeldStudio"));
    }

    TEST_F(UserPathsContractTest, ResetPreferencesWritesDefaults) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());
        auto reset = paths.resetPreferences();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        EXPECT_TRUE(fs::is_regular_file(paths.preferencesFile()));

        std::ifstream preferences(paths.preferencesFile(), std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(preferences)), {});
        EXPECT_NE(contents.find("\"theme\": \"dark\""), std::string::npos);
        EXPECT_NE(contents.find("\"ui_scale\": \"auto\""), std::string::npos);
    }

    TEST_F(UserPathsContractTest, ResetPreferencesBacksUpExistingFile) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        {
            std::ofstream preferences(paths.preferencesFile(), std::ios::binary);
            preferences << R"({"theme":"light","ui_scale":"150"})";
        }

        const auto reset = paths.resetPreferences();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        ASSERT_TRUE(reset->has_value());
        EXPECT_TRUE(fs::is_regular_file(**reset));
        EXPECT_TRUE(fs::is_regular_file(paths.preferencesFile()));
        std::ifstream backup(**reset, std::ios::binary);
        const std::string backup_contents((std::istreambuf_iterator<char>(backup)), {});
        EXPECT_EQ(backup_contents, R"({"theme":"light","ui_scale":"150"})");
    }

    TEST_F(UserPathsContractTest, PortableRootUsesExecutableDirectory) {
        const lfs::core::UserPathOptions options{
            .executable_dir = portableExecutableDir(),
            .portable = true,
        };
        const auto resolved = lfs::core::UserPaths::resolve(options);
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_TRUE(resolved->usesUnifiedRoot());
        EXPECT_EQ(resolved->configDir(), portableExecutableDir() / ".lichtfeld" / "config");
        EXPECT_EQ(resolved->dataDir(), portableExecutableDir() / ".lichtfeld" / "data");
    }

    TEST_F(UserPathsContractTest, PortableRootRequiresExecutableDirectory) {
        const lfs::core::UserPathOptions options{.portable = true};
        const auto resolved = lfs::core::UserPaths::resolve(options);
        ASSERT_FALSE(resolved.has_value());
        EXPECT_NE(resolved.error().find("executable directory"), std::string::npos);
    }

    TEST_F(UserPathsContractTest, AtomicPreferenceWriteCreatesValidJsonAndNoTemporaryFiles) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        const auto write = paths.writePreferencesAtomically(R"({"theme":"light","ui_scale":"auto"})");
        ASSERT_TRUE(write.has_value()) << write.error();
        ASSERT_TRUE(fs::is_regular_file(paths.preferencesFile()));
        std::ifstream file(paths.preferencesFile());
        const auto json = nlohmann::json::parse(file);
        EXPECT_EQ(json.at("theme"), "light");
        EXPECT_EQ(json.at("ui_scale"), "auto");

        for (const auto& entry : fs::directory_iterator(paths.configDir()))
            EXPECT_EQ(entry.path().filename().string().find("preferences.json.tmp-"), std::string::npos);
    }

    TEST_F(UserPathsContractTest, AtomicWindowWriteCreatesParentDirectory) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.writeWindowStateAtomically(R"({"width":1280,"height":720})").has_value());
        EXPECT_TRUE(fs::is_regular_file(paths.windowStateFile()));
    }

    TEST_F(UserPathsContractTest, ResetLayoutBacksUpExistingFile) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        {
            std::ofstream layout(paths.layoutFile(), std::ios::binary);
            layout << R"({"right_panel_width":420})";
        }

        const auto reset = paths.resetLayout();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        ASSERT_TRUE(reset->has_value());
        EXPECT_TRUE(fs::is_regular_file(**reset));
        EXPECT_FALSE(fs::exists(paths.layoutFile()));
    }

    TEST_F(UserPathsContractTest, ResetWindowStateBacksUpExistingFile) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        {
            std::ofstream window(paths.windowStateFile(), std::ios::binary);
            window << R"({"x":10,"y":20,"width":1280,"height":720,"maximized":false})";
        }

        const auto reset = paths.resetWindowState();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        ASSERT_TRUE(reset->has_value());
        EXPECT_TRUE(fs::is_regular_file(**reset));
        EXPECT_FALSE(fs::exists(paths.windowStateFile()));
    }

    TEST_F(UserPathsContractTest, ResetWithoutExistingFilesCreatesNoBackup) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        const auto preferences_reset = paths.resetPreferences();
        ASSERT_TRUE(preferences_reset.has_value()) << preferences_reset.error();
        EXPECT_FALSE(preferences_reset->has_value());

        const auto layout_reset = paths.resetLayout();
        ASSERT_TRUE(layout_reset.has_value()) << layout_reset.error();
        EXPECT_FALSE(layout_reset->has_value());

        const auto window_reset = paths.resetWindowState();
        ASSERT_TRUE(window_reset.has_value()) << window_reset.error();
        EXPECT_FALSE(window_reset->has_value());
    }

} // namespace
