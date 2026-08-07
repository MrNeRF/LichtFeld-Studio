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

} // namespace
