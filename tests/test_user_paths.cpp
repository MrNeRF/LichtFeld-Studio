/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/user_paths.hpp"

#include <chrono>
#include <filesystem>
#include <format>

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
