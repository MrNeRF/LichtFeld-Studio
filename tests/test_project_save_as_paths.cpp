/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/path_utils.hpp"
#include "io/project/project_filesystem.hpp"
#include "io/project/project_path_utils.hpp"

TEST(ProjectSaveAsPathTest, PreservesUnicodeDestinationFilename) {
    const auto destination =
        lfs::core::utf8_to_path("保存_копия.licht");

    const auto staging =
        lfs::io::project::detail::save_as_staging_path(
            destination, "fixed-token");

    EXPECT_EQ(
        lfs::core::path_to_utf8(staging.filename()),
        ".保存_копия.licht.saveas-fixed-token.tmp");
}

TEST(ProjectSaveAsPathTest, LongNamesLeaveRoomForNestedTemporaryFiles) {
    constexpr std::string_view token = "01234567-89ab-cdef-0123-456789abcdef";
    for (const size_t length : {size_t{128}, size_t{173}, size_t{240}}) {
        const auto destination = std::filesystem::path("parent") / (std::string(length - 6, 'x') + ".licht");
        const auto staging = lfs::io::project::detail::save_as_staging_path(destination, token);
        EXPECT_EQ(staging.parent_path(), destination.parent_path());
        constexpr std::string_view largest_suffix =
            ".replace-backup.-9223372036854775808.4294967295.18446744073709551615.tmp";
        EXPECT_LE(lfs::core::path_to_utf8(staging.filename()).size() + largest_suffix.size() + 5, 255u);
        EXPECT_NE(staging, lfs::io::project::detail::save_as_staging_path(destination, "different-token"));
    }
}

#ifdef _WIN32
TEST(ProjectSaveAsPathTest, NativePathConversionPreservesShortPathsAndNormalizesLongPaths) {
    namespace project_fs = lfs::io::project::detail::project_fs;
    const std::filesystem::path short_path = L"C:\\projects\\scene.licht";
    EXPECT_EQ(project_fs::native_path(short_path), short_path);
    const auto long_path = std::filesystem::path(L"C:\\projects") / std::wstring(200, L'a') / std::wstring(100, L'b') / std::wstring(100, L'c') / L".." / L"scene.licht";
    const auto native = project_fs::native_path(long_path);
    EXPECT_TRUE(native.native().starts_with(L"\\\\?\\C:\\"));
    EXPECT_EQ(project_fs::display_path(native), long_path.lexically_normal());
    EXPECT_EQ(project_fs::native_path(native), native);
    const auto unc = std::filesystem::path(L"\\\\server\\share") / std::wstring(200, L'a') / std::wstring(100, L'b') / L"scene.licht";
    const auto native_unc = project_fs::native_path(unc);
    EXPECT_TRUE(native_unc.native().starts_with(L"\\\\?\\UNC\\server\\share\\"));
    EXPECT_EQ(project_fs::display_path(native_unc), unc);
    const auto relative = std::filesystem::path(std::wstring(200, L'a')) / std::wstring(100, L'b') / L"scene.licht";
    EXPECT_EQ(project_fs::display_path(project_fs::native_path(relative)), std::filesystem::absolute(relative));
}
#endif
