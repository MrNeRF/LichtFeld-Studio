/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/file_association.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace {
    namespace detail = lfs::vis::gui::detail;

    TEST(FileAssociationTest, OpenCommandYieldsTheQuotedExecutable) {
        const auto path = detail::executableFromOpenCommand(
            L"\"C:\\Tools\\LichtFeld Studio\\LichtFeld-Studio.exe\" \"%1\"");
        ASSERT_TRUE(path);
        EXPECT_EQ(path->wstring(), L"C:\\Tools\\LichtFeld Studio\\LichtFeld-Studio.exe");
    }

    TEST(FileAssociationTest, OpenCommandWithoutQuotesEndsAtTheFirstSpace) {
        const auto path = detail::executableFromOpenCommand(L"C:\\Tools\\LichtFeld-Studio.exe %1");
        ASSERT_TRUE(path);
        EXPECT_EQ(path->wstring(), L"C:\\Tools\\LichtFeld-Studio.exe");
    }

    TEST(FileAssociationTest, MalformedOpenCommandHasNoExecutable) {
        EXPECT_FALSE(detail::executableFromOpenCommand(L""));
        EXPECT_FALSE(detail::executableFromOpenCommand(L"   "));
        EXPECT_FALSE(detail::executableFromOpenCommand(L"\"C:\\unterminated.exe %1"));
        EXPECT_FALSE(detail::executableFromOpenCommand(L"\"\" \"%1\""));
    }

    TEST(FileAssociationTest, ProjectAssociationRegistersOnlyWhenMissingOrMovedAndNotOptedOut) {
        const auto directory = std::filesystem::path(testing::TempDir()) / "lfs_file_association";
        std::filesystem::create_directories(directory);
        const auto executable = directory / "LichtFeld-Studio.exe";
        std::ofstream(executable) << "";

        EXPECT_TRUE(detail::projectAssociationNeedsRegistration(false, std::nullopt));
        EXPECT_TRUE(detail::projectAssociationNeedsRegistration(false, directory / "moved.exe"));
        EXPECT_FALSE(detail::projectAssociationNeedsRegistration(false, executable));
        EXPECT_FALSE(detail::projectAssociationNeedsRegistration(true, std::nullopt));
        EXPECT_FALSE(detail::projectAssociationNeedsRegistration(true, directory / "moved.exe"));

        std::filesystem::remove_all(directory);
    }
} // namespace
