/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/path_utils.hpp"
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
