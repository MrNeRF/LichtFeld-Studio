/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "app/converter_overwrite.hpp"
#include "core/path_utils.hpp"

#include <sstream>
#include <string>

TEST(ConverterOverwriteTest, PromptPreservesUnicodeFilename) {
    const auto path = lfs::core::utf8_to_path("出力_é.ply");
    std::istringstream input("n\n");
    std::ostringstream output;

    const auto choice = lfs::app::ask_overwrite(path, input, output);

    EXPECT_EQ(choice, lfs::app::OverwriteChoice::NO);
    EXPECT_NE(output.str().find(lfs::core::path_to_utf8(path.filename())),
              std::string::npos);
}
