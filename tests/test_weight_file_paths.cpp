/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/weight_file.hpp"
#include "core/path_utils.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <string_view>

TEST(WeightFileTest, MissingUnicodePathIsReportedAsUtf8) {
    const auto path = std::filesystem::temp_directory_path() /
                      lfs::core::utf8_to_path("missing_weights_重み.lfw");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ASSERT_FALSE(ec) << ec.message();

    const auto result = lfs::core::nn::WeightFile::open(path);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), lfs::ErrorCode::NotFound);
    EXPECT_NE(result.error().detail().find(lfs::core::path_to_utf8(path)),
              std::string_view::npos);
}
