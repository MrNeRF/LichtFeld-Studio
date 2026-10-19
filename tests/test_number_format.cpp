/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/number_format.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

TEST(NumberFormatTest, GroupsDigitsInThrees) {
    using lfs::core::format_count;
    EXPECT_EQ(format_count(0), "0");
    EXPECT_EQ(format_count(999), "999");
    EXPECT_EQ(format_count(1000), "1,000");
    EXPECT_EQ(format_count(30000), "30,000");
    EXPECT_EQ(format_count(std::size_t{5'000'000}), "5,000,000");
    EXPECT_EQ(format_count(std::uint64_t{1} << 62), "4,611,686,018,427,387,904");
    EXPECT_EQ(format_count(-999), "-999");
    EXPECT_EQ(format_count(-1000), "-1,000");
    EXPECT_EQ(format_count(std::numeric_limits<std::int64_t>::min()), "-9,223,372,036,854,775,808");
}
