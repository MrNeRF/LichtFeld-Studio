/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/argument_parser.hpp"

#include <filesystem>
#include <iterator>
#include <string>

namespace {

    std::string make_test_path(const char* name) {
        const auto path = std::filesystem::temp_directory_path() / name;
        std::filesystem::create_directories(path);
        return path.string();
    }

} // namespace

TEST(ArgumentParserTest, TrainingDefaultsDoNotApplyMaxWidthCap) {
    const auto data_path = make_test_path("lfs_arg_parser_default_data");
    const auto output_path = make_test_path("lfs_arg_parser_default_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->dataset.max_width, 0);
    EXPECT_EQ((*parsed)->dataset.resize_factor, 1);
}

TEST(ArgumentParserTest, MaxWidthCanBeExplicitlySet) {
    const auto data_path = make_test_path("lfs_arg_parser_explicit_data");
    const auto output_path = make_test_path("lfs_arg_parser_explicit_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--max-width",
        "8192"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->dataset.max_width, 8192);
}

TEST(ArgumentParserTest, MaxWidthZeroDisablesCapExplicitly) {
    const auto data_path = make_test_path("lfs_arg_parser_zero_data");
    const auto output_path = make_test_path("lfs_arg_parser_zero_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--max-width",
        "0"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->dataset.max_width, 0);
}
