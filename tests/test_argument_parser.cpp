/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/argument_parser.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <variant>
#include <vector>

namespace {

    std::string make_test_path(const char* name) {
        const auto path = std::filesystem::temp_directory_path() / name;
        std::filesystem::create_directories(path);
        return path.string();
    }

} // namespace

TEST(ArgumentParserTest, TrainingDefaultsApplyMaxWidthCap) {
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

    EXPECT_FALSE((*parsed)->cli_bg_color_set);
    EXPECT_EQ((*parsed)->dataset.max_width, 3840);
    EXPECT_EQ((*parsed)->dataset.resize_factor, 1);
    EXPECT_EQ((*parsed)->optimization.depth_loss_mode, "adaptive-warped-l1");
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

TEST(ArgumentParserTest, Mesh2SplatParsesOutputPathAndOptions) {
    const auto dir = make_test_path("lfs_mesh2splat_arg_parser");
    const auto input = std::filesystem::path(dir) / "input.obj";
    const auto output = std::filesystem::path(dir) / "output.spz";
    std::ofstream(input).put('\n');

    const std::string input_str = input.string();
    const std::string output_str = output.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "mesh2splat",
        input_str.c_str(),
        "--output",
        output_str.c_str(),
        "--resolution",
        "512",
        "--sigma",
        "0.5"};

    auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    auto* mode = std::get_if<lfs::core::args::Mesh2SplatMode>(&*parsed);
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(mode->params.input_path, input);
    EXPECT_EQ(mode->params.output_path, output);
    EXPECT_EQ(mode->params.format, lfs::core::param::OutputFormat::SPZ);
    ASSERT_EQ(mode->params.formats.size(), 1u);
    EXPECT_EQ(mode->params.formats[0], lfs::core::param::OutputFormat::SPZ);
    EXPECT_EQ(mode->params.options.resolution_target, 512);
    EXPECT_FLOAT_EQ(mode->params.options.sigma, 0.5f);
}

TEST(ArgumentParserTest, Mesh2SplatParsesMultipleOutputFormats) {
    const auto dir = make_test_path("lfs_mesh2splat_multi_format_arg_parser");
    const auto input = std::filesystem::path(dir) / "input.obj";
    const auto output = std::filesystem::path(dir) / "output";
    std::ofstream(input).put('\n');

    const std::string input_str = input.string();
    const std::string output_str = output.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "mesh2splat",
        input_str.c_str(),
        "--output",
        output_str.c_str(),
        "--format",
        "ply,spz,html"};

    auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    auto* mode = std::get_if<lfs::core::args::Mesh2SplatMode>(&*parsed);
    ASSERT_NE(mode, nullptr);
    ASSERT_EQ(mode->params.formats.size(), 3u);
    EXPECT_EQ(mode->params.formats[0], lfs::core::param::OutputFormat::PLY);
    EXPECT_EQ(mode->params.formats[1], lfs::core::param::OutputFormat::SPZ);
    EXPECT_EQ(mode->params.formats[2], lfs::core::param::OutputFormat::HTML);
}

TEST(ArgumentParserTest, TrainingParsesAddSplats) {
    const auto dir = make_test_path("lfs_arg_parser_add_splat");
    const auto data_path = std::filesystem::path(dir) / "data";
    const auto output_path = std::filesystem::path(dir) / "output";
    const auto splat_a = std::filesystem::path(dir) / "background.ply";
    const auto splat_b = std::filesystem::path(dir) / "sky.sog";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);
    std::ofstream(splat_a).put('\n');
    std::ofstream(splat_b).put('\n');

    const std::string data_str = data_path.string();
    const std::string output_str = output_path.string();
    const std::string splat_a_str = splat_a.string();
    const std::string splat_b_str = splat_b.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--add-splat",
        splat_a_str.c_str(),
        "--add-splat",
        splat_b_str.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    ASSERT_EQ((*parsed)->add_splat_paths.size(), 2u);
    EXPECT_EQ((*parsed)->add_splat_paths[0], splat_a);
    EXPECT_EQ((*parsed)->add_splat_paths[1], splat_b);
    EXPECT_EQ((*parsed)->add_splat_freeze, (std::vector<bool>{false, false}));
    EXPECT_FALSE((*parsed)->exclude_frozen_add_splats_from_export);
}

TEST(ArgumentParserTest, TrainingParsesFrozenAddSplatExcludeExport) {
    const auto dir = make_test_path("lfs_arg_parser_add_splat_exclude");
    const auto data_path = std::filesystem::path(dir) / "data";
    const auto output_path = std::filesystem::path(dir) / "output";
    const auto splat = std::filesystem::path(dir) / "background.ply";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);
    std::ofstream(splat).put('\n');

    const std::string data_str = data_path.string();
    const std::string output_str = output_path.string();
    const std::string splat_str = splat.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--add-splat",
        splat_str.c_str(),
        "--freeze",
        "--exclude-export"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    ASSERT_EQ((*parsed)->add_splat_paths.size(), 1u);
    EXPECT_EQ((*parsed)->add_splat_paths[0], splat);
    EXPECT_EQ((*parsed)->add_splat_freeze, (std::vector<bool>{true}));
    EXPECT_TRUE((*parsed)->exclude_frozen_add_splats_from_export);
}

TEST(ArgumentParserTest, TrainingParsesExplicitDepthLossOptions) {
    const auto data_path = make_test_path("lfs_arg_parser_depth_loss_data");
    const auto output_path = make_test_path("lfs_arg_parser_depth_loss_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--use-depth-loss",
        "--depth-loss-weight",
        "3.25",
        "--depth-loss-mode",
        "adaptive-warped-l1"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->optimization.use_depth_loss);
    EXPECT_FLOAT_EQ((*parsed)->optimization.depth_loss_weight, 3.25f);
    EXPECT_EQ((*parsed)->optimization.depth_loss_mode, "adaptive-warped-l1");
}

TEST(ArgumentParserTest, TrainingRejectsLegacyDepthLossAlias) {
    const auto data_path = make_test_path("lfs_arg_parser_depth_invalid_data");
    const auto output_path = make_test_path("lfs_arg_parser_depth_invalid_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--use-depth-loss",
        "--depth-loss-mode",
        "lod"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("depth_loss_mode must be 'pearson' or 'adaptive-warped-l1'"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesBackgroundModeModulation) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_mode_modulation_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_mode_modulation_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "modulation"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->optimization.bg_mode, lfs::core::param::BackgroundMode::Modulation);
    EXPECT_TRUE((*parsed)->optimization.bg_modulation);
}

TEST(ArgumentParserTest, TrainingRejectsFloatBackgroundColor) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "0.1,0.2,0.3"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-color must be #RRGGBB"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesHexBackgroundColor) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_hex_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_hex_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "#FF8040"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 128.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 64.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingParsesLowercaseHexBackgroundColor) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_lower_hex_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_lower_hex_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "#ff8040"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 128.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 64.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingParsesIntegerRgbBackgroundColorWithSpaces) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_rgb_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_rgb_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "(255, 64, 32)"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 64.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 32.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingParsesIntegerRgbBackgroundColorWithoutSpaces) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_rgb_compact_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_rgb_compact_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "(255,64,32)"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 64.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 32.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingRejectsHexBackgroundColorWithoutHash) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_hex_no_hash_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_hex_no_hash_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "FF8040"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-color must be #RRGGBB"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingRejectsRgbBackgroundColorOutOfRange) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_rgb_out_of_range_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_rgb_out_of_range_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "(256,64,32)"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-color must be #RRGGBB"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesImageBackgroundPath) {
    const auto dir = make_test_path("lfs_arg_parser_bg_image");
    const auto data_path = std::filesystem::path(dir) / "data";
    const auto output_path = std::filesystem::path(dir) / "output";
    const auto image_path = std::filesystem::path(dir) / "bg.png";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);
    std::ofstream(image_path).put('\n');

    const std::string data_str = data_path.string();
    const std::string output_str = output_path.string();
    const std::string image_str = image_path.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--bg-mode",
        "image",
        "--bg-image-path",
        image_str.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->optimization.bg_mode, lfs::core::param::BackgroundMode::Image);
    EXPECT_EQ((*parsed)->optimization.bg_image_path, image_path);
}

TEST(ArgumentParserTest, TrainingRejectsImageBackgroundWithoutPath) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_image_missing_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_image_missing_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "image"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-image-path is required"), std::string::npos);
}
