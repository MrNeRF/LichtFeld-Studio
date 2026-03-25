/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "app/include/app/converter.hpp"
#include "core/argument_parser.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::core::args;
using namespace lfs::core::param;
using namespace lfs::io;

namespace {

    class ConverterUsdTest : public ::testing::Test {
    protected:
        const fs::path temp_dir = fs::temp_directory_path() / "lfs_converter_usd_test";

        void SetUp() override {
            fs::create_directories(temp_dir);
        }

        void TearDown() override {
            fs::remove_all(temp_dir);
        }

        static SplatData create_test_splat() {
            auto means = Tensor::empty({2, 3}, Device::CPU, DataType::Float32);
            auto sh0 = Tensor::empty({2, 1, 3}, Device::CPU, DataType::Float32);
            auto scaling = Tensor::empty({2, 3}, Device::CPU, DataType::Float32);
            auto rotation = Tensor::empty({2, 4}, Device::CPU, DataType::Float32);
            auto opacity = Tensor::empty({2, 1}, Device::CPU, DataType::Float32);

            auto* const means_ptr = static_cast<float*>(means.data_ptr());
            auto* const sh0_ptr = static_cast<float*>(sh0.data_ptr());
            auto* const scaling_ptr = static_cast<float*>(scaling.data_ptr());
            auto* const rotation_ptr = static_cast<float*>(rotation.data_ptr());
            auto* const opacity_ptr = static_cast<float*>(opacity.data_ptr());

            for (size_t i = 0; i < 2; ++i) {
                means_ptr[i * 3 + 0] = static_cast<float>(i);
                means_ptr[i * 3 + 1] = static_cast<float>(i + 1);
                means_ptr[i * 3 + 2] = static_cast<float>(i + 2);

                sh0_ptr[i * 3 + 0] = 0.1f * static_cast<float>(i + 1);
                sh0_ptr[i * 3 + 1] = 0.2f * static_cast<float>(i + 1);
                sh0_ptr[i * 3 + 2] = 0.3f * static_cast<float>(i + 1);

                scaling_ptr[i * 3 + 0] = -1.0f;
                scaling_ptr[i * 3 + 1] = -1.1f;
                scaling_ptr[i * 3 + 2] = -1.2f;

                rotation_ptr[i * 4 + 0] = 1.0f;
                rotation_ptr[i * 4 + 1] = 0.0f;
                rotation_ptr[i * 4 + 2] = 0.0f;
                rotation_ptr[i * 4 + 3] = 0.0f;

                opacity_ptr[i] = 0.0f;
            }

            return SplatData(
                0,
                std::move(means),
                std::move(sh0),
                Tensor{},
                std::move(scaling),
                std::move(rotation),
                std::move(opacity),
                0.5f);
        }
    };

    TEST_F(ConverterUsdTest, ParseArgsPreservesUsdFlavor) {
        const fs::path input = temp_dir / "input.ply";
        std::ofstream(input).put('\n');

        const std::string input_str = input.string();
        const char* argv[] = {"LichtFeld-Studio", "convert", input_str.c_str(), "-f", "usda"};

        auto parsed = lfs::core::args::parse_args(5, argv);
        ASSERT_TRUE(parsed.has_value()) << parsed.error();

        auto* mode = std::get_if<ConvertMode>(&*parsed);
        ASSERT_NE(mode, nullptr);
        EXPECT_EQ(mode->params.format, OutputFormat::USDA);
    }

    TEST_F(ConverterUsdTest, ParseArgsInfersUsdcFromOutputExtension) {
        const fs::path input = temp_dir / "input.ply";
        const fs::path output = temp_dir / "output.usdc";
        std::ofstream(input).put('\n');

        const std::string input_str = input.string();
        const std::string output_str = output.string();
        const char* argv[] = {"LichtFeld-Studio", "convert", input_str.c_str(), output_str.c_str()};

        auto parsed = lfs::core::args::parse_args(4, argv);
        ASSERT_TRUE(parsed.has_value()) << parsed.error();

        auto* mode = std::get_if<ConvertMode>(&*parsed);
        ASSERT_NE(mode, nullptr);
        EXPECT_EQ(mode->params.format, OutputFormat::USDC);
    }

    TEST_F(ConverterUsdTest, DirectoryConversionFindsUsdInputsAndKeepsUsdcExtension) {
        const fs::path input_dir = temp_dir / "input";
        const fs::path output_dir = temp_dir / "output";
        fs::create_directories(input_dir);
        fs::create_directories(output_dir);

        const fs::path usd_input = input_dir / "scene.usda";
        ASSERT_TRUE(save_usd(create_test_splat(), {.output_path = usd_input}).has_value());

        ConvertParameters params;
        params.input_path = input_dir;
        params.output_path = output_dir;
        params.format = OutputFormat::USDC;
        params.sh_degree = -1;
        params.overwrite = true;

        EXPECT_EQ(lfs::app::run_converter(params), 0);
        EXPECT_TRUE(fs::exists(output_dir / "scene_converted.usdc"));
    }

} // namespace
