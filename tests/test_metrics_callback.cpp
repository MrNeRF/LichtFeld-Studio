/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "training/dataset.hpp"
#include "training/metrics/metrics.hpp"

#include <chrono>
#include <cmath>
#include <expected>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

TEST(MetricsEvaluator, UsesRenderCallback) {
    constexpr int width = 16;
    constexpr int height = 16;
    const auto unique_suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_dir = std::filesystem::temp_directory_path() /
                            ("lfs_metrics_callback_" + std::to_string(unique_suffix));
    ASSERT_TRUE(std::filesystem::create_directories(output_dir));
    const auto image_path = output_dir / "target.ppm";

    {
        std::ofstream image(image_path, std::ios::binary);
        ASSERT_TRUE(image.is_open());
        image << "P6\n"
              << width << ' ' << height << "\n255\n";
        const std::vector<unsigned char> pixels(
            static_cast<std::size_t>(width * height * 3), 128);
        image.write(reinterpret_cast<const char*>(pixels.data()),
                    static_cast<std::streamsize>(pixels.size()));
    }

    auto rotation = lfs::core::Tensor::from_vector(
        {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F},
        {std::size_t{3}, std::size_t{3}},
        lfs::core::Device::CPU);
    auto translation = lfs::core::Tensor::from_vector(
        {0.0F, 0.0F, 0.0F}, {std::size_t{3}}, lfs::core::Device::CPU);
    auto camera = std::make_shared<lfs::core::Camera>(
        rotation,
        translation,
        12.0F,
        12.0F,
        8.0F,
        8.0F,
        lfs::core::Tensor{},
        lfs::core::Tensor{},
        lfs::core::CameraModelType::PINHOLE,
        "target.ppm",
        image_path,
        std::filesystem::path{},
        width,
        height,
        1);
    auto dataset = std::make_shared<lfs::training::CameraDataset>(
        std::vector<std::shared_ptr<lfs::core::Camera>>{camera},
        lfs::training::DatasetConfig{},
        lfs::training::CameraDataset::Split::ALL);

    lfs::core::param::TrainingParameters params;
    params.dataset.output_path = output_dir;
    params.dataset.resize_factor = 1;
    params.dataset.max_width = 0;
    params.optimization.enable_eval = true;
    params.optimization.enable_save_eval_images = false;
    params.optimization.mask_mode = lfs::core::param::MaskMode::None;

    int callback_calls = 0;
    {
        lfs::training::MetricsEvaluator evaluator(params);
        const lfs::training::MetricsEvaluator::RenderCallback render =
            [&callback_calls](lfs::core::Camera& requested_camera)
            -> std::expected<lfs::core::Tensor, std::string> {
            ++callback_calls;
            return lfs::core::Tensor::full(
                {std::size_t{3},
                 static_cast<std::size_t>(requested_camera.image_height()),
                 static_cast<std::size_t>(requested_camera.image_width())},
                128.0F / 255.0F,
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        };

        const auto metrics = evaluator.evaluate(25, dataset, std::size_t{42}, render);
        EXPECT_TRUE(metrics.valid);
        EXPECT_EQ(metrics.iteration, 25);
        EXPECT_EQ(metrics.num_gaussians, 42);
        EXPECT_TRUE(std::isfinite(metrics.psnr));
        EXPECT_TRUE(std::isfinite(metrics.ssim));
    }

    EXPECT_EQ(callback_calls, 1);
    std::error_code cleanup_error;
    std::filesystem::remove_all(output_dir, cleanup_error);
    EXPECT_FALSE(cleanup_error);
}
