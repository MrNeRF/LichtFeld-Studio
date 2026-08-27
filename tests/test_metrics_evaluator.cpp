/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "training/dataset.hpp"
#include "training/metrics/metrics.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

using lfs::core::Camera;
using lfs::core::CameraModelType;
using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::SplatData;
using lfs::core::Tensor;
using lfs::training::CameraDataset;
using lfs::training::DatasetConfig;
using lfs::training::EvalMetrics;
using lfs::training::mean_normal_angle_deg;
using lfs::training::median_depth_absrel;
using lfs::training::MetricsEvaluator;

namespace {

    constexpr float kPi = 3.14159265358979323846f;

    std::vector<float> chw_from_normal(const float x, const float y, const float z,
                                       const int height, const int width) {
        const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
        std::vector<float> chw(hw * 3);
        for (size_t i = 0; i < hw; ++i) {
            chw[i] = x;
            chw[hw + i] = y;
            chw[2 * hw + i] = z;
        }
        return chw;
    }

    Tensor cpu_chw(const std::vector<float>& data, const int channels, const int height, const int width) {
        return Tensor::from_blob(
                   const_cast<float*>(data.data()),
                   {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
                   Device::CPU,
                   DataType::Float32)
            .clone();
    }

    Tensor cpu_hw(const std::vector<float>& data, const int height, const int width) {
        return Tensor::from_blob(
                   const_cast<float*>(data.data()),
                   {static_cast<size_t>(height), static_cast<size_t>(width)},
                   Device::CPU,
                   DataType::Float32)
            .clone();
    }

    void write_u8_hwc_png(const std::filesystem::path& path,
                          const std::vector<uint8_t>& hwc,
                          const int height,
                          const int width) {
        auto image = Tensor::from_blob(
                         const_cast<uint8_t*>(hwc.data()),
                         {static_cast<size_t>(height), static_cast<size_t>(width), size_t{3}},
                         Device::CPU,
                         DataType::UInt8)
                         .clone();
        lfs::core::save_image_u8(path, std::move(image));
    }

    void write_rgb_png(const std::filesystem::path& path,
                       const uint8_t r, const uint8_t g, const uint8_t b,
                       const int height, const int width) {
        std::vector<uint8_t> hwc(static_cast<size_t>(height) * static_cast<size_t>(width) * 3);
        for (size_t i = 0; i < hwc.size(); i += 3) {
            hwc[i] = r;
            hwc[i + 1] = g;
            hwc[i + 2] = b;
        }
        write_u8_hwc_png(path, hwc, height, width);
    }

    void write_normal_png(const std::filesystem::path& path,
                          const float nx, const float ny, const float nz,
                          const int height, const int width) {
        const auto encode = [](const float v) {
            const float u = std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f);
            return static_cast<uint8_t>(std::lround(u * 255.0f));
        };
        write_rgb_png(path, encode(nx), encode(ny), encode(nz), height, width);
    }

    SplatData make_front_facing_splat() {
        std::vector<float> means_data{0.0f, 0.0f, 1.0f};
        std::vector<float> scaling_data{2.0f, 1.5f, -3.0f};
        std::vector<float> rotation_data{1.0f, 0.0f, 0.0f, 0.0f};
        auto means = Tensor::from_blob(means_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto sh0 = Tensor::zeros({1, 1, 3}, Device::CUDA);
        auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
        auto scaling = Tensor::from_blob(scaling_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto rotation = Tensor::from_blob(rotation_data.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
        const float opacity_value = 0.99f;
        const float raw_opacity = std::log(opacity_value / (1.0f - opacity_value));
        auto opacity = Tensor::full({1}, raw_opacity, Device::CUDA);
        return SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    std::shared_ptr<Camera> make_eval_camera(const std::filesystem::path& image_path,
                                             const std::filesystem::path& normal_path,
                                             const int width,
                                             const int height) {
        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0.0f, 0.0f, 4.0f};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        const float fx = static_cast<float>(width);
        const float fy = static_cast<float>(height);
        const float cx = 0.5f * static_cast<float>(width);
        const float cy = 0.5f * static_cast<float>(height);
        auto cam = std::make_shared<Camera>(
            R, T, fx, fy, cx, cy,
            Tensor(), Tensor(), CameraModelType::PINHOLE,
            image_path.filename().string(), image_path, std::filesystem::path{},
            width, height, 0);
        if (!normal_path.empty()) {
            cam->set_normal_path(normal_path);
        }
        return cam;
    }

    void ensure_image_loader() {
        static bool initialized = false;
        if (initialized) {
            return;
        }
        lfs::io::CacheLoader::getInstance(false);
        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path,
                {.resize_factor = p.resize_factor,
                 .max_width = p.max_width,
                 .cuda_stream = p.stream,
                 .output_uint8 = p.output_uint8});
        });
        initialized = true;
    }

    lfs::core::param::TrainingParameters make_eval_params(const std::filesystem::path& output_dir) {
        lfs::core::param::TrainingParameters params;
        params.optimization.enable_eval = true;
        params.optimization.enable_save_eval_images = false;
        params.optimization.eval_steps = {1};
        params.optimization.gut = false;
        params.dataset.output_path = output_dir;
        params.dataset.resize_factor = -1;
        params.dataset.max_width = 0;
        return params;
    }

} // namespace

TEST(EvalMetricsCsv, HeaderAppendsGeometryColumnsWithoutRenamingExisting) {
    EXPECT_EQ(EvalMetrics::to_csv_header(),
              "iteration,psnr,ssim,time_per_image,num_gaussians,normal_angle_deg,depth_absrel");

    EvalMetrics missing;
    missing.iteration = 200;
    missing.psnr = 1.0f;
    missing.ssim = 0.5f;
    missing.elapsed_time = 0.01f;
    missing.num_gaussians = 10;
    EXPECT_EQ(missing.to_csv_row(), "200,1.000000,0.500000,0.010000,10,,");

    EvalMetrics present = missing;
    present.normal_angle_deg = 12.5f;
    present.depth_absrel = 0.25f;
    EXPECT_EQ(present.to_csv_row(), "200,1.000000,0.500000,0.010000,10,12.500000,0.250000");
}

TEST(GeomMetricHelpers, MatchingNormalsYieldZeroAngle) {
    constexpr int kH = 4;
    constexpr int kW = 4;
    const auto rendered = chw_from_normal(0.0f, 0.0f, -1.0f, kH, kW);
    const auto prior = chw_from_normal(0.0f, 0.0f, -1.0f, kH, kW);
    const std::vector<float> alpha(static_cast<size_t>(kH * kW), 1.0f);

    const auto angle = mean_normal_angle_deg(
        cpu_chw(rendered, 3, kH, kW),
        cpu_chw(prior, 3, kH, kW),
        cpu_hw(alpha, kH, kW));
    ASSERT_TRUE(angle.has_value());
    EXPECT_NEAR(*angle, 0.0f, 1.0e-4f);
}

TEST(GeomMetricHelpers, RotatedPriorReturnsKnownAngle) {
    constexpr int kH = 4;
    constexpr int kW = 4;
    constexpr float kDeg = 30.0f;
    const float rad = kDeg * kPi / 180.0f;
    const auto rendered = chw_from_normal(0.0f, 0.0f, -1.0f, kH, kW);
    const auto prior = chw_from_normal(0.0f, -std::sin(rad), -std::cos(rad), kH, kW);
    const std::vector<float> alpha(static_cast<size_t>(kH * kW), 1.0f);

    const auto angle = mean_normal_angle_deg(
        cpu_chw(rendered, 3, kH, kW),
        cpu_chw(prior, 3, kH, kW),
        cpu_hw(alpha, kH, kW));
    ASSERT_TRUE(angle.has_value());
    EXPECT_NEAR(*angle, kDeg, 1.0e-3f);
}

TEST(GeomMetricHelpers, ZeroPriorIsMaskedOut) {
    constexpr int kH = 2;
    constexpr int kW = 2;
    const auto rendered = chw_from_normal(0.0f, 0.0f, -1.0f, kH, kW);
    const auto prior = chw_from_normal(0.0f, 0.0f, 0.0f, kH, kW);
    const std::vector<float> alpha(static_cast<size_t>(kH * kW), 1.0f);

    const auto angle = mean_normal_angle_deg(
        cpu_chw(rendered, 3, kH, kW),
        cpu_chw(prior, 3, kH, kW),
        cpu_hw(alpha, kH, kW));
    EXPECT_FALSE(angle.has_value());
}

TEST(GeomMetricHelpers, FlatRenderedDepthAbsRel) {
    constexpr int kH = 8;
    constexpr int kW = 8;
    const std::vector<float> depth(static_cast<size_t>(kH * kW), 4.0f);
    const std::vector<lfs::training::DepthAbsRelSample> samples{{3.5f, 4.5f, 5.0f}};

    const auto absrel = median_depth_absrel(cpu_hw(depth, kH, kW), samples);
    ASSERT_TRUE(absrel.has_value());
    EXPECT_NEAR(*absrel, 0.2f, 1.0e-5f);
}

TEST(MetricsEvaluatorGeom, MatchingRenderedAndPriorNormalIsNearZero) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_geom_metrics_match";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 1;
    constexpr int kH = 1;
    const auto image_path = tmp / "gt.png";
    const auto normal_path = tmp / "normal.png";
    write_rgb_png(image_path, 128, 128, 128, kH, kW);
    write_normal_png(normal_path, 0.0f, 0.0f, -1.0f, kH, kW);

    auto cam = make_eval_camera(image_path, normal_path, kW, kH);
    ASSERT_TRUE(std::filesystem::exists(normal_path));
    ASSERT_EQ(cam->normal_path(), normal_path);
    ASSERT_TRUE(cam->has_normal()) << cam->normal_path();
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam}, DatasetConfig{}, CameraDataset::Split::ALL);
    auto splat = make_front_facing_splat();
    auto background = Tensor::zeros({3}, Device::CUDA);
    auto params = make_eval_params(tmp / "out");
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, splat, dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_TRUE(metrics.normal_angle_deg.has_value());
    EXPECT_NEAR(*metrics.normal_angle_deg, 0.0f, 2.0f);
    EXPECT_EQ(EvalMetrics::to_csv_header(),
              "iteration,psnr,ssim,time_per_image,num_gaussians,normal_angle_deg,depth_absrel");

    std::filesystem::remove_all(tmp);
}

TEST(MetricsEvaluatorGeom, RotatedPriorReportsKnownAngle) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_geom_metrics_rot";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 1;
    constexpr int kH = 1;
    constexpr float kDeg = 30.0f;
    const float rad = kDeg * kPi / 180.0f;
    const auto image_path = tmp / "gt.png";
    const auto normal_path = tmp / "normal.png";
    write_rgb_png(image_path, 128, 128, 128, kH, kW);
    write_normal_png(normal_path, 0.0f, -std::sin(rad), -std::cos(rad), kH, kW);

    auto cam = make_eval_camera(image_path, normal_path, kW, kH);
    ASSERT_TRUE(cam->has_normal()) << cam->normal_path();
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam}, DatasetConfig{}, CameraDataset::Split::ALL);
    auto splat = make_front_facing_splat();
    auto background = Tensor::zeros({3}, Device::CUDA);
    auto params = make_eval_params(tmp / "out");
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, splat, dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_TRUE(metrics.normal_angle_deg.has_value());
    EXPECT_NEAR(*metrics.normal_angle_deg, kDeg, 2.0f);

    std::filesystem::remove_all(tmp);
}

TEST(MetricsEvaluatorGeom, SparsePointAbsRelAgainstRenderedDepth) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_geom_metrics_depth";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 16;
    constexpr int kH = 16;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 128, 128, 128, kH, kW);

    auto cam = make_eval_camera(image_path, {}, kW, kH);
    cam->set_sfm_observations({Camera::SfmObservation{
        .u = 0.5f * static_cast<float>(kW),
        .v = 0.5f * static_cast<float>(kH),
        .x = 0.0f,
        .y = 0.0f,
        .z = 1.0f}});
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam}, DatasetConfig{}, CameraDataset::Split::ALL);
    auto splat = make_front_facing_splat();
    auto background = Tensor::zeros({3}, Device::CUDA);
    auto params = make_eval_params(tmp / "out");
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, splat, dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_TRUE(metrics.depth_absrel.has_value());
    EXPECT_LT(*metrics.depth_absrel, 0.15f);

    std::filesystem::remove_all(tmp);
}
