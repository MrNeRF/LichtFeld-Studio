/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "io/cache_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "lfs/training/ops/fast_services.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/dataset.hpp"
#include "training/metrics/metrics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <nlohmann/json.hpp>
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
using lfs::training::image_for_metrics_and_save;
using lfs::training::mean_normal_angle_deg;
using lfs::training::median_depth_absrel;
using lfs::training::MetricsEvaluator;
using lfs::training::prepare_evaluation_view;

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
        auto means = Tensor::from_blob(means_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto sh0 = Tensor::zeros({1, 1, 3}, Device::GPU);
        auto shN = Tensor::zeros({1, 0, 3}, Device::GPU);
        auto scaling = Tensor::from_blob(scaling_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto rotation = Tensor::from_blob(rotation_data.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        const float opacity_value = 0.99f;
        const float raw_opacity = std::log(opacity_value / (1.0f - opacity_value));
        auto opacity = Tensor::full({1}, raw_opacity, Device::GPU);
        return SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    std::shared_ptr<Camera> make_eval_camera(const std::filesystem::path& image_path,
                                             const std::filesystem::path& normal_path,
                                             const int width,
                                             const int height) {
        auto R = Tensor::eye(3, Device::GPU);
        std::vector<float> t_data{0.0f, 0.0f, 4.0f};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
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

    std::shared_ptr<Camera> make_distorted_eval_camera(const std::filesystem::path& image_path,
                                                       const std::filesystem::path& mask_path,
                                                       const int width,
                                                       const int height) {
        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0.0f, 0.0f, 4.0f};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto radial = Tensor::from_vector({-0.3f}, lfs::core::TensorShape({1}), Device::CPU);
        auto cam = std::make_shared<Camera>(
            R, T, static_cast<float>(width), static_cast<float>(width),
            0.5f * static_cast<float>(width), 0.5f * static_cast<float>(height),
            radial, Tensor(), CameraModelType::PINHOLE,
            image_path.filename().string(), image_path, mask_path,
            width, height, 0);
        cam->prepare_undistortion();
        assert(cam->is_undistort_prepared());
        return cam;
    }

    SplatData make_hidden_splat() {
        auto splat = make_front_facing_splat();
        splat.means() = Tensor::from_vector({0.0f, 0.0f, -10.0f}, lfs::core::TensorShape({1, 3}), Device::CUDA);
        return splat;
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
              "iteration,psnr,ssim,lpips,time_per_image,num_gaussians,normal_angle_deg,depth_absrel,bias_r,bias_g,bias_b,bias_corr_r,bias_corr_g,bias_corr_b");

    EvalMetrics missing;
    missing.iteration = 200;
    missing.psnr = 1.0f;
    missing.ssim = 0.5f;
    missing.elapsed_time = 0.01f;
    missing.num_gaussians = 10;
    EXPECT_EQ(missing.to_csv_row(), "200,1.000000,0.500000,,0.010000,10,,,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000");

    EvalMetrics present = missing;
    present.normal_angle_deg = 12.5f;
    present.depth_absrel = 0.25f;
    present.bias_r = 0.001f;
    present.bias_g = -0.002f;
    present.bias_b = 0.003f;
    EXPECT_EQ(present.to_csv_row(), "200,1.000000,0.500000,,0.010000,10,12.500000,0.250000,0.001000,-0.002000,0.003000,0.000000,0.000000,0.000000");
}

TEST(EvalMetricsEvent, ZeroLpipsRemainsPresent) {
    std::optional<float> observed;
    const auto handler_id = lfs::core::events::state::EvaluationCompleted::when(
        [&observed](const auto& event) { observed = event.lpips; });
    lfs::core::events::state::EvaluationCompleted{
        .iteration = 1,
        .psnr = 1.0f,
        .ssim = 1.0f,
        .lpips = 0.0f,
        .elapsed_time = 0.0f,
        .num_gaussians = 0}
        .emit();
    lfs::event::EventBridge::instance().unsubscribe(
        typeid(lfs::core::events::state::EvaluationCompleted), handler_id);
    ASSERT_TRUE(observed.has_value());
    EXPECT_FLOAT_EQ(*observed, 0.0f);
}

TEST(EvalMetricsImage, QuantizesWithImageSaverRounding) {
    const std::vector<float> values{
        1.0f,
        2.0f,
        254.0f / 255.0f,
        0.5f,
        0.4999f / 255.0f,
        0.5001f / 255.0f,
        -0.1f,
        1.1f};
    std::vector<float> image_data;
    image_data.reserve(values.size() * 3);
    for (const float value : values) {
        image_data.insert(image_data.end(), {value, value, value});
    }
    const auto input = Tensor::from_blob(
                           image_data.data(), {values.size(), 1, 3}, Device::CPU,
                           DataType::Float32)
                           .clone();
    const auto quantized = image_for_metrics_and_save(input).to_vector();
    ASSERT_EQ(quantized.size(), image_data.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto expected = static_cast<float>(std::lround(std::clamp(values[i], 0.0f, 1.0f) * 255.0f)) /
                              255.0f;
        for (int channel = 0; channel < 3; ++channel) {
            EXPECT_FLOAT_EQ(quantized[i * 3 + channel], expected) << "index " << i;
        }
    }

    const auto path = std::filesystem::temp_directory_path() / "lfs_metrics_quantized_rounding.png";
    lfs::core::save_image(path, image_for_metrics_and_save(input));
    auto [saved, width, height, channels] = lfs::core::load_image(path);
    ASSERT_NE(saved, nullptr);
    ASSERT_EQ(width, 1);
    ASSERT_EQ(height, static_cast<int>(values.size()));
    ASSERT_GE(channels, 3);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto expected = static_cast<unsigned char>(std::lround(std::clamp(values[i], 0.0f, 1.0f) * 255.0f));
        for (int channel = 0; channel < 3; ++channel) {
            EXPECT_EQ(saved[i * static_cast<std::size_t>(channels) + channel], expected)
                << "index " << i << ", channel " << channel;
        }
    }
    lfs::core::free_image(saved);
    std::error_code ec;
    std::filesystem::remove(path, ec);
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

class MetricsEvaluatorGeom : public lfs::test::CudaBackendTest {};

// Catches a sample halfway between a rendered pixel and an empty one being blended toward zero depth.
TEST(GeomMetricHelpers, SampleNextToEmptyDepthIsSkipped) {
    constexpr int kH = 8;
    constexpr int kW = 8;
    std::vector<float> depth(static_cast<size_t>(kH * kW), 2.0f);
    for (int y = 0; y < kH; ++y)
        depth[static_cast<size_t>(y) * kW + 4] = 0.0f;
    const std::vector<lfs::training::DepthAbsRelSample> samples{{4.0f, 4.5f, 2.0f}, {1.5f, 4.5f, 2.0f}};

    const auto absrel = median_depth_absrel(cpu_hw(depth, kH, kW), samples);
    ASSERT_TRUE(absrel.has_value());
    EXPECT_NEAR(*absrel, 0.0f, 1.0e-6f);
}

TEST_F(MetricsEvaluatorGeom, MatchingRenderedAndPriorNormalIsNearZero) {
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
    auto background = Tensor::zeros({3}, Device::GPU);
    auto params = make_eval_params(tmp / "out");
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, splat, dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_EQ(metrics.views.size(), 1u);
    EXPECT_FALSE(metrics.views[0].validity_mask_applied);
    EXPECT_FLOAT_EQ(metrics.views[0].evaluated_pixel_fraction, 1.0f);
    ASSERT_TRUE(metrics.normal_angle_deg.has_value());
    EXPECT_NEAR(*metrics.normal_angle_deg, 0.0f, 2.0f);
    EXPECT_EQ(EvalMetrics::to_csv_header(),
              "iteration,psnr,ssim,lpips,time_per_image,num_gaussians,normal_angle_deg,depth_absrel,bias_r,bias_g,bias_b,bias_corr_r,bias_corr_g,bias_corr_b");

    std::filesystem::remove_all(tmp);
}

// The warped render is zero outside the undistorted frame; a bias averaged over the whole
// image would be pulled toward -GT by those pixels instead of reporting the 2/255 offset.
TEST(MetricsEvaluatorUndistort, BiasCountsOnlyEvaluatedPixels) {
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_undistort_eval_bias";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 130, 130, 130, kH, kW);

    auto cam = make_distorted_eval_camera(image_path, {}, kW, kH);
    ASSERT_FALSE(cam->image_size_loaded());
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam}, DatasetConfig{}, CameraDataset::Split::ALL);
    auto background = Tensor::full({3}, 128.0f / 255.0f, Device::CUDA);
    auto params = make_eval_params(tmp / "out");
    params.optimization.undistort = true;
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, make_hidden_splat(), dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_EQ(metrics.views.size(), 1u);
    EXPECT_TRUE(metrics.views[0].validity_mask_applied);
    EXPECT_GT(metrics.views[0].evaluated_pixel_fraction, 0.5f);
    EXPECT_LT(metrics.views[0].evaluated_pixel_fraction, 0.99f);
    EXPECT_NEAR(metrics.bias_r, -2.0f / 255.0f, 1.0e-4f);
    EXPECT_NEAR(metrics.bias_corr_r, -2.0f / 255.0f, 1.0e-4f);
    EXPECT_FALSE(cam->image_size_loaded());
    EXPECT_EQ(cam->image_width(), kW);
    EXPECT_EQ(cam->image_height(), kH);

    std::filesystem::remove_all(tmp);
}

// Catches an undistorted reference path that still warps the render, keeps the validity mask,
// or compares against the original source dimensions.
TEST(MetricsEvaluatorUndistort, UndistortedSpaceUsesAllUndistortedPixels) {
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_undistorted_eval_space";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 130, 130, 130, kH, kW);

    auto cam = make_distorted_eval_camera(image_path, {}, kW, kH);
    const auto scaled = lfs::core::prepare_undistort_params(
        cam->undistort_params(), kW, kH, 1, 0);
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam}, DatasetConfig{}, CameraDataset::Split::ALL);
    auto background = Tensor::full({3}, 128.0f / 255.0f, Device::CUDA);
    auto params = make_eval_params(tmp / "out");
    params.optimization.undistort = true;
    params.optimization.eval_space = lfs::core::param::EvalSpace::Undistorted;
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, make_hidden_splat(), dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_EQ(metrics.views.size(), 1u);
    const auto& view = metrics.views[0];
    EXPECT_EQ(view.width, scaled.dst_width);
    EXPECT_EQ(view.height, scaled.dst_height);
    EXPECT_FALSE(view.validity_mask_applied);
    EXPECT_FALSE(view.masked);
    EXPECT_FLOAT_EQ(view.evaluated_pixel_fraction, 1.0f);
    ASSERT_TRUE(view.ssim.has_value());
    EXPECT_NEAR(metrics.bias_r, -2.0f / 255.0f, 1.0e-4f);
    EXPECT_NEAR(metrics.bias_g, -2.0f / 255.0f, 1.0e-4f);
    EXPECT_NEAR(metrics.bias_b, -2.0f / 255.0f, 1.0e-4f);
    EXPECT_NEAR(metrics.bias_corr_r, -2.0f / 255.0f, 1.0e-4f);
    EXPECT_FALSE(cam->image_size_loaded());
    EXPECT_EQ(cam->image_width(), kW);
    EXPECT_EQ(cam->image_height(), kH);

    std::filesystem::remove_all(tmp);
}

TEST(MetricsEvaluatorUndistort, UndistortedGroundTruthEqualsTrainingLoaderImage) {
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_undistorted_eval_training_gt";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    std::vector<uint8_t> pixels(static_cast<size_t>(kW) * kH * 3);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const size_t index = (static_cast<size_t>(y) * kW + x) * 3;
            pixels[index] = static_cast<uint8_t>((3 * x + y) & 255);
            pixels[index + 1] = static_cast<uint8_t>((x + 5 * y) & 255);
            pixels[index + 2] = static_cast<uint8_t>((7 * x + 11 * y) & 255);
        }
    }
    const auto image_path = tmp / "gt.png";
    write_u8_hwc_png(image_path, pixels, kH, kW);

    auto camera = make_distorted_eval_camera(image_path, {}, kW, kH);
    auto params = make_eval_params(tmp / "out");
    params.optimization.undistort = true;
    params.optimization.eval_space = lfs::core::param::EvalSpace::Undistorted;
    params.dataset.max_width = 40;

    lfs::io::PipelinedLoaderConfig loader_config;
    loader_config.jpeg_batch_size = 1;
    loader_config.prefetch_count = 1;
    loader_config.output_queue_size = 1;
    loader_config.decode_frame_ring_capacity = 2;
    loader_config.decoder_pool_size = 1;
    loader_config.io_threads = 1;
    loader_config.cold_process_threads = 1;
    lfs::io::PipelinedImageLoader loader(loader_config);
    lfs::io::LoadParams load_params;
    load_params.resize_factor = params.dataset.resize_factor;
    load_params.max_width = params.dataset.max_width;
    load_params.output_uint8 = true;
    load_params.undistort = &camera->undistort_params();
    const auto training_image = loader.load_image_immediate(image_path, load_params);

    const auto render = [](Camera& render_camera, float)
        -> lfs::Result<lfs::training::EvaluationRenderResult> {
        const auto height = static_cast<size_t>(render_camera.image_height());
        const auto width = static_cast<size_t>(render_camera.image_width());
        lfs::training::RenderOutput output;
        output.image = Tensor::zeros({size_t{3}, height, width}, Device::CUDA);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };
    const auto prepared = prepare_evaluation_view(
        *camera, params, render, nullptr, &loader);
    ASSERT_TRUE(prepared.has_value()) << prepared.error().detail();
    ASSERT_EQ(prepared->inputs.gt_image.shape(), training_image.shape());
    ASSERT_EQ(prepared->inputs.gt_image.dtype(), training_image.dtype());
    const auto actual_cpu = prepared->inputs.gt_image.cpu().contiguous();
    const auto expected_cpu = training_image.cpu().contiguous();
    ASSERT_EQ(actual_cpu.bytes(), expected_cpu.bytes());
    EXPECT_EQ(std::memcmp(actual_cpu.data_ptr(), expected_cpu.data_ptr(), actual_cpu.bytes()), 0);

    std::filesystem::remove_all(tmp);
}

// Catches batch and interactive callers preparing different GT tensors, render geometry,
// masks, or SSIM behavior when the interactive caller reuses cached image inputs.
TEST(MetricsEvaluatorUndistort, SharedPreparationMatchesCachedInteractiveInputsInBothSpaces) {
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_shared_eval_preparation";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 130, 130, 130, kH, kW);

    for (const auto space : {lfs::core::param::EvalSpace::Distorted,
                             lfs::core::param::EvalSpace::Undistorted}) {
        auto cam = make_distorted_eval_camera(image_path, {}, kW, kH);
        const auto scaled = lfs::core::prepare_undistort_params(
            cam->undistort_params(), kW, kH, -1, 0);
        auto params = make_eval_params(tmp / "out");
        params.optimization.undistort = true;
        params.optimization.eval_space = space;

        int render_calls = 0;
        std::vector<std::pair<int, int>> render_sizes;
        std::vector<float> render_dilations;
        const auto render = [&render_calls, &render_sizes, &render_dilations](
                                Camera& render_camera, const float dilation_scale)
            -> lfs::Result<lfs::training::EvaluationRenderResult> {
            ++render_calls;
            const auto height = static_cast<size_t>(render_camera.image_height());
            const auto width = static_cast<size_t>(render_camera.image_width());
            render_sizes.emplace_back(static_cast<int>(width), static_cast<int>(height));
            render_dilations.push_back(dilation_scale);
            assert(height > 0);
            assert(width > 0);
            auto image = Tensor::full({size_t{3}, height, width}, 128.0f / 255.0f,
                                      Device::CUDA);
            lfs::training::RenderOutput output;
            output.image = image;
            output.width = static_cast<int>(width);
            output.height = static_cast<int>(height);
            return lfs::training::EvaluationRenderResult{
                .output = std::move(output),
                .raw_image = image.clone()};
        };

        auto batch = prepare_evaluation_view(*cam, params, render);
        ASSERT_TRUE(batch.has_value()) << batch.error().detail();
        auto interactive = prepare_evaluation_view(*cam, params, render, &batch->inputs);
        ASSERT_TRUE(interactive.has_value()) << interactive.error().detail();
        EXPECT_EQ(render_calls, 2);

        EXPECT_EQ(batch->inputs.gt_image.shape(), interactive->inputs.gt_image.shape());
        EXPECT_EQ(batch->output.image.shape(), interactive->output.image.shape());
        EXPECT_EQ(batch->render_geometry.width, interactive->render_geometry.width);
        EXPECT_EQ(batch->render_geometry.height, interactive->render_geometry.height);
        EXPECT_FLOAT_EQ(batch->render_geometry.fx, interactive->render_geometry.fx);
        EXPECT_FLOAT_EQ(batch->render_geometry.fy, interactive->render_geometry.fy);
        EXPECT_FLOAT_EQ(batch->render_geometry.cx, interactive->render_geometry.cx);
        EXPECT_FLOAT_EQ(batch->render_geometry.cy, interactive->render_geometry.cy);
        EXPECT_EQ(batch->render_geometry.undistorted,
                  interactive->render_geometry.undistorted);
        EXPECT_EQ(batch->validity_mask_applied, interactive->validity_mask_applied);
        EXPECT_EQ(batch->erode_ssim_mask, interactive->erode_ssim_mask);
        EXPECT_FLOAT_EQ(
            (batch->inputs.gt_image.to(DataType::Float32) -
             interactive->inputs.gt_image.to(DataType::Float32))
                .abs()
                .max()
                .item<float>(),
            0.0f);
        EXPECT_FLOAT_EQ(
            (batch->output.image - interactive->output.image).abs().max().item<float>(),
            0.0f);

        if (space == lfs::core::param::EvalSpace::Distorted) {
            EXPECT_EQ(render_sizes, (std::vector<std::pair<int, int>>{
                                        {scaled.dst_width * 2, scaled.dst_height * 2},
                                        {scaled.dst_width * 2, scaled.dst_height * 2}}));
            EXPECT_EQ(render_dilations, (std::vector<float>{4.0f, 4.0f}));
            EXPECT_EQ(batch->render_geometry.width, kW);
            EXPECT_EQ(batch->render_geometry.height, kH);
            EXPECT_FALSE(batch->render_geometry.undistorted);
            EXPECT_EQ(batch->inputs.gt_image.shape(),
                      lfs::core::TensorShape({size_t{3}, size_t{kH}, size_t{kW}}));
            EXPECT_TRUE(batch->metric_mask.is_valid());
            EXPECT_TRUE(batch->validity_mask_applied);
            EXPECT_TRUE(batch->erode_ssim_mask);
            ASSERT_TRUE(interactive->metric_mask.is_valid());
            EXPECT_FLOAT_EQ(
                (batch->metric_mask.to(DataType::Float32) -
                 interactive->metric_mask.to(DataType::Float32))
                    .abs()
                    .max()
                    .item<float>(),
                0.0f);
        } else {
            EXPECT_EQ(render_sizes, (std::vector<std::pair<int, int>>{
                                        {scaled.dst_width, scaled.dst_height},
                                        {scaled.dst_width, scaled.dst_height}}));
            EXPECT_EQ(render_dilations, (std::vector<float>{1.0f, 1.0f}));
            EXPECT_EQ(batch->render_geometry.width, scaled.dst_width);
            EXPECT_EQ(batch->render_geometry.height, scaled.dst_height);
            EXPECT_TRUE(batch->render_geometry.undistorted);
            EXPECT_EQ(batch->inputs.gt_image.shape(),
                      lfs::core::TensorShape(
                          {size_t{3}, static_cast<size_t>(scaled.dst_height),
                           static_cast<size_t>(scaled.dst_width)}));
            EXPECT_FALSE(batch->metric_mask.is_valid());
            EXPECT_FALSE(batch->validity_mask_applied);
            EXPECT_FALSE(batch->erode_ssim_mask);
        }

        EXPECT_FALSE(cam->image_size_loaded());
        EXPECT_EQ(cam->image_width(), kW);
        EXPECT_EQ(cam->image_height(), kH);
    }

    std::filesystem::remove_all(tmp);
}

// A keep-mask narrower than the 11x11 SSIM window has no complete window after erosion; the
// view must stay measured (SSIM over partial windows) instead of being dropped or reported as 0.
TEST(MetricsEvaluatorUndistort, ThinMaskFallsBackToPartialSsimWindows) {
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_undistort_eval_thin_mask";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    const auto image_path = tmp / "gt.png";
    const auto mask_path = tmp / "mask.png";
    write_rgb_png(image_path, 130, 130, 130, kH, kW);
    std::vector<uint8_t> strip(static_cast<size_t>(kH) * kW * 3, 0);
    for (int y = 0; y < kH; ++y) {
        for (int x = 28; x < 37; ++x) {
            const size_t i = (static_cast<size_t>(y) * kW + x) * 3;
            strip[i] = strip[i + 1] = strip[i + 2] = 255;
        }
    }
    write_u8_hwc_png(mask_path, strip, kH, kW);

    auto cam = make_distorted_eval_camera(image_path, mask_path, kW, kH);
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam}, DatasetConfig{}, CameraDataset::Split::ALL);
    auto background = Tensor::full({3}, 128.0f / 255.0f, Device::CUDA);
    auto params = make_eval_params(tmp / "out");
    params.optimization.undistort = true;
    params.optimization.mask_mode = lfs::core::param::MaskMode::Ignore;
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, make_hidden_splat(), dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_EQ(metrics.views.size(), 1u);
    const auto& view = metrics.views[0];
    EXPECT_TRUE(view.skipped_reason.empty()) << view.skipped_reason;
    EXPECT_TRUE(view.masked);
    ASSERT_TRUE(view.psnr.has_value());
    EXPECT_NEAR(*view.psnr, 20.0f * std::log10(255.0f / 2.0f), 0.01f);
    ASSERT_TRUE(view.ssim.has_value());
    EXPECT_GT(*view.ssim, 0.0f);
    EXPECT_FLOAT_EQ(metrics.ssim, *view.ssim);

    std::filesystem::remove_all(tmp);
}

// Catches the evaluation copy of a prepared camera losing its undistorted state, which makes the
// GUT renderer apply the lens distortion a second time.
TEST(MetricsEvaluatorUndistort, TransformCopyKeepsPreparedUndistortion) {
    const auto camera = make_distorted_eval_camera("copy.png", {}, 64, 48);
    const Camera copy(*camera, camera->world_view_transform());
    EXPECT_TRUE(copy.is_undistort_prepared());
    EXPECT_EQ(copy.undistort_params().dst_width, camera->undistort_params().dst_width);
    EXPECT_EQ(copy.undistort_params().dst_fx, camera->undistort_params().dst_fx);
}

// Catches a supersampled evaluation render whose screen-space dilation ignores the scale when the
// mip filter is off: the footprint of a subpixel splat then shrinks about threefold.
TEST(MetricsEvaluatorUndistort, SupersampledRenderKeepsTheSplatFootprint) {
    auto splat = make_front_facing_splat();
    splat.scaling_raw() = Tensor::full({1, 3}, -4.2f, Device::CUDA);
    splat.opacity_raw() = Tensor::zeros({1}, Device::CUDA);
    auto background = Tensor::zeros({3}, Device::CUDA);
    auto base = make_eval_camera("footprint.png", {}, 64, 64);
    auto supersampled = make_eval_camera("footprint.png", {}, 128, 128);
    const auto& ops = *lfs::training::training_ops(lfs::core::default_gpu_backend()).fast;
    lfs::gpu_ops::FastSaved saved{.backend = ops.create()};
    const auto alpha_area = [&](Camera& camera, const float dilation_scale) {
        return lfs::training::fast_infer(ops, saved, camera, splat, background, false, {}, false, dilation_scale)
            .alpha.sum()
            .item<float>();
    };
    const float base_area = alpha_area(*base, 1.0f);
    ASSERT_GT(base_area, 0.1f);
    EXPECT_NEAR(alpha_area(*supersampled, 4.0f) / 4.0f, base_area, 0.03f * base_area);
}

TEST_F(MetricsEvaluatorGeom, RotatedPriorReportsKnownAngle) {
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
    auto background = Tensor::zeros({3}, Device::GPU);
    auto params = make_eval_params(tmp / "out");
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, splat, dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_TRUE(metrics.normal_angle_deg.has_value());
    EXPECT_NEAR(*metrics.normal_angle_deg, kDeg, 2.0f);

    std::filesystem::remove_all(tmp);
}

TEST_F(MetricsEvaluatorGeom, SparsePointAbsRelAgainstRenderedDepth) {

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
    auto background = Tensor::zeros({3}, Device::GPU);
    auto params = make_eval_params(tmp / "out");
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, splat, dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_TRUE(metrics.depth_absrel.has_value());
    EXPECT_LT(*metrics.depth_absrel, 0.15f);

    std::filesystem::remove_all(tmp);
}

TEST(ViewEvaluationJson, AddsStepsInOrderAndReplacesARepeatedStep) {
    const lfs::training::ViewMetrics measured{
        .index = 0,
        .image_name = "a.png",
        .width = 8,
        .height = 4,
        .psnr = 24.0f,
        .ssim = 0.8f,
        .lpips = 0.2f,
        .evaluated_pixel_fraction = 0.75f,
        .validity_mask_applied = true};
    const lfs::training::ViewMetrics remeasured{
        .index = 0,
        .image_name = "a.png",
        .width = 8,
        .height = 4,
        .psnr = 26.0f,
        .ssim = 0.9f,
        .masked = true};
    const lfs::training::ViewMetrics skipped{
        .index = 0,
        .image_name = "a.png",
        .skipped_reason = "failed to load ground truth image: gone"};

    auto document = lfs::training::add_view_evaluation({}, measured, 7000, "test");
    document = lfs::training::add_view_evaluation(std::move(document), skipped, 3000, "test");
    document = lfs::training::add_view_evaluation(std::move(document), remeasured, 7000, "test");
    document = lfs::training::add_view_evaluation(std::move(document), measured, 7000, "train");

    ASSERT_EQ(document.size(), 1u);
    const auto& record = document.at("a.png");
    EXPECT_EQ(record.at("width"), 8);
    EXPECT_EQ(record.at("height"), 4);
    const auto& evaluations = record.at("evaluations");
    ASSERT_EQ(evaluations.size(), 3u);
    EXPECT_EQ(evaluations[0].at("step"), 3000);
    EXPECT_TRUE(evaluations[0].at("psnr").is_null());
    EXPECT_EQ(evaluations[0].at("skipped_reason"), "failed to load ground truth image: gone");
    EXPECT_EQ(evaluations[1].at("step"), 7000);
    EXPECT_EQ(evaluations[1].at("split"), "test");
    EXPECT_FLOAT_EQ(evaluations[1].at("psnr").get<float>(), 26.0f);
    EXPECT_TRUE(evaluations[1].at("lpips").is_null());
    EXPECT_EQ(evaluations[1].at("masked"), true);
    EXPECT_FALSE(evaluations[1].contains("skipped_reason"));
    EXPECT_EQ(evaluations[2].at("step"), 7000);
    EXPECT_EQ(evaluations[2].at("split"), "train");
    EXPECT_FLOAT_EQ(evaluations[2].at("psnr").get<float>(), 24.0f);
}

TEST(ViewEvaluationJson, OneFileKeyedByImageNameInTheOutputFolder) {
    const auto tmp = std::filesystem::temp_directory_path() / "lfs_view_json_keyed";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    lfs::training::MetricsReporter reporter(tmp);
    EXPECT_EQ(reporter.per_image_path(), tmp / "per_image_metrics.json");

    EvalMetrics metrics;
    metrics.iteration = 100;
    metrics.views = {
        {.index = 0, .image_name = "frame.png", .width = 2, .height = 2, .psnr = 30.0f, .ssim = 0.9f},
        {.index = 1, .image_name = "frame.jpg", .width = 2, .height = 2, .psnr = 20.0f, .ssim = 0.5f},
        {.index = 2, .image_name = "cam0/frame.png", .width = 2, .height = 2, .psnr = 25.0f, .ssim = 0.7f},
    };
    reporter.write_view_evaluations(metrics, "test");
    metrics.iteration = 200;
    reporter.write_view_evaluations(metrics, "test");

    std::ifstream in(tmp / "per_image_metrics.json");
    const auto document = nlohmann::json::parse(in);
    ASSERT_EQ(document.size(), 3u);
    for (const char* name : {"frame.png", "frame.jpg", "cam0/frame.png"}) {
        ASSERT_TRUE(document.contains(name)) << name;
        ASSERT_EQ(document.at(name).at("evaluations").size(), 2u) << name;
        EXPECT_EQ(document.at(name).at("evaluations")[1].at("step"), 200) << name;
    }
    EXPECT_FALSE(std::filesystem::exists(tmp / "eval"));
    EXPECT_FALSE(std::filesystem::exists(tmp / "per_image_metrics.json.tmp"));
    std::filesystem::remove_all(tmp);
}

class MetricsEvaluatorJson : public lfs::test::CudaBackendTest {};

TEST_F(MetricsEvaluatorJson, WritesTheTrainingConfigAndPerImageMetricsNextToTheCsv) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_view_json_eval";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 16;
    constexpr int kH = 12;
    const auto close_path = tmp / "close.png";
    const auto far_path = tmp / "far.png";
    write_rgb_png(close_path, 126, 126, 126, kH, kW);
    write_rgb_png(far_path, 20, 200, 60, kH, kW);
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{
            make_eval_camera(close_path, {}, kW, kH),
            make_eval_camera(far_path, {}, kW, kH)},
        DatasetConfig{}, CameraDataset::Split::ALL);
    auto splat = make_front_facing_splat();
    auto background = Tensor::zeros({3}, Device::GPU);

    auto params = make_eval_params(tmp / "out");
    params.optimization.eval_steps = {3};
    std::filesystem::create_directories(params.dataset.output_path);
    MetricsEvaluator evaluator(params);
    EXPECT_TRUE(evaluator.should_evaluate(3, 5));
    EXPECT_TRUE(evaluator.should_evaluate(5, 5));
    EXPECT_FALSE(evaluator.should_evaluate(4, 5));

    evaluator.write_training_config(params);
    const auto out_dir = params.dataset.output_path;
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream in(path);
        return nlohmann::json::parse(in);
    };
    const auto config = read(out_dir / "training_config.json");
    EXPECT_EQ(config.at("optimization").at("eval_steps"), nlohmann::json::array({3}));
    EXPECT_EQ(config.at("optimization").at("enable_eval"), true);
    EXPECT_EQ(config.at("optimization").at("eval_space"), "distorted");
    EXPECT_TRUE(config.contains("dataset"));

    const auto first = evaluator.evaluate(3, splat, dataset, background);
    const auto final_step = evaluator.evaluate(5, splat, dataset, background);
    ASSERT_TRUE(first.valid);
    ASSERT_TRUE(final_step.valid);

    EXPECT_TRUE(std::filesystem::exists(out_dir / "metrics.csv"));
    EXPECT_FALSE(std::filesystem::exists(out_dir / "eval"));
    const auto per_image = read(out_dir / "per_image_metrics.json");
    ASSERT_EQ(per_image.size(), 2u);
    const auto& close = per_image.at("close.png");
    const auto& far = per_image.at("far.png");
    for (const auto& document : {close, far}) {
        EXPECT_EQ(document.at("width"), kW);
        EXPECT_EQ(document.at("height"), kH);
        ASSERT_EQ(document.at("evaluations").size(), 2u);
        EXPECT_EQ(document.at("evaluations")[0].at("step"), 3);
        EXPECT_EQ(document.at("evaluations")[1].at("step"), 5);
        EXPECT_EQ(document.at("evaluations")[1].at("split"), "test");
        EXPECT_EQ(document.at("evaluations")[1].at("masked"), false);
    }
    EXPECT_FLOAT_EQ(close.at("evaluations")[1].at("psnr").get<float>(), *final_step.views[0].psnr);
    EXPECT_FLOAT_EQ(far.at("evaluations")[1].at("psnr").get<float>(), *final_step.views[1].psnr);
    EXPECT_GT(close.at("evaluations")[1].at("psnr").get<float>(),
              far.at("evaluations")[1].at("psnr").get<float>() + 10.0f);

    std::filesystem::remove_all(tmp);
}

TEST(ViewEvaluationJson, StepsBeyondTheLastIterationAreReported) {
    using lfs::training::unreachable_eval_steps;
    EXPECT_EQ(unreachable_eval_steps({7000, 10000, 200000}, 30000), (std::vector<size_t>{200000}));
    EXPECT_EQ(unreachable_eval_steps({7000, 30000}, 30000), (std::vector<size_t>{}));
    EXPECT_EQ(unreachable_eval_steps({7000, 30000}, 5000), (std::vector<size_t>{7000, 30000}));
}

// per_image_metrics.json is read by scripts outside LichtFeld Studio: its keys
// only change together with a documented format change.
TEST(ViewEvaluationJson, FileFormatStaysFixed) {
    const lfs::training::ViewMetrics measured{
        .index = 3,
        .image_name = "cam/frame.png",
        .width = 8,
        .height = 4,
        .psnr = 24.0f,
        .ssim = 0.8f,
        .lpips = 0.2f,
        .evaluated_pixel_fraction = 0.75f,
        .validity_mask_applied = true};
    const lfs::training::ViewMetrics skipped{
        .index = 3,
        .image_name = "cam/frame.png",
        .skipped_reason = "failed to load mask: gone"};
    auto document = lfs::training::add_view_evaluation({}, measured, 7000, "test");
    document = lfs::training::add_view_evaluation(std::move(document), skipped, 30000, "test");

    const auto keys = [](const nlohmann::json& object) {
        std::vector<std::string> names;
        for (const auto& item : object.items())
            names.push_back(item.key());
        return names;
    };
    EXPECT_EQ(keys(document), (std::vector<std::string>{"cam/frame.png"}));
    const auto& record = document.at("cam/frame.png");
    EXPECT_EQ(keys(record), (std::vector<std::string>{"evaluations", "height", "width"}));
    ASSERT_EQ(record.at("evaluations").size(), 2u);
    EXPECT_EQ(keys(record.at("evaluations")[0]),
              (std::vector<std::string>{"evaluated_pixel_fraction", "lpips", "masked", "psnr", "split", "ssim", "step", "validity_mask_applied"}));
    EXPECT_EQ(keys(record.at("evaluations")[1]),
              (std::vector<std::string>{"evaluated_pixel_fraction", "lpips", "masked", "psnr", "skipped_reason", "split", "ssim", "step", "validity_mask_applied"}));
    EXPECT_TRUE(record.at("evaluations")[0].at("step").is_number_integer());
    EXPECT_TRUE(record.at("evaluations")[0].at("psnr").is_number_float());
    EXPECT_TRUE(record.at("evaluations")[0].at("masked").is_boolean());
    EXPECT_FLOAT_EQ(record.at("evaluations")[0].at("evaluated_pixel_fraction"), 0.75f);
    EXPECT_TRUE(record.at("evaluations")[0].at("validity_mask_applied"));
    EXPECT_EQ(record.at("evaluations")[0].at("split"), "test");
    EXPECT_TRUE(record.at("evaluations")[1].at("psnr").is_null());
    EXPECT_FLOAT_EQ(record.at("evaluations")[1].at("evaluated_pixel_fraction"), 0.0f);
    EXPECT_FALSE(record.at("evaluations")[1].at("validity_mask_applied"));
}

TEST(MetricsEvaluatorVulkan, ConcurrentLpipsPreservesPerImageMetricsAndWaitsBeforeReturning) {
    if (lfs::core::default_gpu_backend() != lfs::core::GpuBackend::Vulkan ||
        !lfs::core::gpu_backend_available(lfs::core::GpuBackend::Vulkan))
        GTEST_SKIP();
    const char* home = std::getenv("HOME");
    if (!home)
        GTEST_SKIP() << "No model cache";
    const auto weights = std::filesystem::path(home) / ".lichtfeld/onnx/lpips-vgg16-v0.1.lfw";
    if (!std::filesystem::exists(weights))
        GTEST_SKIP() << "LPIPS weights unavailable";
    const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::Vulkan);
    ensure_image_loader();
    const auto tmp = std::filesystem::temp_directory_path() / "lfs_vulkan_eval_overlap";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int width = 35, height = 33;
    const auto image = tmp / "image.png", normal = tmp / "normal.png";
    write_rgb_png(image, 60, 120, 180, height, width);
    write_normal_png(normal, 0.f, 0.f, -1.f, height, width);
    auto concurrent = make_eval_camera(image, normal, width, height);
    auto serial = make_eval_camera(image, {}, width, height);
    auto missing = make_eval_camera(image, tmp / "missing.png", width, height);
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{concurrent, serial, missing},
        DatasetConfig{}, CameraDataset::Split::ALL);
    auto model = make_front_facing_splat();
    auto background = Tensor::zeros({3}, Device::GPU);
    auto params = make_eval_params(tmp / "output");
    std::filesystem::create_directories(params.dataset.output_path);
    MetricsEvaluator evaluator(params);
    evaluator.set_lpips_weights_path(weights);
    for (int repeat = 0; repeat < 2; ++repeat) {
        SCOPED_TRACE(repeat);
        const auto metrics = evaluator.evaluate(repeat + 1, model, dataset, background);
        ASSERT_TRUE(metrics.valid);
        ASSERT_EQ(metrics.views.size(), 3u);
        ASSERT_TRUE(metrics.normal_angle_deg.has_value());
        EXPECT_TRUE(std::isfinite(*metrics.normal_angle_deg));
        for (const auto& view : metrics.views) {
            ASSERT_TRUE(view.lpips.has_value());
            EXPECT_GT(*view.lpips, 0.f);
            EXPECT_EQ(view.psnr, metrics.views[1].psnr);
            EXPECT_EQ(view.ssim, metrics.views[1].ssim);
            EXPECT_NEAR(*view.lpips, *metrics.views[1].lpips, 2e-6f);
        }
    }
    std::filesystem::remove_all(tmp);
}

TEST(MetricsEvaluatorInference, PreservesScreenShareWhileTrainingStillUpdatesIt) {
    const auto& ops = *lfs::training::training_ops(lfs::core::default_gpu_backend()).fast;
    lfs::gpu_ops::FastSaved saved{.backend = ops.create()};
    auto model = make_front_facing_splat();
    model.scaling_raw() = Tensor::zeros({1, 3}, Device::GPU);
    model._max_screen_share = Tensor::zeros({1}, Device::GPU);
    auto camera = make_eval_camera("statistics.png", {}, 64, 64);
    auto background = Tensor::zeros({3}, Device::GPU);
    const auto inference = lfs::training::fast_infer(ops, saved, *camera, model, background, false);
    ASSERT_TRUE(inference.image.is_valid());
    EXPECT_FLOAT_EQ(model._max_screen_share.item<float>(), 0.0f);
    lfs::training::RenderOutput training;
    const auto result = lfs::training::fast_render(
        ops, saved, *camera, model, background, 0, 0, 0, 0, false, {}, false, true, training);
    ASSERT_EQ(result.code, lfs::gpu_ops::RasterResult::Code::Success);
    ops.release(saved);
    EXPECT_GT(model._max_screen_share.item<float>(), 0.0f);
}
