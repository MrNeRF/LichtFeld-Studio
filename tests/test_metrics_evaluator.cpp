/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/parameters.hpp"
#include "core/shared_image_ops.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "io/cache_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "lfs/training/ops/fast_services.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/dataset.hpp"
#include "training/metrics/metrics.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

using lfs::core::Camera;
using lfs::core::CameraModelType;
using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::SplatData;
using lfs::core::Tensor;
using lfs::core::param::EvalBitDepth;
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
        auto R = Tensor::eye(3, Device::GPU);
        std::vector<float> t_data{0.0f, 0.0f, 4.0f};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
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
        splat.means() = Tensor::from_vector({0.0f, 0.0f, -10.0f}, lfs::core::TensorShape({1, 3}), Device::GPU);
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
              "iteration,psnr,ssim,lpips,time_per_image,num_gaussians,normal_angle_deg,depth_absrel,bias_r,bias_g,bias_b,bias_corr_r,bias_corr_g,bias_corr_b,flip");

    EvalMetrics missing;
    missing.iteration = 200;
    missing.psnr = 1.0f;
    missing.ssim = 0.5f;
    missing.elapsed_time = 0.01f;
    missing.num_gaussians = 10;
    EXPECT_EQ(missing.to_csv_row(), "200,1.000000,0.500000,,0.010000,10,,,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,");

    EvalMetrics present = missing;
    present.normal_angle_deg = 12.5f;
    present.depth_absrel = 0.25f;
    present.bias_r = 0.001f;
    present.bias_g = -0.002f;
    present.bias_b = 0.003f;
    present.flip = 0.125f;
    EXPECT_EQ(present.to_csv_row(), "200,1.000000,0.500000,,0.010000,10,12.500000,0.250000,0.001000,-0.002000,0.003000,0.000000,0.000000,0.000000,0.125000");
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

TEST(MetricsEvaluatorTest, GpuQuantizationMatchesTensorExpressionBitwise) {
    std::vector<float> values{0.0f, 1.0f, -0.0f, -1.0f, 2.0f, std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (int level = 0; level < 256; ++level) {
        float boundary = (static_cast<float>(level) + 0.5f) / 255.0f;
        for (int step = 0; step < 1024; ++step)
            boundary = std::nextafter(boundary, 0.0f);
        for (int step = 0; step < 2048; ++step) {
            values.push_back(boundary);
            boundary = std::nextafter(boundary, 1.0f);
        }
    }
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> uniform(-0.25f, 1.25f);
    for (int i = 0; i < (1 << 20); ++i)
        values.push_back(uniform(generator));

    const auto input = Tensor::from_vector(values, {values.size()}, Device::GPU);
    const auto expression = input.clamp(0.0f, 1.0f)
                                .mul(255.0f)
                                .add(0.5f)
                                .to(DataType::UInt8)
                                .to(DataType::Float32)
                                .div(255.0f)
                                .contiguous()
                                .to_vector();
    const auto quantized = image_for_metrics_and_save(input).to_vector();
    ASSERT_EQ(quantized.size(), expression.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(quantized[i]), std::bit_cast<std::uint32_t>(expression[i]))
            << "value " << values[i];
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

// The evaluator follows the default backend, so these run once per backend.
class MetricsEvaluatorBackendTest : public ::testing::TestWithParam<lfs::core::GpuBackend> {
protected:
    void SetUp() override {
        if (!lfs::core::gpu_backend_available(GetParam()))
            GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " unavailable";
        backend_.emplace(GetParam());
        ASSERT_TRUE(backend_->switched());
    }

private:
    std::optional<lfs::test::DefaultGpuBackendForTesting> backend_;
};

std::string backend_test_name(const ::testing::TestParamInfo<lfs::core::GpuBackend>& info) {
    return lfs::core::gpu_backend_name(info.param);
}

class MetricsEvaluatorGeom : public MetricsEvaluatorBackendTest {};
INSTANTIATE_TEST_SUITE_P(Backends, MetricsEvaluatorGeom,
                         ::testing::Values(lfs::core::GpuBackend::CUDA, lfs::core::GpuBackend::Vulkan,
                                           lfs::core::GpuBackend::Metal),
                         backend_test_name);

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

TEST_P(MetricsEvaluatorGeom, MatchingRenderedAndPriorNormalIsNearZero) {
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
              "iteration,psnr,ssim,lpips,time_per_image,num_gaussians,normal_angle_deg,depth_absrel,bias_r,bias_g,bias_b,bias_corr_r,bias_corr_g,bias_corr_b,flip");

    std::filesystem::remove_all(tmp);
}

// Catches FLIP missing from the per-view record, the CSV row or the saved error map when enabled, and
// FLIP computed when it is not.
TEST(MetricsEvaluator, FlipIsReportedAndSavedOnlyWhenEnabled) {
    if (!lfs::core::gpu_backend_available(lfs::core::default_gpu_backend())) {
        GTEST_SKIP() << "Selected GPU backend not available";
    }
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_eval_flip";
    std::filesystem::remove_all(tmp);
    const auto image_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle" / "images_8" / "_DSC8679.JPG";
    const auto [width, height, channels] = lfs::core::get_image_info(image_path);
    ASSERT_GT(width, 0);
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{make_eval_camera(image_path, {}, width, height)}, DatasetConfig{},
        CameraDataset::Split::ALL);
    auto splat = make_front_facing_splat();
    auto background = Tensor::zeros({3}, Device::GPU);

    for (const bool enabled : {true, false}) {
        auto params = make_eval_params(tmp / (enabled ? "on" : "off"));
        params.optimization.eval_flip = enabled;
        params.optimization.enable_save_eval_images = true;
        std::filesystem::create_directories(params.dataset.output_path);
        MetricsEvaluator evaluator(params);
        const auto metrics = evaluator.evaluate(1, splat, dataset, background);
        ASSERT_TRUE(metrics.valid);
        ASSERT_EQ(metrics.views.size(), 1u);
        const auto flip_png = params.dataset.output_path / "eval_step_1" / "_DSC8679_flip.png";
        EXPECT_EQ(metrics.views[0].flip.has_value(), enabled);
        EXPECT_EQ(metrics.flip.has_value(), enabled);
        EXPECT_EQ(std::filesystem::exists(flip_png), enabled);
        EXPECT_EQ(metrics.to_csv_row().ends_with(","), !enabled);
        if (!enabled)
            continue;
        EXPECT_GT(*metrics.views[0].flip, 0.0f);
        EXPECT_LE(*metrics.views[0].flip, 1.0f);
        EXPECT_FLOAT_EQ(*metrics.flip, *metrics.views[0].flip);
        const auto [flip_width, flip_height, flip_channels] = lfs::core::get_image_info(flip_png);
        EXPECT_EQ(flip_width, width);
        EXPECT_EQ(flip_height, height);
    }
    std::filesystem::remove_all(tmp);
}

// Fails when a masked evaluation does not save one 3x2 image per view (as rendered, mask applied, inverted mask
// applied) and nothing else, or names its files by evaluation index instead of the camera; an evaluation without a
// mask keeps the 1x2 pair.
TEST(MetricsEvaluator, MaskedEvaluationSavesGridNamedAfterTheCamera) {
    if (!lfs::core::gpu_backend_available(lfs::core::default_gpu_backend())) {
        GTEST_SKIP() << "Selected GPU backend not available";
    }
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_eval_mask_grid";
    std::filesystem::remove_all(tmp);
    const auto image_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle" / "images_8" / "_DSC8679.JPG";
    const auto [width, height, channels] = lfs::core::get_image_info(image_path);
    ASSERT_GT(width, 0);
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{make_eval_camera(image_path, {}, width, height),
                                             make_eval_camera(image_path, {}, width, height)},
        DatasetConfig{}, CameraDataset::Split::ALL);
    auto splat = make_front_facing_splat();
    auto background = Tensor::zeros({3}, Device::GPU);

    struct Rgb {
        std::vector<uint8_t> pixels;
        int width = 0, height = 0;
        const uint8_t* at(int x, int y) const { return &pixels[(static_cast<size_t>(y) * width + x) * 3]; }
    };
    const auto read_rgb = [](const std::filesystem::path& path) {
        auto [data, image_width, image_height, image_channels] = lfs::core::load_image(path);
        EXPECT_NE(data, nullptr) << path;
        Rgb rgb{std::vector<uint8_t>(static_cast<size_t>(image_width) * image_height * 3), image_width, image_height};
        for (size_t pixel = 0; data && pixel < rgb.pixels.size() / 3; ++pixel)
            std::memcpy(&rgb.pixels[pixel * 3], data + pixel * image_channels, 3);
        lfs::core::free_image(data);
        return rgb;
    };

    for (const bool masked : {true, false}) {
        auto params = make_eval_params(tmp / (masked ? "masked" : "plain"));
        params.optimization.enable_save_eval_images = true;
        std::filesystem::create_directories(params.dataset.output_path);
        MetricsEvaluator evaluator(params);
        if (masked)
            evaluator.set_eval_mesh(lfs::training::make_evaluation_box(
                lfs::training::axis_aligned_box_corners({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}),
                false));
        ASSERT_TRUE(evaluator.evaluate(1, splat, dataset, background).valid);

        const auto dir = params.dataset.output_path / "eval_step_1";
        EXPECT_FALSE(std::filesystem::exists(dir / "0.png"));
        EXPECT_TRUE(std::filesystem::exists(dir / "_DSC8679_1.png"));
        EXPECT_FALSE(std::filesystem::exists(dir / "_DSC8679_unmasked.png"));
        EXPECT_FALSE(std::filesystem::exists(dir / "_DSC8679_metric_mask.png"));
        const auto saved = read_rgb(dir / "_DSC8679.png");
        ASSERT_EQ(saved.width, 2 * width + 4);
        ASSERT_EQ(saved.height, masked ? 3 * height + 8 : height);
        if (!masked)
            continue;

        // Every pixel shows in exactly one of the two lower rows, unchanged, and the box splits the view.
        size_t inside = 0, outside = 0, mismatches = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (const int half : {0, width + 4}) {
                    const uint8_t* rendered = saved.at(half + x, y);
                    const uint8_t* applied = saved.at(half + x, y + height + 4);
                    const uint8_t* inverted = saved.at(half + x, y + 2 * (height + 4));
                    const bool in_applied = applied[0] | applied[1] | applied[2];
                    const bool in_inverted = inverted[0] | inverted[1] | inverted[2];
                    inside += in_applied;
                    outside += in_inverted;
                    mismatches += in_applied && in_inverted;
                    for (int c = 0; c < 3; ++c)
                        mismatches += applied[c] + inverted[c] != rendered[c];
                }
            }
        }
        EXPECT_GT(inside, 0u);
        EXPECT_GT(outside, 0u);
        EXPECT_EQ(mismatches, 0u);
    }
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
    auto background = Tensor::full({3}, 128.0f / 255.0f, Device::GPU);
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
    auto background = Tensor::full({3}, 128.0f / 255.0f, Device::GPU);
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
        output.image = Tensor::zeros({size_t{3}, height, width}, Device::GPU);
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

// Catches the evaluation reference being downscaled by the host bilinear decoder while
// training images get the GPU Lanczos filter.
class MetricsEvaluatorImages : public MetricsEvaluatorBackendTest {};
INSTANTIATE_TEST_SUITE_P(Backends, MetricsEvaluatorImages,
                         ::testing::Values(lfs::core::GpuBackend::CUDA, lfs::core::GpuBackend::Vulkan,
                                           lfs::core::GpuBackend::Metal),
                         backend_test_name);

TEST_P(MetricsEvaluatorImages, DownscaledGroundTruthMatchesGpuLanczos) {
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_downscaled_eval_gt";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    constexpr int kMaxWidth = 40;
    std::vector<uint8_t> pixels(static_cast<size_t>(kW) * kH * 3);
    for (size_t index = 0; index < pixels.size(); ++index)
        pixels[index] = static_cast<uint8_t>((index * 73 + index / 11) % 256);
    const auto image_path = tmp / "gt.png";
    write_u8_hwc_png(image_path, pixels, kH, kW);

    auto camera = make_eval_camera(image_path, {}, kW, kH);
    auto params = make_eval_params(tmp / "out");
    params.dataset.max_width = kMaxWidth;

    const auto render = [](Camera& render_camera, float)
        -> lfs::Result<lfs::training::EvaluationRenderResult> {
        const auto height = static_cast<size_t>(render_camera.image_height());
        const auto width = static_cast<size_t>(render_camera.image_width());
        lfs::training::RenderOutput output;
        output.image = Tensor::zeros({size_t{3}, height, width}, Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };
    const auto prepared = prepare_evaluation_view(*camera, params, render);
    ASSERT_TRUE(prepared.has_value()) << prepared.error().detail();
    const auto& gt = prepared->inputs.gt_image;
    ASSERT_EQ(gt.dtype(), DataType::UInt8);
    ASSERT_EQ(gt.shape(), lfs::core::TensorShape({3, kH * kMaxWidth / kW, kMaxWidth}));

    const auto expected = lfs::core::shared_image_ops(GetParam())
                              ->resize(Tensor::from_blob(pixels.data(), lfs::core::TensorShape({kH, kW, 3}),
                                                         Device::CPU, DataType::UInt8)
                                           .to(Device::GPU),
                                       static_cast<int>(gt.shape()[1]), static_cast<int>(gt.shape()[2]),
                                       lfs::gpu_ops::Resample::LanczosRGB, 2);
    const auto expected_values = expected.cpu().to_vector();
    const auto actual_values = gt.cpu().to_vector_uint8();
    ASSERT_EQ(expected_values.size(), actual_values.size());
    for (size_t index = 0; index < expected_values.size(); ++index) {
        const float expected_u8 = std::clamp(expected_values[index], 0.0f, 1.0f) * 255.0f;
        EXPECT_NEAR(static_cast<float>(actual_values[index]), expected_u8, 1.0f) << index;
    }

    std::filesystem::remove_all(tmp);
}

// Catches a 16-bit reference rounded to 8 bits, a render quantized to a grid other than the reference's,
// a forced depth that does not override the file's own encoding, and a forced float depth that decodes a
// 16-bit file through 8 bits.
TEST(MetricsEvaluator, EvaluationBitDepthFollowsTheReferenceEncoding) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_eval_bit_depth";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 8;
    constexpr int kH = 4;
    std::vector<uint16_t> pixels16(static_cast<size_t>(kW) * kH * 3);
    for (size_t index = 0; index < pixels16.size(); ++index)
        pixels16[index] = static_cast<uint16_t>(1000 + 37 * index);
    const auto path16 = tmp / "gt16.png";
    ASSERT_TRUE(lfs::core::save_png(path16, pixels16.data(), kW, kH, 3, 16, 0));
    const std::vector<uint8_t> pixels8(static_cast<size_t>(kW) * kH * 3, 77);
    const auto path8 = tmp / "gt8.png";
    write_u8_hwc_png(path8, pixels8, kH, kW);

    constexpr float kRendered = 0.123456f;
    const auto render = [](Camera& render_camera, float) -> lfs::Result<lfs::training::EvaluationRenderResult> {
        lfs::training::RenderOutput output;
        output.image = Tensor::full({size_t{3}, static_cast<size_t>(render_camera.image_height()),
                                     static_cast<size_t>(render_camera.image_width())},
                                    kRendered, Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };
    const auto prepare = [&](const std::filesystem::path& path, const EvalBitDepth setting) {
        auto camera = make_eval_camera(path, {}, kW, kH);
        auto params = make_eval_params(tmp / "out");
        params.optimization.eval_bit_depth = setting;
        auto prepared = prepare_evaluation_view(*camera, params, render);
        EXPECT_TRUE(prepared.has_value()) << prepared.error().detail();
        return std::move(*prepared);
    };
    const auto rendered_value = [](const lfs::training::PreparedEvaluationView& view) {
        return view.output.image.cpu().to_vector().front();
    };

    const auto native16 = prepare(path16, EvalBitDepth::Auto);
    EXPECT_EQ(native16.inputs.bit_depth, 16);
    ASSERT_EQ(native16.inputs.gt_image.dtype(), DataType::Float32);
    const auto gt16 = native16.inputs.gt_image.cpu().to_vector();
    ASSERT_EQ(gt16.size(), pixels16.size());
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < kW * kH; ++i)
            EXPECT_NEAR(gt16[static_cast<size_t>(c) * kW * kH + i], pixels16[static_cast<size_t>(i) * 3 + c] / 65535.0f,
                        1e-6f);
    EXPECT_FLOAT_EQ(rendered_value(native16), std::round(kRendered * 65535.0f) / 65535.0f);

    const auto forced8 = prepare(path16, EvalBitDepth::Eight);
    EXPECT_EQ(forced8.inputs.bit_depth, 8);
    const auto gt8 = forced8.inputs.gt_image.cpu().to_vector();
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < kW * kH; ++i)
            EXPECT_FLOAT_EQ(gt8[static_cast<size_t>(c) * kW * kH + i] * 255.0f,
                            std::round(pixels16[static_cast<size_t>(i) * 3 + c] * 255.0f / 65535.0f));
    EXPECT_FLOAT_EQ(rendered_value(forced8), std::round(kRendered * 255.0f) / 255.0f);

    const auto float16 = prepare(path16, EvalBitDepth::Float);
    EXPECT_EQ(float16.inputs.bit_depth, 32);
    const auto gt_float = float16.inputs.gt_image.cpu().to_vector();
    ASSERT_EQ(gt_float.size(), gt16.size());
    for (size_t i = 0; i < gt_float.size(); ++i)
        EXPECT_NEAR(gt_float[i], gt16[i], 1e-6f) << i;

    const auto forced_float = prepare(path8, EvalBitDepth::Float);
    EXPECT_EQ(forced_float.inputs.bit_depth, 32);
    EXPECT_FLOAT_EQ(rendered_value(forced_float), kRendered);

    EXPECT_EQ(prepare(path8, EvalBitDepth::Auto).inputs.bit_depth, 8);
    std::filesystem::remove_all(tmp);
}

// Catches an RGBA reference that keeps the colour stored under transparent pixels, a binarised alpha, or a
// reference composited over the configured colour instead of the background the render uses.
TEST_P(MetricsEvaluatorImages, RgbaReferenceShowsTheRenderBackgroundWhereTransparent) {
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_rgba_eval_reference";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 16;
    constexpr int kH = 12;
    constexpr std::array<uint8_t, 3> kColor{200, 100, 50};
    std::vector<uint8_t> rgba(static_cast<size_t>(kW) * kH * 4);
    for (size_t pixel = 0; pixel < static_cast<size_t>(kW) * kH; ++pixel) {
        std::copy(kColor.begin(), kColor.end(), rgba.begin() + pixel * 4);
        rgba[pixel * 4 + 3] = static_cast<uint8_t>((pixel % kW) * 17);
    }
    const auto image_path = tmp / "rgba.png";
    ASSERT_TRUE(lfs::core::save_png(image_path, rgba.data(), kW, kH, 4, 8, 0));

    auto camera = make_eval_camera(image_path, {}, kW, kH);
    camera->set_has_alpha(true);
    auto params = make_eval_params(tmp / "out");
    ASSERT_EQ(params.optimization.mask_mode, lfs::core::param::MaskMode::None);
    ASSERT_TRUE(params.optimization.use_alpha_as_mask);
    params.optimization.bg_color = {0.0f, 0.0f, 0.0f};
    const std::array<float, 3> render_background{0.25f, 0.5f, 1.0f};
    const auto background = Tensor::from_vector(
        std::vector<float>(render_background.begin(), render_background.end()), {3}, Device::GPU);

    const auto render = [](Camera& render_camera, float)
        -> lfs::Result<lfs::training::EvaluationRenderResult> {
        lfs::training::RenderOutput output;
        output.image = Tensor::zeros({size_t{3}, static_cast<size_t>(render_camera.image_height()),
                                      static_cast<size_t>(render_camera.image_width())},
                                     Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };
    const auto prepared = prepare_evaluation_view(*camera, params, render, nullptr, nullptr, {}, background);
    ASSERT_TRUE(prepared.has_value()) << prepared.error().detail();
    const auto gt = prepared->inputs.gt_image.to(DataType::Float32).cpu().contiguous();
    ASSERT_EQ(gt.shape(), lfs::core::TensorShape({3, kH, kW}));
    const auto values = gt.to_vector();
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                const float alpha = static_cast<float>(x * 17) / 255.0f;
                const float expected = kColor[c] / 255.0f * alpha + render_background[c] * (1.0f - alpha);
                EXPECT_NEAR(values[(static_cast<size_t>(c) * kH + y) * kW + x], expected, 1e-5f)
                    << "c=" << c << " x=" << x << " y=" << y;
            }

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
                                      Device::GPU);
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

class MetricsEvaluatorStreams : public lfs::test::CudaBackendTest {};

// Catches evaluation reading render outputs on the caller's stream before the render stream has
// produced them: the render waits behind a gate while the evaluation work is queued.
TEST_F(MetricsEvaluatorStreams, WaitsForRenderOutputsFromAnotherStream) {
    ensure_image_loader();
    const auto tmp = std::filesystem::temp_directory_path() / "lfs_eval_render_stream";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    const auto image_path = tmp / "gt.png";
    const auto mask_path = tmp / "mask.png";
    write_rgb_png(image_path, 130, 130, 130, kH, kW);
    write_rgb_png(mask_path, 255, 255, 255, kH, kW);
    auto cam = make_distorted_eval_camera(image_path, mask_path, kW, kH);
    auto params = make_eval_params(tmp / "out");
    params.optimization.undistort = true;
    params.optimization.mask_mode = lfs::core::param::MaskMode::Ignore;

    cudaStream_t render_stream = nullptr;
    cudaStream_t gate_holder = nullptr;
    cudaEvent_t gate = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&render_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&gate_holder, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming), cudaSuccess);
    std::atomic<bool> released{false};
    bool gated = false;
    const auto render = [&](Camera& render_camera, float) -> lfs::Result<lfs::training::EvaluationRenderResult> {
        if (gated) {
            EXPECT_EQ(cudaLaunchHostFunc(
                          gate_holder,
                          [](void* flag) {
                              while (!static_cast<std::atomic<bool>*>(flag)->load(std::memory_order_acquire)) {
                              }
                          },
                          &released),
                      cudaSuccess);
            EXPECT_EQ(cudaEventRecord(gate, gate_holder), cudaSuccess);
            EXPECT_EQ(cudaStreamWaitEvent(render_stream, gate, 0), cudaSuccess);
        }
        const lfs::core::CUDAStreamGuard guard(render_stream);
        lfs::training::RenderOutput output;
        output.image = Tensor::full({size_t{3}, static_cast<size_t>(render_camera.image_height()),
                                     static_cast<size_t>(render_camera.image_width())},
                                    130.0f / 255.0f, Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };

    const auto ungated = prepare_evaluation_view(*cam, params, render);
    ASSERT_TRUE(ungated.has_value()) << ungated.error().detail();
    const auto expected = ungated->metric_mask.to(lfs::core::DataType::Float32).sum().item<float>();
    ASSERT_GT(expected, 0.0f);

    gated = true;
    std::thread releaser([&released] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        released.store(true, std::memory_order_release);
    });
    const auto prepared = prepare_evaluation_view(*cam, params, render);
    const auto actual = prepared ? prepared->metric_mask.to(lfs::core::DataType::Float32).sum().item<float>() : -1.0f;
    releaser.join();
    ASSERT_TRUE(prepared.has_value()) << prepared.error().detail();
    EXPECT_EQ(actual, expected);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaEventDestroy(gate);
    cudaStreamDestroy(gate_holder);
    lfs::core::release_cuda_stream(render_stream);
    cudaStreamDestroy(render_stream);
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
    auto background = Tensor::full({3}, 128.0f / 255.0f, Device::GPU);
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

// Catches geometric masks projected through a pinhole when GUT renders a distorted camera natively,
// without --undistort: they must select the same pixels as the source-lens path of --undistort.
// Catches the splat mask scoring anything but what the splat covers when rendered with all its information:
// the mask must equal the pixels whose rendered opacity reaches the cutoff, a higher cutoff a part of it, and the
// inverse the rest.
TEST(MetricsEvaluator, SplatMaskSelectsTheRenderedCoverage) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    ensure_image_loader();
    const auto tmp = std::filesystem::temp_directory_path() / "lfs_splat_mask";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 130, 130, 130, kH, kW);
    auto camera = make_eval_camera(image_path, {}, kW, kH);
    // An elongated splat smaller than the view, so the mask has both covered and uncovered pixels.
    const float opaque = std::log(0.99f / 0.01f);
    lfs::training::EvaluationSplat splat{
        .model = SplatData(0, Tensor::from_vector({0.3f, -0.2f, 1.0f}, {1, 3}, Device::GPU),
                           Tensor::zeros({1, 1, 3}, Device::GPU), Tensor::zeros({1, 0, 3}, Device::GPU),
                           Tensor::from_vector({-0.4f, -1.2f, -3.0f}, {1, 3}, Device::GPU),
                           Tensor::from_vector({0.92f, 0.0f, 0.0f, 0.39f}, {1, 4}, Device::GPU),
                           Tensor::full({1}, opaque, Device::GPU), 1.0f)};
    const auto render = [](Camera& render_camera, float) -> lfs::Result<lfs::training::EvaluationRenderResult> {
        lfs::training::RenderOutput output;
        output.image = Tensor::zeros({size_t{3}, static_cast<size_t>(render_camera.image_height()),
                                      static_cast<size_t>(render_camera.image_width())},
                                     Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };
    const auto params = make_eval_params(tmp / "out");
    const auto mask_of = [&] {
        const auto prepared = prepare_evaluation_view(*camera, params, render, nullptr, nullptr, {.splat = &splat});
        EXPECT_TRUE(prepared.has_value()) << prepared.error().detail();
        return prepared->metric_mask.cpu().to_vector_uint8();
    };

    auto background = Tensor::zeros({3}, Device::GPU);
    const auto& ops = *lfs::training::training_ops(lfs::core::default_gpu_backend()).fast;
    lfs::gpu_ops::FastSaved saved{.backend = ops.create()};
    const auto alpha = lfs::training::fast_infer(ops, saved, *camera, splat.model, background, params.optimization.mip_filter)
                           .alpha.cpu()
                           .to_vector();
    ASSERT_EQ(alpha.size(), static_cast<size_t>(kW * kH));
    const auto expected_at = [&](const float cutoff) {
        std::vector<uint8_t> expected(alpha.size());
        std::ranges::transform(alpha, expected.begin(), [cutoff](const float value) { return value >= cutoff ? 1 : 0; });
        return expected;
    };
    splat.opacity = 0.5f;
    const auto wide = expected_at(0.5f);
    EXPECT_EQ(mask_of(), wide);
    splat.opacity = 0.85f;
    const auto expected = expected_at(0.85f);
    const auto covered = std::ranges::count(expected, uint8_t{1});
    ASSERT_GT(covered, 0);
    ASSERT_LT(covered, std::ranges::count(wide, uint8_t{1}));
    EXPECT_EQ(mask_of(), expected);
    splat.invert = true;
    const auto inverted = mask_of();
    for (size_t i = 0; i < expected.size(); ++i)
        ASSERT_EQ(inverted[i], 1 - expected[i]) << i;
    std::filesystem::remove_all(tmp);
}

TEST(MetricsEvaluatorUndistort, PointMaskFollowsTheNativeGutLens) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    ensure_image_loader();
    const auto tmp = std::filesystem::temp_directory_path() / "lfs_native_gut_point_mask";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 64;
    constexpr int kH = 48;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 130, 130, 130, kH, kW);

    const lfs::training::EvaluationPoints points{
        .means = Tensor::from_vector({-1.4f, -0.9f, 0.0f, 1.4f, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f},
                                     lfs::core::TensorShape({3, 3}), Device::GPU),
        .radius = 1,
        .close = 0};
    const auto render = [](Camera& render_camera, float) -> lfs::Result<lfs::training::EvaluationRenderResult> {
        lfs::training::RenderOutput output;
        output.image = Tensor::zeros({size_t{3}, static_cast<size_t>(render_camera.image_height()),
                                      static_cast<size_t>(render_camera.image_width())},
                                     Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };
    const auto mask_of = [&](Camera& camera, const bool gut, const bool undistort) {
        auto params = make_eval_params(tmp / "out");
        params.optimization.gut = gut;
        params.optimization.undistort = undistort;
        const auto prepared = prepare_evaluation_view(camera, params, render, nullptr, nullptr, {.points = &points});
        EXPECT_TRUE(prepared.has_value()) << prepared.error().detail();
        return prepared->metric_mask.cpu().to_vector_uint8();
    };

    auto prepared_camera = make_distorted_eval_camera(image_path, {}, kW, kH);
    const auto source_lens = mask_of(*prepared_camera, false, true);
    Camera native(Tensor::eye(3, Device::GPU), prepared_camera->T(), static_cast<float>(kW), static_cast<float>(kW),
                  0.5f * kW, 0.5f * kH, Tensor::from_vector({-0.3f}, lfs::core::TensorShape({1}), Device::CPU),
                  Tensor(), CameraModelType::PINHOLE, "gt.png", image_path, {}, kW, kH, 0);
    const auto native_lens = mask_of(native, true, false);
    EXPECT_EQ(native_lens, source_lens);
    EXPECT_NE(mask_of(native, false, false), source_lens) << "the lens no longer moves these points";
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
    splat.scaling_raw() = Tensor::full({1, 3}, -4.2f, Device::GPU);
    splat.opacity_raw() = Tensor::zeros({1}, Device::GPU);
    auto background = Tensor::zeros({3}, Device::GPU);
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

// Catches a mesh mask that is not projected through the evaluation camera, ignores the invert
// flag, or lets a view without any covered pixel through.
TEST_P(MetricsEvaluatorGeom, CoverageSelectsTheEvaluatedPixels) {
    ensure_image_loader();
    const auto tmp = std::filesystem::temp_directory_path() / "lfs_eval_mesh_mask";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 32;
    constexpr int kH = 24;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 128, 128, 128, kH, kW);
    const auto camera = make_eval_camera(image_path, {}, kW, kH);
    const auto params = make_eval_params(tmp / "out");
    const auto render = [](Camera& render_camera, float) -> lfs::Result<lfs::training::EvaluationRenderResult> {
        lfs::training::RenderOutput output;
        output.image = Tensor::zeros({size_t{3}, static_cast<size_t>(render_camera.image_height()),
                                      static_cast<size_t>(render_camera.image_width())},
                                     Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };
    // The camera sits at z = -4 looking down +z; the quad covers the left half of the view.
    lfs::training::EvaluationMesh mesh{
        .vertices = Tensor::from_vector({-2.0f, -2.4f, 0.0f, 0.0f, -2.4f, 0.0f, 0.0f, 2.4f, 0.0f, -2.0f, 2.4f, 0.0f},
                                        lfs::core::TensorShape({4, 3}), Device::GPU),
        .indices = Tensor::from_vector(std::vector<int32_t>{0, 1, 2, 0, 2, 3},
                                       lfs::core::TensorShape({2, 3}), Device::GPU),
        .z_near = 1.0e-3f};

    for (const bool invert : {false, true}) {
        mesh.invert = invert;
        const auto prepared = prepare_evaluation_view(*camera, params, render, nullptr, nullptr, {.mesh = &mesh});
        ASSERT_TRUE(prepared.has_value());
        EXPECT_TRUE(prepared->erode_ssim_mask);
        const auto mask = prepared->metric_mask.cpu().contiguous();
        ASSERT_EQ(mask.shape(), (lfs::core::TensorShape({kH, kW})));
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x)
                EXPECT_EQ(mask.ptr<uint8_t>()[y * kW + x], (x < kW / 2) != invert) << x << ',' << y;
        }
    }

    mesh.invert = false;
    mesh.vertices = mesh.vertices - Tensor::from_vector({0.0f, 0.0f, 10.0f}, lfs::core::TensorShape({1, 3}),
                                                        Device::GPU);
    EXPECT_FALSE(prepare_evaluation_view(*camera, params, render, nullptr, nullptr, {.mesh = &mesh}).has_value());
    std::filesystem::remove_all(tmp);
}

TEST_P(MetricsEvaluatorGeom, RotatedPriorReportsKnownAngle) {
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
    for (const bool gut : {false, true}) {
        SCOPED_TRACE(gut ? "GUT" : "FastGS");
        auto params = make_eval_params(tmp / "out");
        params.optimization.gut = gut;
        std::filesystem::create_directories(params.dataset.output_path);

        MetricsEvaluator evaluator(params);
        const auto metrics = evaluator.evaluate(1, splat, dataset, background);
        ASSERT_TRUE(metrics.valid);
        ASSERT_TRUE(metrics.normal_angle_deg.has_value());
        EXPECT_NEAR(*metrics.normal_angle_deg, kDeg, 2.0f);
    }

    std::filesystem::remove_all(tmp);
}

TEST_P(MetricsEvaluatorGeom, SparsePointAbsRelAgainstRenderedDepth) {

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
    for (const bool gut : {false, true}) {
        SCOPED_TRACE(gut ? "GUT" : "FastGS");
        auto params = make_eval_params(tmp / "out");
        params.optimization.gut = gut;
        std::filesystem::create_directories(params.dataset.output_path);

        MetricsEvaluator evaluator(params);
        const auto metrics = evaluator.evaluate(1, splat, dataset, background);
        ASSERT_TRUE(metrics.valid);
        ASSERT_TRUE(metrics.depth_absrel.has_value());
        EXPECT_LT(*metrics.depth_absrel, 0.15f);
    }

    std::filesystem::remove_all(tmp);
}

// GUT renders radial depth for native fisheye views; scoring it against camera Z
// would report ~0.27 for this off-axis splat instead of ~0.
TEST_P(MetricsEvaluatorGeom, NativeFisheyeGutAbsRelUsesRadialDepth) {

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_geom_metrics_fisheye_depth";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kSize = 64;
    constexpr float kFocal = 32.0f;
    const auto image_path = tmp / "gt.png";
    write_rgb_png(image_path, 128, 128, 128, kSize, kSize);

    auto R = Tensor::eye(3, Device::GPU);
    std::vector<float> t_data{0.0f, 0.0f, 4.0f};
    auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
    auto cam = std::make_shared<Camera>(
        R, T, kFocal, kFocal, 0.5f * kSize, 0.5f * kSize,
        Tensor(), Tensor(), CameraModelType::FISHEYE,
        image_path.filename().string(), image_path, std::filesystem::path{},
        kSize, kSize, 0);
    const float theta = std::atan2(4.0f, 5.0f);
    cam->set_sfm_observations({Camera::SfmObservation{
        .u = 0.5f * kSize + kFocal * theta,
        .v = 0.5f * kSize,
        .x = 4.0f,
        .y = 0.0f,
        .z = 1.0f}});
    auto dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam}, DatasetConfig{}, CameraDataset::Split::ALL);

    std::vector<float> means_data{4.0f, 0.0f, 1.0f};
    std::vector<float> scaling_data{0.0f, 0.0f, -3.0f};
    std::vector<float> rotation_data{1.0f, 0.0f, 0.0f, 0.0f};
    auto splat = SplatData(
        0,
        Tensor::from_blob(means_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::GPU),
        Tensor::zeros({1, 1, 3}, Device::GPU),
        Tensor::zeros({1, 0, 3}, Device::GPU),
        Tensor::from_blob(scaling_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::GPU),
        Tensor::from_blob(rotation_data.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::GPU),
        Tensor::full({1}, std::log(0.99f / 0.01f), Device::GPU),
        1.0f);
    auto background = Tensor::zeros({3}, Device::GPU);
    auto params = make_eval_params(tmp / "out");
    params.optimization.gut = true;
    std::filesystem::create_directories(params.dataset.output_path);

    MetricsEvaluator evaluator(params);
    const auto metrics = evaluator.evaluate(1, splat, dataset, background);
    ASSERT_TRUE(metrics.valid);
    ASSERT_TRUE(metrics.depth_absrel.has_value());
    EXPECT_LT(*metrics.depth_absrel, 0.05f);

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
        .flip = 0.1f,
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
              (std::vector<std::string>{"bit_depth", "evaluated_pixel_fraction", "flip", "lpips", "masked", "psnr", "split", "ssim", "step", "validity_mask_applied"}));
    EXPECT_EQ(keys(record.at("evaluations")[1]),
              (std::vector<std::string>{"bit_depth", "evaluated_pixel_fraction", "lpips", "masked", "psnr", "skipped_reason", "split", "ssim", "step", "validity_mask_applied"}));
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

// Catches a depth mask that compares accumulated instead of expected depth, keeps faint pixels,
// excludes the range bounds, or loses inversion.
TEST(MetricsEvaluator, DepthMaskSelectsSolidPixelsInsideTheRange) {
    if (!lfs::core::gpu_backend_available(lfs::core::default_gpu_backend())) {
        GTEST_SKIP() << "CUDA not available";
    }
    ensure_image_loader();

    const auto tmp = std::filesystem::temp_directory_path() / "lfs_depth_eval_mask";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    constexpr int kW = 6;
    constexpr int kH = 4;
    write_u8_hwc_png(tmp / "gt.png", std::vector<uint8_t>(static_cast<size_t>(kW) * kH * 3, 128), kH, kW);
    auto camera = make_eval_camera(tmp / "gt.png", {}, kW, kH);
    auto params = make_eval_params(tmp / "out");
    params.optimization.eval_mask = "depth:2,4";

    const std::array<float, kW> alpha_row{1.0f, 0.6f, 0.3f, 1.0f, 1.0f, 0.8f};
    const std::array<float, kW> accumulated_row{3.0f, 1.8f, 0.9f, 5.0f, 2.0f, 4.0f};
    const std::array<uint8_t, kW> expected_row{1, 1, 0, 0, 1, 0};
    std::vector<float> alpha, accumulated;
    for (int y = 0; y < kH; ++y) {
        alpha.insert(alpha.end(), alpha_row.begin(), alpha_row.end());
        accumulated.insert(accumulated.end(), accumulated_row.begin(), accumulated_row.end());
    }
    const auto render = [&](Camera&, float) -> lfs::Result<lfs::training::EvaluationRenderResult> {
        lfs::training::RenderOutput output;
        output.image = Tensor::zeros({size_t{3}, size_t{kH}, size_t{kW}}, Device::GPU);
        output.alpha = Tensor::from_vector(alpha, {size_t{1}, size_t{kH}, size_t{kW}}, Device::GPU);
        output.depth = Tensor::from_vector(accumulated, {size_t{1}, size_t{kH}, size_t{kW}}, Device::GPU);
        return lfs::training::EvaluationRenderResult{.output = std::move(output)};
    };

    for (const bool gut : {false, true}) {
        params.optimization.gut = gut;
        for (const bool invert : {false, true}) {
            params.optimization.eval_mask_invert = invert;
            const auto prepared = prepare_evaluation_view(*camera, params, render);
            ASSERT_TRUE(prepared.has_value()) << prepared.error().detail();
            const auto mask = prepared->metric_mask.cpu().contiguous().to_vector_uint8();
            ASSERT_EQ(mask.size(), static_cast<size_t>(kW) * kH);
            for (size_t i = 0; i < mask.size(); ++i)
                EXPECT_EQ(mask[i], invert ? 1 - expected_row[i % kW] : expected_row[i % kW])
                    << i << " invert " << invert << " gut " << gut;
        }
    }
    std::filesystem::remove_all(tmp);
}
