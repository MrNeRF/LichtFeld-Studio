/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "core/tensor_upload.hpp"
#include "cuda_backend_test.hpp"
#include "io/cache_image_loader.hpp"
#include "training/kernels/mask_preprocess.hpp"
#include "training/metrics/eval_mask.hpp"
#include "training/metrics/metrics.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

using lfs::core::Camera;
using lfs::core::CameraModelType;
using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;
using lfs::core::param::MaskMode;
using lfs::training::classify_keep_mask_for_metrics;
using lfs::training::load_alpha_masked_metrics_inputs;
using lfs::training::load_eval_mask;
using lfs::training::load_external_mask_for_metrics;
using lfs::training::LoadedMetricsMask;
using lfs::training::MetricsEvaluator;
using lfs::training::MetricsMaskLoadConfig;

namespace {

    constexpr std::array<uint8_t, 6> kBandBytes{127, 128, 200, 250, 251, 255};
    constexpr std::array<uint8_t, 6> kExpectedKeep{0, 0, 0, 0, 1, 1};
    constexpr int kBandW = 3;
    constexpr int kBandH = 2;

    class UniqueTempDir {
    public:
        explicit UniqueTempDir(const std::string_view prefix) {
            static std::atomic<uint64_t> seq{0};
            path_ = std::filesystem::temp_directory_path() /
                    std::format("{}_{}_{}", prefix,
                                std::chrono::steady_clock::now().time_since_epoch().count(),
                                seq.fetch_add(1, std::memory_order_relaxed));
            std::filesystem::create_directories(path_);
        }

        ~UniqueTempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
        }

        UniqueTempDir(const UniqueTempDir&) = delete;
        UniqueTempDir& operator=(const UniqueTempDir&) = delete;

        const std::filesystem::path& path() const { return path_; }

    private:
        std::filesystem::path path_;
    };

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

    MetricsMaskLoadConfig sai_config() {
        MetricsMaskLoadConfig cfg;
        cfg.mask_mode = MaskMode::SegmentAndIgnore;
        cfg.mask_threshold = 0.5f;
        cfg.invert_masks = false;
        cfg.resize_factor = -1;
        cfg.max_width = 0;
        return cfg;
    }

    std::shared_ptr<Camera> make_camera(const std::filesystem::path& image_path,
                                        const std::filesystem::path& mask_path,
                                        const int width,
                                        const int height,
                                        const Tensor& radial = Tensor()) {
        auto R = Tensor::eye(3, Device::CPU);
        auto T = Tensor::from_vector({0.0f, 0.0f, 1.0f}, {3}, Device::CPU);
        const float fx = static_cast<float>(width);
        const float fy = static_cast<float>(height);
        const float cx = 0.5f * static_cast<float>(width);
        const float cy = 0.5f * static_cast<float>(height);
        return std::make_shared<Camera>(
            R, T, fx, fy, cx, cy,
            radial, Tensor(), CameraModelType::PINHOLE,
            image_path.filename().string(), image_path, mask_path,
            width, height, 0);
    }

    void write_gray_png(const std::filesystem::path& path,
                        const std::vector<uint8_t>& hw,
                        const int height,
                        const int width) {
        ASSERT_EQ(hw.size(), static_cast<size_t>(height) * static_cast<size_t>(width));
        ASSERT_TRUE(lfs::core::save_png(path, hw.data(), width, height, 1, 8, 0));
    }

    void write_rgb_png(const std::filesystem::path& path,
                       const int height,
                       const int width,
                       const uint8_t value = 128) {
        std::vector<uint8_t> hwc(static_cast<size_t>(height) * static_cast<size_t>(width) * 3, value);
        ASSERT_TRUE(lfs::core::save_png(path, hwc.data(), width, height, 3, 8, 0));
    }

    void write_rgba_png(const std::filesystem::path& path,
                        const std::vector<uint8_t>& alpha_hw,
                        const int height,
                        const int width) {
        ASSERT_EQ(alpha_hw.size(), static_cast<size_t>(height) * static_cast<size_t>(width));
        std::vector<uint8_t> rgba(static_cast<size_t>(height) * static_cast<size_t>(width) * 4);
        for (size_t i = 0; i < alpha_hw.size(); ++i) {
            rgba[i * 4 + 0] = 64;
            rgba[i * 4 + 1] = 96;
            rgba[i * 4 + 2] = 160;
            rgba[i * 4 + 3] = alpha_hw[i];
        }
        ASSERT_TRUE(lfs::core::save_png(path, rgba.data(), width, height, 4, 8, 0));
    }

    std::vector<uint8_t> mask_bytes(const Tensor& mask) {
        EXPECT_TRUE(mask.is_valid());
        EXPECT_EQ(mask.dtype(), DataType::UInt8);
        auto cpu = mask.cpu().contiguous();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return cpu.to_vector_uint8();
    }

    void expect_keep(const Tensor& mask, const std::vector<uint8_t>& expected, const char* where) {
        const auto got = mask_bytes(mask);
        ASSERT_EQ(got.size(), expected.size()) << where;
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(got[i], expected[i]) << where << " pixel " << i
                                           << " authored=" << static_cast<int>(kBandBytes[i % kBandBytes.size()]);
        }
    }

    Tensor uint8_hw(const std::vector<uint8_t>& hw, const int height, const int width) {
        auto t = Tensor::empty(
            {static_cast<size_t>(height), static_cast<size_t>(width)},
            Device::CPU, DataType::UInt8);
        std::copy(hw.begin(), hw.end(), t.ptr<uint8_t>());
        return t;
    }

} // namespace

class MetricsEvalMask : public lfs::test::CudaBackendTest {};

TEST_F(MetricsEvalMask, ClassifyKeepBandMatchesTrainingContract) {
    auto u8 = uint8_hw({kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW).to(Device::CUDA);
    auto f32 = u8.to(DataType::Float32) / 255.0f;
    expect_keep(classify_keep_mask_for_metrics(u8), {kExpectedKeep.begin(), kExpectedKeep.end()}, "u8");
    expect_keep(classify_keep_mask_for_metrics(f32), {kExpectedKeep.begin(), kExpectedKeep.end()}, "f32");
    EXPECT_GT(lfs::training::kernels::kMaskKeepMin, 250.0f / 255.0f);
    EXPECT_LT(lfs::training::kernels::kMaskKeepMin, 251.0f / 255.0f);
}

TEST_F(MetricsEvalMask, PngSidecarThroughEvalAndInteractiveLoaders) {
    ensure_image_loader();
    UniqueTempDir tmp("lfs_eval_mask_sidecar");
    const auto image_path = tmp.path() / "gt.png";
    const auto mask_path = tmp.path() / "mask.png";
    write_rgb_png(image_path, kBandH, kBandW);
    write_gray_png(mask_path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);

    auto cam = make_camera(image_path, mask_path, kBandW, kBandH);
    ASSERT_TRUE(cam->has_mask());
    const auto cfg = sai_config();
    Tensor gt;

    auto eval_mask = load_eval_mask(cam.get(), gt, false, cfg);
    expect_keep(eval_mask, {kExpectedKeep.begin(), kExpectedKeep.end()}, "load_eval_mask sidecar");

    auto interactive = load_external_mask_for_metrics(*cam, cfg);
    ASSERT_TRUE(interactive.has_value()) << interactive.error();
    expect_keep(*interactive, {kExpectedKeep.begin(), kExpectedKeep.end()},
                "load_external_mask_for_metrics");
}

TEST_F(MetricsEvalMask, RgbaAlphaThroughEvalAndInteractiveLoaders) {
    UniqueTempDir tmp("lfs_eval_mask_alpha");
    const auto image_path = tmp.path() / "rgba.png";
    write_rgba_png(image_path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);

    auto cam = make_camera(image_path, {}, kBandW, kBandH);
    cam->set_has_alpha(true);
    ASSERT_FALSE(cam->has_mask());
    const auto cfg = sai_config();
    Tensor gt;

    auto eval_mask = load_eval_mask(cam.get(), gt, true, cfg);
    expect_keep(eval_mask, {kExpectedKeep.begin(), kExpectedKeep.end()}, "load_eval_mask alpha");
    ASSERT_TRUE(gt.is_valid());
    EXPECT_EQ(gt.dtype(), DataType::UInt8);

    auto interactive = load_alpha_masked_metrics_inputs(*cam, cfg);
    ASSERT_TRUE(interactive.has_value()) << interactive.error();
    expect_keep(interactive->mask, {kExpectedKeep.begin(), kExpectedKeep.end()},
                "load_alpha_masked_metrics_inputs");
}

TEST_F(MetricsEvalMask, DefaultThresholdDoesNotPromoteSegmentBand) {
    ensure_image_loader();
    UniqueTempDir tmp("lfs_eval_mask_threshold");
    const auto image_path = tmp.path() / "gt.png";
    const auto mask_path = tmp.path() / "mask.png";
    write_rgb_png(image_path, kBandH, kBandW);
    write_gray_png(mask_path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);

    auto cam = make_camera(image_path, mask_path, kBandW, kBandH);
    auto cfg = sai_config();
    ASSERT_FLOAT_EQ(cfg.mask_threshold, 0.5f);
    Tensor gt;
    auto mask = load_eval_mask(cam.get(), gt, false, cfg);
    // 128 and 200 are >= 0.5 in [0,1] but are segment, not keep.
    const auto got = mask_bytes(mask);
    ASSERT_EQ(got.size(), 6u);
    EXPECT_EQ(got[1], 0) << "byte 128 must not be keep";
    EXPECT_EQ(got[2], 0) << "byte 200 must not be keep";
    EXPECT_EQ(got[3], 0) << "byte 250 must not be keep";
    EXPECT_EQ(got[4], 1) << "byte 251 must be keep";
}

TEST_F(MetricsEvalMask, InvertHappensBeforeKeepClassification) {
    ensure_image_loader();
    UniqueTempDir tmp("lfs_eval_mask_invert");
    const auto image_path = tmp.path() / "gt.png";
    const auto mask_path = tmp.path() / "mask.png";
    write_rgb_png(image_path, kBandH, kBandW);
    write_gray_png(mask_path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);

    auto cam = make_camera(image_path, mask_path, kBandW, kBandH);
    auto cfg = sai_config();
    cfg.invert_masks = true;
    Tensor gt;
    auto eval_mask = load_eval_mask(cam.get(), gt, false, cfg);
    // 255-v for {127,128,200,250,251,255} is {128,127,55,5,4,0}: none > 250.
    expect_keep(eval_mask, {0, 0, 0, 0, 0, 0}, "inverted sidecar");

    auto interactive = load_external_mask_for_metrics(*cam, cfg);
    ASSERT_TRUE(interactive.has_value()) << interactive.error();
    expect_keep(*interactive, {0, 0, 0, 0, 0, 0}, "inverted interactive sidecar");

    UniqueTempDir tmp_alpha("lfs_eval_mask_invert_alpha");
    const auto rgba_path = tmp_alpha.path() / "rgba.png";
    write_rgba_png(rgba_path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);
    auto alpha_cam = make_camera(rgba_path, {}, kBandW, kBandH);
    alpha_cam->set_has_alpha(true);
    auto alpha = load_alpha_masked_metrics_inputs(*alpha_cam, cfg);
    ASSERT_TRUE(alpha.has_value()) << alpha.error();
    expect_keep(alpha->mask, {0, 0, 0, 0, 0, 0}, "inverted alpha");
}

TEST_F(MetricsEvalMask, SidecarWinsOverAlpha) {
    ensure_image_loader();
    UniqueTempDir tmp("lfs_eval_mask_precedence");
    const auto image_path = tmp.path() / "rgba.png";
    const auto mask_path = tmp.path() / "mask.png";
    // Alpha would keep every pixel (255); sidecar is the band fixture.
    write_rgba_png(image_path, std::vector<uint8_t>(6, 255), kBandH, kBandW);
    write_gray_png(mask_path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);

    auto cam = make_camera(image_path, mask_path, kBandW, kBandH);
    cam->set_has_alpha(true);
    ASSERT_TRUE(cam->has_mask());
    Tensor gt;
    auto mask = load_eval_mask(cam.get(), gt, true, sai_config());
    expect_keep(mask, {kExpectedKeep.begin(), kExpectedKeep.end()}, "sidecar precedence");
}

TEST_F(MetricsEvalMask, InMemoryMaskAndCacheKeying) {
    UniqueTempDir tmp("lfs_eval_mask_inmemory");
    const auto image_path = tmp.path() / "gt.png";
    write_rgb_png(image_path, kBandH, kBandW);
    auto cam = make_camera(image_path, {}, kBandW, kBandH);
    cam->set_mask_tensor(uint8_hw({kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW));
    ASSERT_TRUE(cam->has_in_memory_mask());
    ASSERT_TRUE(cam->has_mask());

    // A 0.5 binarize must not poison a later SegmentAndIgnore classify.
    auto poisoned = cam->load_and_get_mask(-1, 0, false, 0.5f, true);
    ASSERT_TRUE(poisoned.is_valid());
    const auto binary = mask_bytes(poisoned);
    ASSERT_EQ(binary.size(), 6u);
    EXPECT_EQ(binary[1], 1) << "byte 128 is keep under binary 0.5";
    EXPECT_EQ(binary[2], 1) << "byte 200 is keep under binary 0.5";

    const auto cfg = sai_config();
    Tensor gt;
    auto eval_mask = load_eval_mask(cam.get(), gt, false, cfg);
    expect_keep(eval_mask, {kExpectedKeep.begin(), kExpectedKeep.end()},
                "in-memory load_eval_mask after binary cache");

    auto interactive = load_external_mask_for_metrics(*cam, cfg);
    ASSERT_TRUE(interactive.has_value()) << interactive.error();
    expect_keep(*interactive, {kExpectedKeep.begin(), kExpectedKeep.end()},
                "in-memory load_external_mask_for_metrics after binary cache");
}

TEST_F(MetricsEvalMask, UndistortClassifiesFloatKeepBand) {
    ensure_image_loader();
    UniqueTempDir tmp("lfs_eval_mask_undistort");
    constexpr int kW = 32;
    constexpr int kH = 32;
    const auto image_path = tmp.path() / "gt.png";
    const auto mask200 = tmp.path() / "mask200.png";
    const auto mask251 = tmp.path() / "mask251.png";
    write_rgb_png(image_path, kH, kW);
    write_gray_png(mask200, std::vector<uint8_t>(static_cast<size_t>(kW * kH), 200), kH, kW);
    write_gray_png(mask251, std::vector<uint8_t>(static_cast<size_t>(kW * kH), 251), kH, kW);

    auto radial = Tensor::from_vector({-0.1f, 0.02f}, {2}, Device::CPU);
    auto cam200 = make_camera(image_path, mask200, kW, kH, radial);
    cam200->prepare_undistortion();
    ASSERT_TRUE(cam200->is_undistort_prepared());

    const auto cfg = sai_config();
    Tensor gt;
    auto m200 = load_eval_mask(cam200.get(), gt, false, cfg);
    const auto bytes200 = mask_bytes(m200);
    ASSERT_FALSE(bytes200.empty());
    EXPECT_EQ(*std::max_element(bytes200.begin(), bytes200.end()), 0)
        << "uniform 200 must stay not-keep after undistort; 0.5 snap would keep it";

    auto cam251 = make_camera(image_path, mask251, kW, kH, radial);
    cam251->prepare_undistortion();
    auto m251 = load_eval_mask(cam251.get(), gt, false, cfg);
    const auto bytes251 = mask_bytes(m251);
    ASSERT_FALSE(bytes251.empty());
    EXPECT_EQ(*std::max_element(bytes251.begin(), bytes251.end()), 1)
        << "uniform 251 must stay keep; gt(250) on [0,1] floats would empty the mask";

    auto ext200 = load_external_mask_for_metrics(*cam200, cfg);
    ASSERT_TRUE(ext200.has_value()) << ext200.error();
    const auto ext200_bytes = mask_bytes(*ext200);
    EXPECT_EQ(*std::max_element(ext200_bytes.begin(), ext200_bytes.end()), 0);

    const auto rgba200 = tmp.path() / "rgba200.png";
    write_rgba_png(rgba200, std::vector<uint8_t>(static_cast<size_t>(kW * kH), 200), kH, kW);
    auto alpha_cam = make_camera(rgba200, {}, kW, kH, radial);
    alpha_cam->set_has_alpha(true);
    alpha_cam->prepare_undistortion();
    auto alpha = load_alpha_masked_metrics_inputs(*alpha_cam, cfg);
    ASSERT_TRUE(alpha.has_value()) << alpha.error();
    const auto alpha_bytes = mask_bytes(alpha->mask);
    EXPECT_EQ(*std::max_element(alpha_bytes.begin(), alpha_bytes.end()), 0)
        << "alpha 200 must stay not-keep after undistort";
}

TEST_F(MetricsEvalMask, SegmentModeStillUsesBinaryThreshold) {
    ensure_image_loader();
    UniqueTempDir tmp("lfs_eval_mask_segment");
    const auto image_path = tmp.path() / "gt.png";
    const auto mask_path = tmp.path() / "mask.png";
    write_rgb_png(image_path, kBandH, kBandW);
    write_gray_png(mask_path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);
    auto cam = make_camera(image_path, mask_path, kBandW, kBandH);
    auto cfg = sai_config();
    cfg.mask_mode = MaskMode::Segment;
    Tensor gt;
    auto mask = load_eval_mask(cam.get(), gt, false, cfg);
    const auto got = mask_bytes(mask);
    ASSERT_EQ(got.size(), 6u);
    EXPECT_EQ(got[0], 0) << "byte 127 is below 0.5";
    EXPECT_EQ(got[1], 1) << "byte 128 is keep in binary Segment mode";
    EXPECT_EQ(got[2], 1) << "byte 200 is keep in binary Segment mode";
}

TEST_F(MetricsEvalMask, BatchEvaluatorIgnoresPhotometricErrorInSegmentBand) {
    ensure_image_loader();
    UniqueTempDir tmp("lfs_eval_mask_evaluator");
    constexpr size_t size = 32;
    std::vector<uint8_t> bands(size * size, 255);
    std::vector<float> prediction(3 * size * size);
    constexpr std::array<float, 3> color{64.0f / 255, 96.0f / 255, 160.0f / 255};
    for (size_t y = 0; y < size; ++y) {
        for (size_t x = 0; x < size; ++x) {
            if (x < size / 2)
                bands[y * size + x] = 200;
            for (size_t c = 0; c < 3; ++c)
                prediction[c * size * size + y * size + x] = x < size / 2 ? 0.0f : color[c];
        }
    }
    const auto image_path = tmp.path() / "gt.png";
    const auto mask_path = tmp.path() / "mask.png";
    write_rgba_png(image_path, bands, size, size);
    write_gray_png(mask_path, bands, size, size);
    auto predicted = Tensor::from_vector(prediction, {3, size, size}, Device::CUDA);
    lfs::core::SplatData splat(
        0, Tensor::zeros({1, 3}, Device::CUDA),
        Tensor::zeros({1, 1, 3}, Device::CUDA), Tensor::zeros({1, 0, 3}, Device::CUDA),
        Tensor::full({1, 3}, -2.0f, Device::CUDA),
        Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {1, 4}, Device::CUDA),
        Tensor::full({1, 1}, 2.0f, Device::CUDA), 1.0f);
    auto background = Tensor::zeros({3}, Device::CUDA);
    for (const bool sidecar : {false, true}) {
        SCOPED_TRACE(sidecar ? "sidecar" : "alpha");
        auto cam = make_camera(image_path, sidecar ? mask_path : std::filesystem::path{}, size, size);
        cam->set_has_alpha(true);
        auto dataset = std::make_shared<lfs::training::CameraDataset>(
            std::vector<std::shared_ptr<Camera>>{cam}, lfs::training::DatasetConfig{},
            lfs::training::CameraDataset::Split::ALL);
        for (const auto mode : {MaskMode::SegmentAndIgnore, MaskMode::Ignore}) {
            SCOPED_TRACE(static_cast<int>(mode));
            lfs::core::param::TrainingParameters params;
            params.optimization.enable_eval = true;
            params.optimization.headless = true;
            params.optimization.mask_mode = mode;
            params.optimization.mask_threshold = 0.5f;
            params.optimization.use_alpha_as_mask = true;
            params.dataset.resize_factor = -1;
            params.dataset.max_width = 0;
            params.dataset.output_path = tmp.path();
            MetricsEvaluator evaluator(params);
            evaluator.set_appearance([&](const Tensor&, const Camera&) { return predicted; });
            const auto metrics = evaluator.evaluate(1, splat, dataset, background);
            ASSERT_TRUE(metrics.valid);
            // The only prediction error is in authored band 200. Binary Ignore
            // includes it; SegmentAndIgnore must exclude it from reported PSNR.
            if (mode == MaskMode::SegmentAndIgnore)
                EXPECT_NEAR(metrics.psnr, 100.0f, 0.5f);
            else
                EXPECT_LT(metrics.psnr, 20.0f);
        }
    }
}

TEST_F(MetricsEvalMask, MovedCameraPreservesMaskCacheProcessingKey) {
    for (const bool assignment : {false, true}) {
        SCOPED_TRACE(assignment);
        auto source = make_camera({}, {}, 2, 2);
        source->set_mask_tensor(uint8_hw({255, 0, 255, 0}, 2, 2));
        expect_keep(source->load_and_get_mask(0, 0, true, 0.5f, true),
                    {0, 1, 0, 1}, "source inverted cache");
        auto destination = make_camera({}, {}, 2, 2);
        if (assignment) {
            *destination = std::move(*source);
        } else {
            destination = std::make_shared<Camera>(std::move(*source));
        }
        expect_keep(destination->load_and_get_mask(0, 0, false, 0.5f, true),
                    {1, 0, 1, 0}, "non-inverted load after move");
    }
}

TEST_F(MetricsEvalMask, RgbaPreprocessingUsesExecutionQueue) {
    lfs::core::TensorWorkQueue queue(lfs::core::GpuBackend::CUDA);
    lfs::core::TensorWorkQueue::Scope scope(queue);
    UniqueTempDir tmp("lfs_eval_mask_stream");
    const auto path = tmp.path() / "rgba.png";
    write_rgba_png(path, {kBandBytes.begin(), kBandBytes.end()}, kBandH, kBandW);
    auto camera = make_camera(path, {}, kBandW, kBandH);
    const auto loaded = load_alpha_masked_metrics_inputs(*camera, sai_config());
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    EXPECT_EQ(loaded->gt_image.stream(), queue.native_handle());
    EXPECT_EQ(loaded->mask.stream(), queue.native_handle());
    expect_keep(loaded->mask, {kExpectedKeep.begin(), kExpectedKeep.end()}, "scoped RGBA");
}
