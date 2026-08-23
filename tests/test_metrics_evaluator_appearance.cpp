/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/image_io.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/dataset.hpp"
#include "training/metrics/metrics.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

    using lfs::core::Camera;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::training::CameraDataset;
    using lfs::training::MetricsEvaluator;

    constexpr int kImageSize = 32;
    constexpr int kSeparator = 4;

    // Unique per call so parallel/repeated runs do not share /tmp/lfs_eval_appearance_*.
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

    void write_constant_png(const std::filesystem::path& path, const float value) {
        auto image = Tensor::full({3, static_cast<size_t>(kImageSize), static_cast<size_t>(kImageSize)},
                                  value, Device::CPU);
        lfs::core::save_image(path, image);
    }

    std::shared_ptr<Camera> make_eval_camera(const std::filesystem::path& image_path, const int uid) {
        auto R = Tensor::eye(3, Device::CPU);
        auto T = Tensor::from_vector({0.0f, 0.0f, 3.0f}, {3}, Device::CPU);
        return std::make_shared<Camera>(
            R, T,
            40.0f, 40.0f,
            kImageSize * 0.5f, kImageSize * 0.5f,
            Tensor(), Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            image_path.filename().string(),
            image_path,
            std::filesystem::path{},
            kImageSize, kImageSize,
            uid);
    }

    std::unique_ptr<SplatData> make_eval_splat() {
        const size_t n = 1;
        auto means = Tensor::zeros({n, 3}, Device::CUDA);
        auto sh0 = Tensor::full({n, 1, 3}, 0.5f, Device::CUDA);
        auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
        auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
        auto rotation = Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {n, 4}, Device::CUDA);
        auto opacity = Tensor::full({n, 1}, 2.0f, Device::CUDA);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    lfs::core::param::TrainingParameters make_eval_params(const std::filesystem::path& output) {
        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = output;
        params.optimization.enable_eval = true;
        params.optimization.enable_save_eval_images = true;
        params.optimization.headless = true;
        params.optimization.gut = false;
        params.optimization.mask_mode = lfs::core::param::MaskMode::None;
        params.optimization.use_alpha_as_mask = false;
        return params;
    }

} // namespace

class MetricsEvaluatorAppearanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Drop leftover EvaluationCompleted handlers (TrainerManager never unsubscribes).
        lfs::event::EventBridge::instance().clear_all();
    }

    void TearDown() override {
        lfs::event::EventBridge::instance().clear_all();
    }
};

TEST_F(MetricsEvaluatorAppearanceTest, HookNotConsultedWhenUnset) {
    const UniqueTempDir root("lfs_eval_appearance_unset");
    write_constant_png(root.path() / "gt.png", 0.5f);

    auto cam = make_eval_camera(root.path() / "gt.png", 1);
    CameraDataset dataset({cam}, {}, CameraDataset::Split::ALL);
    auto splat = make_eval_splat();
    auto background = Tensor::zeros({3}, Device::CUDA);

    auto params = make_eval_params(root.path());
    MetricsEvaluator evaluator(params);

    int calls = 0;
    auto metrics = evaluator.evaluate(1, *splat, std::make_shared<CameraDataset>(dataset), background);
    EXPECT_TRUE(metrics.valid);
    EXPECT_EQ(calls, 0);

    evaluator.set_appearance([&](const Tensor& rgb, const Camera&) {
        ++calls;
        return rgb;
    });
    metrics = evaluator.evaluate(1, *splat, std::make_shared<CameraDataset>(dataset), background);
    EXPECT_TRUE(metrics.valid);
    EXPECT_EQ(calls, 1);
}

TEST_F(MetricsEvaluatorAppearanceTest, HookRunsOncePerFrameBeforeClampAndSavesCorrectedRender) {
    const UniqueTempDir root("lfs_eval_appearance_hook");
    write_constant_png(root.path() / "a.png", 1.0f);
    write_constant_png(root.path() / "b.png", 1.0f);

    auto cam_a = make_eval_camera(root.path() / "a.png", 1);
    auto cam_b = make_eval_camera(root.path() / "b.png", 2);
    auto splat = make_eval_splat();
    auto background = Tensor::zeros({3}, Device::CUDA);

    auto params = make_eval_params(root.path());
    MetricsEvaluator evaluator(params);

    int calls = 0;
    evaluator.set_appearance([&](const Tensor& rgb, const Camera&) {
        ++calls;
        return Tensor::full(rgb.shape(), 2.0f, rgb.device());
    });

    auto val_dataset = std::make_shared<CameraDataset>(
        std::vector<std::shared_ptr<Camera>>{cam_a, cam_b}, lfs::training::DatasetConfig{},
        CameraDataset::Split::ALL);
    auto metrics = evaluator.evaluate(7, *splat, val_dataset, background);
    ASSERT_TRUE(metrics.valid);
    EXPECT_EQ(calls, 2);

    // Hook returns 2; clamp after the hook yields 1, matching the white GT.
    // Without the post-hook clamp, PSNR against 1 would be 0 dB.
    EXPECT_NEAR(metrics.psnr, 100.0f, 0.5f);

    const auto saved = root.path() / "eval_step_7" / "0.png";
    ASSERT_TRUE(std::filesystem::exists(saved));
    auto [data, width, height, channels] = lfs::core::load_image(saved);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(height, kImageSize);
    ASSERT_EQ(width, kImageSize + kSeparator + kImageSize);
    ASSERT_GE(channels, 3);
    const int render_x = kImageSize + kSeparator;
    const int idx = (0 * width + render_x) * channels;
    EXPECT_EQ(data[idx], 255);
    EXPECT_EQ(data[idx + 1], 255);
    EXPECT_EQ(data[idx + 2], 255);
    lfs::core::free_image(data);
}
