/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include "training/dataset.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

namespace {
    using namespace lfs::core;
    constexpr size_t width = 129;
    constexpr size_t height = 97;

    class InMemoryMaskAlignment : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "CUDA unavailable";
        }

        static std::shared_ptr<Camera> camera(const std::filesystem::path& image,
                                              const std::filesystem::path& mask = {},
                                              const int calibration_scale = 1) {
            return std::make_shared<Camera>(
                Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                100.f, 100.f, width / 2.f, height / 2.f, Tensor{}, Tensor{}, CameraModelType::PINHOLE,
                "frame.png", image, mask, width * calibration_scale, height * calibration_scale, 0);
        }

        static void image(const std::filesystem::path& path) {
            const std::vector<uint8_t> pixels(width * height * 3, 96);
            ASSERT_TRUE(save_png(path, pixels.data(), width, height, 3, 8, 0));
        }

        static lfs::io::PipelinedLoaderConfig loader_config() {
            lfs::io::PipelinedLoaderConfig result;
            result.prefetch_count = 1;
            result.output_queue_size = 2;
            result.jpeg_batch_size = 1;
            result.decoder_pool_size = 1;
            result.io_threads = 1;
            result.cold_process_threads = 1;
            result.max_cache_bytes = 8 * 1024 * 1024;
            return result;
        }
    };
} // namespace

TEST_F(InMemoryMaskAlignment, CameraResizesOnlySourceGridMasks) {
    const lfs::test::licht::TemporaryDirectory temp("mask-memory-camera-grid");
    const auto path = temp.path / "frame.png";
    image(path);
    for (const int factor : {1, 2, 4}) {
        const auto [target_w, target_h] = resized_image_dimensions(width, height, factor, 0);
        for (const bool channel_last : {false, true}) {
            SCOPED_TRACE(factor);
            SCOPED_TRACE(channel_last);
            auto cam = camera(path, {}, 2); // Calibration is larger than the selected image file.
            const auto shape = channel_last ? TensorShape({height, width, size_t{1}})
                                            : TensorShape({size_t{1}, height, width});
            cam->set_mask_tensor(Tensor::full(shape, .25f, Device::CPU));
            const auto mask = cam->load_and_get_mask(factor, 0, false, 0.f, false);
            ASSERT_TRUE(mask.is_valid());
            EXPECT_EQ(mask.shape(), (TensorShape({static_cast<size_t>(target_h), static_cast<size_t>(target_w)})));
            const auto values = mask.cpu().to_vector();
            EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float v) { return std::abs(v - .25f) < 1e-4f; }));
            EXPECT_EQ(cam->load_and_get_mask(factor, 0, false, 0.f, false).shape(), mask.shape());
        }
    }

    auto max_width_camera = camera(path);
    max_width_camera->set_mask_tensor(Tensor::full({height, width}, .25f, Device::CPU));
    const auto [target_w, target_h] = resized_image_dimensions(width, height, 1, 32);
    const auto narrowed = max_width_camera->load_and_get_mask(1, 32, true, .5f, true);
    EXPECT_EQ(narrowed.shape(), (TensorShape({static_cast<size_t>(target_h), static_cast<size_t>(target_w)})));
    const auto values = narrowed.cpu().to_vector();
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float v) { return v == 1.f; }));
}

TEST_F(InMemoryMaskAlignment, AlreadyScaledAndUnknownGridsKeepTheirAcceptedShape) {
    const lfs::test::licht::TemporaryDirectory temp("mask-memory-accepted-grids");
    const auto path = temp.path / "frame.png";
    image(path);
    auto cam = camera(path);
    cam->set_mask_tensor(Tensor::full({48, 64}, .25f, Device::CPU));
    EXPECT_EQ(cam->load_and_get_mask(2, 0, false, 0.f, false).shape(), (TensorShape({48, 64})));
    cam->set_mask_tensor(Tensor::full({7, 9}, .75f, Device::CPU));
    EXPECT_EQ(cam->load_and_get_mask(2, 0, false, 0.f, false).shape(), (TensorShape({7, 9})));

    auto no_file = camera({});
    no_file->set_mask_tensor(Tensor::ones({height, width}, Device::CPU));
    EXPECT_EQ(no_file->load_and_get_mask(2, 0, false, 0.f, false).shape(), (TensorShape({48, 64})));
}

TEST_F(InMemoryMaskAlignment, TrainingLoaderUsesMemoryMaskOnTheImageGrid) {
    const lfs::test::licht::TemporaryDirectory temp("mask-memory-training-grid");
    const auto image_path = temp.path / "frame.png";
    const auto sidecar_path = temp.path / "sidecar.png";
    image(image_path);
    const std::vector<uint8_t> white(width * height, 255);
    ASSERT_TRUE(save_png(sidecar_path, white.data(), width, height, 1, 8, 0));
    for (const int factor : {2, 4}) {
        SCOPED_TRACE(factor);
        auto cam = camera(image_path, sidecar_path);
        cam->set_mask_tensor(Tensor::zeros({height, width}, Device::CPU));
        lfs::training::DatasetConfig config;
        config.resize_factor = factor;
        auto dataset = std::make_shared<lfs::training::CameraDataset>(
            std::vector<std::shared_ptr<Camera>>{cam}, config);
        lfs::training::PipelinedAuxiliaryImageConfig aux;
        aux.load_masks = true;
        aux.mask_threshold = .5f;
        lfs::training::PipelinedDataLoader<lfs::training::InfiniteRandomSampler> loader(
            dataset, lfs::training::InfiniteRandomSampler(1, 42), loader_config(), aux);
        const auto ready = loader.next();
        ASSERT_TRUE(ready);
        ASSERT_TRUE(ready->mask);
        EXPECT_EQ(ready->mask->shape()[0], ready->data.image.shape()[1]);
        EXPECT_EQ(ready->mask->shape()[1], ready->data.image.shape()[2]);
        const auto values = ready->mask->to(DataType::Float32).cpu().to_vector();
        EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float v) { return v == 0.f; }));
    }
}
