/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "io/pipelined_image_loader.hpp"
#include "training/dataset.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread>
#include <vector>

namespace {

    std::filesystem::path bicycle_image(const char* name) {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4" / name;
    }

    lfs::io::ReadyImage load_one(lfs::io::PipelinedImageLoader& loader,
                                 const size_t sequence,
                                 const std::filesystem::path& path) {
        lfs::io::ImageRequest request;
        request.sequence_id = sequence;
        request.path = path;
        request.params.output_uint8 = false;
        loader.prefetch({request});
        return loader.get();
    }

    std::shared_ptr<lfs::training::CameraDataset> make_training_dataset(
        const std::filesystem::path& path, const int resize_factor = 8) {
        auto camera = std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            1800.0f, 1800.0f, 1920.0f, 1080.0f,
            lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::PINHOLE,
            path.filename().string(), path, std::filesystem::path{}, 3840, 2160, 1);
        lfs::training::DatasetConfig dataset_config;
        dataset_config.resize_factor = resize_factor;
        return std::make_shared<lfs::training::CameraDataset>(
            std::vector<std::shared_ptr<lfs::core::Camera>>{std::move(camera)},
            dataset_config, lfs::training::CameraDataset::Split::ALL);
    }

} // namespace

TEST(CompressedBlobTier, HostCacheHit) {
    const auto path = bicycle_image("_DSC8739.JPG");
    if (!std::filesystem::is_regular_file(path))
        GTEST_SKIP() << "bicycle dataset is absent: " << path;

    lfs::io::PipelinedLoaderConfig config;
    config.jpeg_batch_size = 1;
    config.prefetch_count = 2;
    config.output_queue_size = 1;
    lfs::io::PipelinedImageLoader loader(config);
    ASSERT_TRUE(load_one(loader, 1, path).tensor.is_valid());
    ASSERT_TRUE(load_one(loader, 2, path).tensor.is_valid());
    const auto stats = loader.get_stats();
    EXPECT_GT(stats.jpeg_cache_bytes, 0u);
    EXPECT_GT(stats.jpeg_cache_entries, 0u);
}

TEST(CompressedBlobTier, CacheHitMatchesProcessedReferenceWithResizeAndUndistort) {
    const auto path = bicycle_image("_DSC8739.JPG");
    if (!std::filesystem::is_regular_file(path))
        GTEST_SKIP() << "bicycle dataset is absent: " << path;

    lfs::core::UndistortParams undistort{};
    undistort.src_fx = 1800.0f;
    undistort.src_fy = 1800.0f;
    undistort.src_cx = 1920.0f;
    undistort.src_cy = 1080.0f;
    undistort.dst_fx = 1750.0f;
    undistort.dst_fy = 1750.0f;
    undistort.dst_cx = 960.0f;
    undistort.dst_cy = 540.0f;
    undistort.src_width = 3840;
    undistort.src_height = 2160;
    undistort.dst_width = 1920;
    undistort.dst_height = 1080;
    undistort.model_type = lfs::core::CameraModelType::PINHOLE;
    undistort.distortion[0] = -0.08f;
    undistort.num_distortion = 1;

    lfs::io::PipelinedLoaderConfig config;
    config.jpeg_batch_size = 1;
    config.prefetch_count = 2;
    config.output_queue_size = 1;
    lfs::io::PipelinedImageLoader loader(config);

    lfs::io::ImageRequest request;
    request.path = path;
    request.params.resize_factor = 2;
    request.params.max_width = 900;
    request.undistort = &undistort;
    request.params.undistort = request.undistort;

    request.sequence_id = 1;
    loader.prefetch({request});
    const auto reference = loader.get();
    ASSERT_TRUE(reference.error.empty()) << reference.error;
    ASSERT_TRUE(reference.tensor.is_valid());

    request.sequence_id = 2;
    loader.prefetch({request});
    const auto cached = loader.get();
    ASSERT_TRUE(cached.error.empty()) << cached.error;
    ASSERT_TRUE(cached.tensor.is_valid());
    ASSERT_EQ(reference.tensor.shape(), cached.tensor.shape());

    const auto reference_pixels = reference.tensor.cpu().to_vector();
    const auto cached_pixels = cached.tensor.cpu().to_vector();
    ASSERT_EQ(reference_pixels.size(), cached_pixels.size());
    double mean_abs_error = 0.0;
    for (size_t i = 0; i < reference_pixels.size(); ++i)
        mean_abs_error += std::abs(reference_pixels[i] - cached_pixels[i]);
    mean_abs_error /= static_cast<double>(reference_pixels.size());
    EXPECT_LT(mean_abs_error, 0.02);
}

TEST(CompressedBlobTier, AdaptiveControllerUsesRealDataLoaderAndSlowConsumer) {
    const auto path = bicycle_image("_DSC8739.JPG");
    if (!std::filesystem::is_regular_file(path))
        GTEST_SKIP() << "bicycle dataset is absent: " << path;

    const auto settle_target = [&](const bool slow) {
        auto dataset = make_training_dataset(path);
        lfs::io::PipelinedLoaderConfig config;
        config.prefetch_count = 8;
        config.jpeg_batch_size = 8;
        config.output_queue_size = 2;
        auto loader = lfs::training::create_infinite_pipelined_dataloader(dataset, config);
        for (size_t iter = 1; iter <= 256; ++iter) {
            auto example = loader->next();
            EXPECT_TRUE(example.has_value());
            if (!example)
                return loader->get_loader()->adaptive_prefetch_target();
            loader->get_loader()->record_decode_latency(2.0);
            loader->observe_training_iteration(slow ? 20.0 : 1.0, 0.0, iter);
            if (slow)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return loader->get_loader()->adaptive_prefetch_target();
    };

    const auto fast_target = settle_target(false);
    const auto slow_target = settle_target(true);
    EXPECT_GT(fast_target, slow_target);
    EXPECT_GE(slow_target, 4u);
    EXPECT_LE(fast_target, 12u);
}

TEST(CompressedBlobTier, AdaptiveControllerWaitGrowthArmsShrinkCooldown) {
    const auto path = bicycle_image("_DSC8739.JPG");
    if (!std::filesystem::is_regular_file(path))
        GTEST_SKIP() << "bicycle dataset is absent: " << path;

    auto dataset = make_training_dataset(path);
    lfs::io::PipelinedLoaderConfig config;
    config.prefetch_count = 8;
    config.jpeg_batch_size = 8;
    config.output_queue_size = 2;
    auto loader = lfs::training::create_infinite_pipelined_dataloader(dataset, config);

    for (size_t iter = 1; iter <= 64; ++iter) {
        auto example = loader->next();
        ASSERT_TRUE(example.has_value());
        loader->get_loader()->record_decode_latency(2.0);
        loader->observe_training_iteration(1.0, 1.0, iter);
    }
    const auto grown_target = loader->get_loader()->adaptive_prefetch_target();
    ASSERT_GT(grown_target, 8u);

    for (size_t iter = 65; iter <= 256; ++iter) {
        auto example = loader->next();
        ASSERT_TRUE(example.has_value());
        loader->get_loader()->record_decode_latency(2.0);
        loader->observe_training_iteration(20.0, 0.0, iter);
    }
    EXPECT_EQ(loader->get_loader()->adaptive_prefetch_target(), grown_target);
}

TEST(CompressedBlobTier, HeldCameraExampleKeepsRingFrameStable) {
    const auto path = bicycle_image("_DSC8739.JPG");
    if (!std::filesystem::is_regular_file(path))
        GTEST_SKIP() << "bicycle dataset is absent: " << path;

    auto dataset = make_training_dataset(path);
    lfs::io::PipelinedLoaderConfig config;
    config.prefetch_count = 8;
    config.jpeg_batch_size = 8;
    config.output_queue_size = 2;
    auto loader = lfs::training::create_infinite_pipelined_dataloader(dataset, config);
    auto held = loader->next();
    ASSERT_TRUE(held.has_value());
    const auto before = held->data.image.cpu().to_vector();

    for (size_t i = 0; i < lfs::io::DECODE_FRAME_RING_CAPACITY + 2; ++i) {
        auto next = loader->next();
        ASSERT_TRUE(next.has_value());
    }

    const auto after = held->data.image.cpu().to_vector();
    EXPECT_EQ(before, after);
}
