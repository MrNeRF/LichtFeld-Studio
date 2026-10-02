/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "io/pipelined_image_loader.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::default_gpu_backend;
    using lfs::core::Device;
    using lfs::core::gpu_backend_available;
    using lfs::core::gpu_backend_of;
    using lfs::core::SplatData;
    using lfs::core::SplatExportableStorage;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    TEST(SessionImageLoader, DeliversARealImageOnTheSessionBackend) {
        if (!gpu_backend_available(default_gpu_backend()))
            GTEST_SKIP() << "session GPU backend is unavailable";

        const auto directory = std::filesystem::temp_directory_path() / "lfs_session_loader";
        std::filesystem::create_directories(directory);
        const auto path = directory / "sample.png";
        const std::uint8_t rgb[] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0};
        ASSERT_TRUE(lfs::core::save_png(path, rgb, 2, 2, 3, 8, 1));

        lfs::io::PipelinedLoaderConfig config;
        config.backend = default_gpu_backend();
        config.jpeg_batch_size = 1;
        config.prefetch_count = 1;
        config.output_queue_size = 1;
        config.decoder_pool_size = 1;
        config.io_threads = 1;
        config.cold_process_threads = 1;
        config.max_cache_bytes = 1u << 20;
        lfs::io::PipelinedImageLoader loader(config);
        lfs::io::ImageRequest request;
        request.sequence_id = 0;
        request.path = path;
        loader.prefetch({request});
        const auto completion = loader.try_get_completion_for(std::chrono::seconds(20));
        ASSERT_TRUE(completion && completion->outcome) << "session loader produced no image";
        const auto& image = completion->outcome->tensor;
        ASSERT_TRUE(image.is_valid());
        EXPECT_EQ(gpu_backend_of(image), default_gpu_backend());
        EXPECT_EQ(image.shape(), TensorShape({3, 2, 2}));
        const auto host = image.cpu().to_vector();
        ASSERT_EQ(host.size(), 12u);
        EXPECT_NEAR(host[0], 1.0f, 1e-6f);
        EXPECT_NEAR(host[1], 0.0f, 1e-6f);

        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }

    // Training storage carves the q16 codes and their bounds from one block;
    // re-quantizing a float shN into it must not treat the two as aliases.
    TEST(SessionSplatStorage, QuantizesFloatShNIntoTheExportableBlock) {
        if (!gpu_backend_available(default_gpu_backend()))
            GTEST_SKIP() << "session GPU backend is unavailable";
        lfs::core::sh_value_quant::set_enabled_for_testing(true);
        constexpr std::size_t kLive = 40;
        constexpr std::size_t kRest = 15;
        auto created = SplatExportableStorage::create(64, 3);
        ASSERT_TRUE(created.has_value()) << created.error();
        const auto allocator = created->make_allocator();
        const auto param = [&](const TensorShape& shape, const char* name) {
            Tensor tensor = allocator(shape, 64, DataType::Float32, name);
            tensor.zero_();
            return tensor;
        };
        std::vector<float> rest(kLive * kRest * 3);
        for (std::size_t i = 0; i < rest.size(); ++i)
            rest[i] = std::sin(0.37f * static_cast<float>(i));
        SplatData model(3, param(TensorShape({kLive, 3}), "SplatData.means"),
                        param(TensorShape({kLive, 1, 3}), "SplatData.sh0"),
                        Tensor::from_vector(rest, TensorShape({kLive, kRest, 3}), Device::CPU).to(Device::GPU),
                        param(TensorShape({kLive, 3}), "SplatData.scaling"),
                        param(TensorShape({kLive, 4}), "SplatData.rotation"),
                        param(TensorShape({kLive, 1}), "SplatData.opacity"), 1.0f, SplatData::ShNLayout::Canonical);
        model.set_tensor_allocator(allocator);

        ASSERT_TRUE(model.apply_shN_value_quant());
        EXPECT_TRUE(model.shN_value_quantized());
        const auto decoded = model.shN_canonical().cpu().to_vector();
        ASSERT_EQ(decoded.size(), rest.size());
        for (std::size_t i = 0; i < rest.size(); ++i)
            EXPECT_NEAR(decoded[i], rest[i], 2.0f / 65535.0f) << i;
        lfs::core::sh_value_quant::set_enabled_for_testing(std::nullopt);
    }

    TEST(SessionSplatStorage, CreateGrowRebindOnTheSessionBackend) {
        if (!gpu_backend_available(default_gpu_backend()))
            GTEST_SKIP() << "session GPU backend is unavailable";

        constexpr std::size_t kInitial = 32;
        constexpr std::size_t kGrown = 64;
        constexpr std::size_t kLive = 16;
        constexpr int kShDegree = 0;

        auto created = SplatExportableStorage::create(kInitial, kShDegree, 0, kGrown * 2);
        ASSERT_TRUE(created.has_value()) << created.error();
        auto storage = std::move(*created);
        ASSERT_TRUE(storage.valid());
        const void* const stable = storage.block->device_ptr;
        EXPECT_EQ(storage.capacity(), kInitial);
        EXPECT_GE(storage.block->committed_bytes, storage.region_bytes[SplatExportableStorage::Means]);

        auto allocator = storage.make_allocator();
        Tensor means = allocator(TensorShape({kLive, 3}), kInitial, DataType::Float32, "SplatData.means");
        Tensor scaling = allocator(TensorShape({kLive, 3}), kInitial, DataType::Float32, "SplatData.scaling");
        Tensor rotation = allocator(TensorShape({kLive, 4}), kInitial, DataType::Float32, "SplatData.rotation");
        Tensor opacity = allocator(TensorShape({kLive, 1}), kInitial, DataType::Float32, "SplatData.opacity");
        Tensor sh0 = allocator(TensorShape({kLive, 1, 3}), kInitial, DataType::Float32, "SplatData.sh0");
        EXPECT_EQ(gpu_backend_of(means), default_gpu_backend());
        EXPECT_EQ(means.external_storage_kind(), "splat.exportable");

        std::vector<float> pattern(kLive * 3);
        for (std::size_t i = 0; i < pattern.size(); ++i)
            pattern[i] = static_cast<float>(i + 1);
        means.copy_from(Tensor::from_vector(pattern, TensorShape({kLive, 3}), Device::CPU));

        auto grew = storage.grow(kGrown);
        ASSERT_TRUE(grew.has_value()) << grew.error();
        EXPECT_TRUE(*grew);
        EXPECT_EQ(storage.block->device_ptr, stable);
        EXPECT_EQ(storage.capacity(), kGrown);

        SplatData model(kShDegree, std::move(means), std::move(sh0), Tensor{}, std::move(scaling),
                        std::move(rotation), std::move(opacity), 1.0f, SplatData::ShNLayout::Swizzled);
        auto rebound = storage.rebindSplatData(model);
        ASSERT_TRUE(rebound.has_value()) << rebound.error();
        EXPECT_EQ(model.means_raw().capacity(), kGrown);
        EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");
        EXPECT_TRUE(model.means_raw().has_exportable_provenance());
        EXPECT_EQ(model.means_raw().cpu().to_vector(), pattern);

        const auto opacity_values =
            storage.make_allocator()(TensorShape({kGrown, 1}), kGrown, DataType::Float32, "SplatData.opacity")
                .cpu()
                .to_vector();
        ASSERT_EQ(opacity_values.size(), kGrown);
        for (std::size_t i = kInitial; i < kGrown; ++i)
            EXPECT_TRUE(std::isinf(opacity_values[i]) && opacity_values[i] < 0.0f) << i;

        const auto rotation_values =
            storage.make_allocator()(TensorShape({kGrown, 4}), kGrown, DataType::Float32, "SplatData.rotation")
                .cpu()
                .to_vector();
        ASSERT_EQ(rotation_values.size(), kGrown * 4);
        for (std::size_t row = kInitial; row < kGrown; ++row) {
            EXPECT_FLOAT_EQ(rotation_values[row * 4], 1.0f);
            EXPECT_FLOAT_EQ(rotation_values[row * 4 + 1], 0.0f);
            EXPECT_FLOAT_EQ(rotation_values[row * 4 + 2], 0.0f);
            EXPECT_FLOAT_EQ(rotation_values[row * 4 + 3], 0.0f);
        }
    }

} // namespace
