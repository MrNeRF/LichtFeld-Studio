/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "io/cache_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"

#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace {
    constexpr int width = 32;
    constexpr int height = 24;

    void write_gray_tiff(const std::filesystem::path& path, const uint16_t ignored_sample = 129) {
        std::vector<uint8_t> bytes{'I', 'I'};
        const auto u16 = [&](uint16_t value) {
            bytes.push_back(static_cast<uint8_t>(value));
            bytes.push_back(static_cast<uint8_t>(value >> 8));
        };
        const auto u32 = [&](uint32_t value) {
            u16(static_cast<uint16_t>(value));
            u16(static_cast<uint16_t>(value >> 16));
        };
        constexpr uint16_t fields = 10;
        constexpr uint32_t pixel_offset = 8 + 2 + fields * 12 + 4;
        u16(42);
        u32(8);
        u16(fields);
        const auto field = [&](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
            u16(tag);
            u16(type);
            u32(count);
            u32(value);
        };
        field(256, 4, 1, width);
        field(257, 4, 1, height);
        field(258, 3, 1, 16);
        field(259, 3, 1, 1);
        field(262, 3, 1, 1);
        field(273, 4, 1, pixel_offset);
        field(277, 3, 1, 1);
        field(278, 4, 1, height);
        field(279, 4, 1, width * height * 2);
        field(284, 3, 1, 1);
        u32(0);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                u16(x >= 8 && x < 20 ? ignored_sample : 65535);
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    lfs::io::PipelinedLoaderConfig config() {
        lfs::io::PipelinedLoaderConfig value;
        value.prefetch_count = 1;
        value.output_queue_size = 2;
        value.jpeg_batch_size = 1;
        value.decoder_pool_size = 1;
        value.io_threads = 1;
        value.cold_process_threads = 1;
        value.max_cache_bytes = 8 * 1024 * 1024;
        return value;
    }
} // namespace

TEST(MaskTiffConsistency, ThresholdBoundaryAgreesAcrossTrainingCameraAndRgbFill) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";
    lfs::io::CacheLoader::getInstance(false);
    lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& params) {
        return lfs::io::CacheLoader::getInstance().load_cached_image(
            params.path, {.resize_factor = params.resize_factor,
                          .max_width = params.max_width,
                          .cuda_stream = params.stream,
                          .output_uint8 = params.output_uint8,
                          .mask_grayscale = params.mask_grayscale});
    });
    const lfs::test::licht::TemporaryDirectory temp("mask-tiff-consumers");
    const auto mask_path = temp.path / "mask.tiff";
    write_gray_tiff(mask_path);
    ASSERT_EQ(std::filesystem::file_size(mask_path), static_cast<size_t>(8 + 2 + 10 * 12 + 4 + width * height * 2));

    std::array<uint8_t, width * height * 3> original{}, changed{};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto i = (y * width + x) * 3 + c;
                original[i] = static_cast<uint8_t>(40 + c * 45);
                changed[i] = x >= 8 && x < 20 ? static_cast<uint8_t>(210 - c * 35) : original[i];
            }
    ASSERT_TRUE(lfs::core::save_png(temp.path / "original.png", original.data(), width, height, 3, 8, 0));
    ASSERT_TRUE(lfs::core::save_png(temp.path / "changed.png", changed.data(), width, height, 3, 8, 0));

    lfs::core::Camera camera(
        lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
        lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
        100.f, 100.f, width / 2.f, height / 2.f, {}, {}, lfs::core::CameraModelType::PINHOLE,
        "original.png", temp.path / "original.png", mask_path, width, height, 0);
    constexpr float cutoff = std::bit_cast<float>(uint32_t{0x3b808081}); // fl(1/255)
    const auto evaluation = camera.load_and_get_mask(1, 0, false, cutoff, true).to(lfs::core::DataType::Float32).cpu().to_vector();

    std::vector<std::vector<float>> images;
    std::vector<float> training;
    for (const char* name : {"original.png", "changed.png"}) {
        lfs::io::PipelinedImageLoader loader(config());
        lfs::io::ImageRequest request;
        request.path = temp.path / name;
        request.mask_path = mask_path;
        request.mask_params.threshold = cutoff;
        loader.prefetch({request});
        auto ready = loader.get();
        ASSERT_TRUE(ready.error.empty()) << ready.error;
        ASSERT_TRUE(ready.mask);
        if (training.empty())
            training = ready.mask->cpu().to_vector();
        else
            EXPECT_EQ(training, ready.mask->cpu().to_vector());
        images.push_back(ready.tensor.cpu().to_vector());
    }
    ASSERT_EQ(evaluation.size(), training.size());
    EXPECT_EQ(evaluation, training) << "Every TIFF mask consumer must use the same byte-grid cutoff";
    EXPECT_EQ(training[8], 0.f) << "Source sample 129/65535 is below byte 1/255";
    ASSERT_EQ(images[0].size(), images[1].size());
    size_t changed_kept = 0;
    for (size_t i = 0; i < images[0].size(); ++i)
        if (training[i % training.size()] > 0.f)
            changed_kept += images[0][i] != images[1][i];
    EXPECT_EQ(changed_kept, 0u) << "Changing ignored RGB must not alter retained decoded pixels";
}
