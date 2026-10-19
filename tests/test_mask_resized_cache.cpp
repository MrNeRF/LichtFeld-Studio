/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/camera.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "io/cache_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {
    using namespace lfs::core;
    constexpr size_t width = 129, height = 97;
    class MaskResizedCache : public ::testing::Test {
    protected:
        lfs::test::licht::TemporaryDirectory temp{"mask-resized-cache"};
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
                GTEST_SKIP() << "CUDA unavailable";
            lfs::io::CacheLoader::getInstance(false);
            set_image_loader([](const ImageLoadParams& p) {
                return lfs::io::CacheLoader::getInstance().load_cached_image(p.path,
                                                                             {.resize_factor = p.resize_factor, .max_width = p.max_width, .cuda_stream = p.stream, .output_uint8 = p.output_uint8, .mask_grayscale = p.mask_grayscale});
            });
            const std::vector<uint8_t> rgb(width * height * 3, 96);
            ASSERT_TRUE(save_png(temp.path / "image.png", rgb.data(), width, height, 3, 8, 0));
        }
        auto config() {
            lfs::io::PipelinedLoaderConfig c;
            c.prefetch_count = 1;
            c.output_queue_size = 2;
            c.jpeg_batch_size = 1;
            c.decoder_pool_size = 1;
            c.io_threads = 1;
            c.cold_process_threads = 1;
            c.max_cache_bytes = 8 * 1024 * 1024;
            return c;
        }
        auto camera(bool warped) {
            auto radial = warped ? Tensor::from_vector({-.1f, .02f}, {2}, Device::CPU) : Tensor{};
            auto cam = std::make_shared<Camera>(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                                                110.f, 110.f, width / 2.f, height / 2.f, radial, Tensor{}, CameraModelType::PINHOLE,
                                                "image.png", temp.path / "image.png", temp.path / "mask.png", width, height, 0);
            if (warped)
                cam->prepare_undistortion();
            return cam;
        }
    };
} // namespace

TEST_F(MaskResizedCache, CameraColdHotAndSpilledMasksShareTheExistingByteGrid) {
    nlohmann::json evidence = nlohmann::json::array();
    for (const bool every_byte : {false, true}) {
        constexpr uint8_t bands[]{0, 1, 63, 127, 128, 129, 200, 249, 250, 251, 254, 255};
        std::vector<uint8_t> bytes(width * height);
        std::vector<float> normalized(bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = every_byte ? static_cast<uint8_t>(i % 256) : bands[(i % width) * 12 / width];
            normalized[i] = static_cast<float>(static_cast<double>(bytes[i]) / 255.0);
        }
        ASSERT_TRUE(save_png(temp.path / "mask.png", bytes.data(), width, height, 1, 8, 0));
        const auto reference_source = Tensor::from_vector(normalized, {height, width}, Device::CUDA);
        for (const bool warped : {false, true}) {
            for (const int resize : {1, 2, 4}) {
                auto cam = camera(warped);
                auto reference = reference_source;
                if (warped) {
                    const auto scaled = prepare_undistort_params(cam->undistort_params(), width, height, resize, 0);
                    reference = undistort_mask_area(reference_source, scaled, nullptr);
                } else if (resize != 1) {
                    const auto [w, h] = resized_image_dimensions(width, height, resize, 0);
                    reference = lanczos_resize_grayscale(reference_source, h, w);
                }
                auto grid = reference.cpu().to_vector();
                // Independent host version of the established cache truncation,
                // followed by correctly rounded normalization.
                for (float& value : grid) {
                    const auto byte = static_cast<uint8_t>(std::clamp(value * 255.f, 0.f, 255.f));
                    value = static_cast<float>(static_cast<double>(byte) / 255.0);
                }
                for (const bool invert : {false, true}) {
                    for (const float threshold : {0.f, std::nextafter(0.f, 1.f), .5f, 1.f}) {
                        SCOPED_TRACE(every_byte);
                        SCOPED_TRACE(warped);
                        SCOPED_TRACE(resize);
                        SCOPED_TRACE(invert);
                        SCOPED_TRACE(threshold);
                        std::vector<float> expected(grid.size());
                        for (size_t i = 0; i < grid.size(); ++i) {
                            const float value = invert ? 1.f - grid[i] : grid[i];
                            expected[i] = threshold > 0.f ? float(value >= threshold) : value;
                        }
                        lfs::io::PipelinedImageLoader loader(config());
                        lfs::io::ImageRequest r;
                        r.path = temp.path / "image.png";
                        r.mask_path = temp.path / "mask.png";
                        r.params.resize_factor = resize;
                        r.mask_params = {.invert = invert, .threshold = threshold};
                        if (warped)
                            r.params.undistort = r.undistort = &cam->undistort_params();
                        std::vector<float> first;
                        for (size_t visit = 0; visit < 3; ++visit) {
                            if (visit == 2) {
                                const auto stats = loader.get_stats();
                                ASSERT_GT(loader.release_host_cache(stats.jpeg_cache_bytes), 0u);
                            }
                            r.sequence_id = visit;
                            loader.prefetch({r});
                            const auto ready = loader.get();
                            ASSERT_TRUE(ready.error.empty()) << ready.error;
                            ASSERT_TRUE(ready.mask);
                            const auto actual = ready.mask->cpu().to_vector();
                            ASSERT_EQ(actual.size(), expected.size());
                            size_t wrong = 0;
                            for (size_t i = 0; i < actual.size(); ++i)
                                wrong += std::bit_cast<uint32_t>(actual[i]) != std::bit_cast<uint32_t>(expected[i]);
                            EXPECT_EQ(wrong, 0u) << "visit=" << visit;
                            if (!visit)
                                first = actual;
                            if (!every_byte && !warped && !invert && visit == 1)
                                evidence.push_back({{"resize", resize}, {"threshold", threshold}, {"first", {{"shape", {ready.mask->shape()[0], ready.mask->shape()[1]}}, {"values", first}}}, {"second", {{"shape", {ready.mask->shape()[0], ready.mask->shape()[1]}}, {"values", actual}}}});
                        }
                        const auto evaluated = cam->load_and_get_mask(resize, 0, invert, threshold, threshold > 0.f).to(DataType::Float32).cpu().to_vector();
                        ASSERT_EQ(evaluated.size(), expected.size());
                        size_t wrong = 0;
                        for (size_t i = 0; i < evaluated.size(); ++i)
                            wrong += std::bit_cast<uint32_t>(evaluated[i]) != std::bit_cast<uint32_t>(expected[i]);
                        EXPECT_EQ(wrong, 0u) << "Camera evaluation";
                    }
                }
            }
        }
    }
    if (const char* path = std::getenv("LFS_MASK_RESIZED_EVIDENCE"))
        std::ofstream(path) << evidence.dump();
}
