/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "io/cache_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include <array>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    constexpr size_t width = 129, height = 97;
    class MaskColorConsistency : public ::testing::Test {
    protected:
        lfs::test::licht::TemporaryDirectory temp{"mask-color-consistency"};
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
                GTEST_SKIP() << "CUDA unavailable";
            // The loader is a process singleton that an earlier test's trainer may have switched to CPU
            // caching; its cache would keep serving the first mask.png after this test rewrites it.
            auto& cache = lfs::io::CacheLoader::getInstance(false);
            cache.reset_cache();
            cache.update_cache_params(false, 0, lfs::io::DEFAULT_MIN_FREE_GB, lfs::io::DEFAULT_MIN_FREE_MEMORY_RATIO,
                                      true, lfs::io::DEFAULT_PRINT_STATUS_FREQ);
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
        auto camera() {
            return std::make_shared<Camera>(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                                            110.f, 110.f, width / 2.f, height / 2.f, Tensor{}, Tensor{}, CameraModelType::PINHOLE,
                                            "image.png", temp.path / "image.png", temp.path / "mask.png", width, height, 0);
        }
    };
} // namespace

TEST_F(MaskColorConsistency, ColorAndGrayMasksUseTheTrainingLuminanceBeforeResampling) {
    const std::array<std::array<uint8_t, 3>, 6> bands{{{0, 255, 0}, {255, 0, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255}, {255, 255, 255}}};
    for (const bool gray : {false, true}) {
        std::vector<uint8_t> rgb(width * height * 3), values(width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                auto color = bands[x * bands.size() / width];
                if (gray)
                    color.fill(static_cast<uint8_t>((y * width + x) % 256));
                const size_t i = y * width + x;
                for (int c = 0; c < 3; ++c)
                    rgb[i * 3 + c] = color[c];
                values[i] = (77 * color[0] + 150 * color[1] + 29 * color[2]) >> 8;
            }
        }
        ASSERT_TRUE(save_png(temp.path / "mask.png", rgb.data(), width, height, 3, 8, 0));
        for (int resize : {1, 2, 4}) {
            auto cam = camera();
            for (bool invert : {false, true}) {
                for (float threshold : {.1f, .25f, .5f, .75f, .9f, 1.f}) {
                    SCOPED_TRACE(gray);
                    SCOPED_TRACE(resize);
                    SCOPED_TRACE(invert);
                    SCOPED_TRACE(threshold);
                    lfs::io::ImageRequest request;
                    request.path = temp.path / "image.png";
                    request.mask_path = temp.path / "mask.png";
                    request.params.resize_factor = resize;
                    request.mask_params = {.invert = invert, .threshold = threshold};
                    // Each comparison checks the decoded/resampled first-visit domain.
                    // Cache-grid agreement after resizing is covered separately.
                    lfs::io::PipelinedImageLoader loader(config());
                    loader.prefetch({request});
                    const auto ready = loader.get();
                    ASSERT_TRUE(ready.error.empty()) << ready.error;
                    ASSERT_TRUE(ready.mask);
                    const auto training = ready.mask->cpu().to_vector();
                    const auto evaluation = cam->load_and_get_mask(resize, 0, invert, threshold, true).cpu().to_vector_uint8();
                    ASSERT_EQ(training.size(), evaluation.size());
                    size_t mismatches = 0;
                    for (size_t i = 0; i < training.size(); ++i) {
                        mismatches += training[i] != evaluation[i];
                        if (resize == 1) {
                            const float value = static_cast<float>(static_cast<double>(values[i]) / 255.0);
                            const auto expected = (invert ? 1.f - value : value) >= threshold;
                            mismatches += (training[i] > 0.f) != expected;
                            mismatches += (evaluation[i] > 0) != expected;
                        }
                    }
                    EXPECT_EQ(mismatches, 0u);
                }
            }
        }
    }
}
