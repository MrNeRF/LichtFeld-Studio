/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "io/pipelined_image_loader.hpp"
#include "training/dataset.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace lfs::core;
using namespace lfs::io;

namespace {

    class PipelinedImageLoaderTest : public ::testing::Test {
    protected:
        static std::uint64_t process_id() {
#ifdef _WIN32
            return static_cast<std::uint64_t>(_getpid());
#else
            return static_cast<std::uint64_t>(getpid());
#endif
        }

        static void SetUpTestSuite() {
            image_path_ = std::filesystem::path(TEST_DATA_DIR) /
                          "bicycle/images_4/_DSC8744.JPG";
            ASSERT_TRUE(std::filesystem::is_regular_file(image_path_)) << image_path_;
            mask_path_ = std::filesystem::temp_directory_path() /
                         ("lfs_pipelined_loader_bicycle_mask_" +
                          std::to_string(process_id()) + ".png");
            if (!std::filesystem::is_regular_file(mask_path_)) {
                auto [pixels, width, height, channels] = lfs::core::load_image(image_path_);
                const bool valid = pixels != nullptr && width > 0 && height > 0 && channels > 0;
                lfs::core::Tensor mask;
                if (valid) {
                    mask = lfs::core::Tensor::empty(
                        {static_cast<size_t>(height), static_cast<size_t>(width), size_t{1}},
                        lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                    auto* target = mask.ptr<uint8_t>();
                    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
                        const size_t x = i % static_cast<size_t>(width);
                        target[i] = width > 1
                                        ? static_cast<uint8_t>((255u * x) /
                                                               static_cast<size_t>(width - 1))
                                        : uint8_t{0};
                    }
                }
                if (pixels)
                    lfs::core::free_image(pixels);
                ASSERT_TRUE(valid) << image_path_;
                ASSERT_NO_THROW(lfs::core::save_image(mask_path_, std::move(mask)));
            }
            ASSERT_TRUE(std::filesystem::is_regular_file(mask_path_)) << mask_path_;
        }

        static void TearDownTestSuite() {
            std::error_code ec;
            std::filesystem::remove(mask_path_, ec);
        }

        void SetUp() override {
            ASSERT_TRUE(std::filesystem::is_regular_file(image_path_)) << image_path_;
            ASSERT_TRUE(std::filesystem::is_regular_file(mask_path_)) << mask_path_;
        }

        static PipelinedLoaderConfig config() {
            PipelinedLoaderConfig result;
            result.jpeg_batch_size = 2;
            result.prefetch_count = 4;
            result.output_queue_size = 4;
            result.decoder_pool_size = 2;
            result.io_threads = 1;
            result.cold_process_threads = 1;
            result.max_cache_bytes = 64 * 1024 * 1024;
            return result;
        }

        ImageRequest request(const size_t sequence_id,
                             const int max_width,
                             const bool with_mask = true) const {
            ImageRequest result;
            result.sequence_id = sequence_id;
            result.path = image_path_;
            result.params.resize_factor = 1;
            result.params.max_width = max_width;
            if (with_mask) {
                result.mask_path = mask_path_;
            }
            return result;
        }

        static std::filesystem::path image_path_;
        static std::filesystem::path mask_path_;
    };

    std::filesystem::path PipelinedImageLoaderTest::image_path_;
    std::filesystem::path PipelinedImageLoaderTest::mask_path_;

    std::vector<float> mask_values(const ReadyImage& ready) {
        EXPECT_TRUE(ready.mask.has_value());
        if (!ready.mask) {
            return {};
        }
        return ready.mask->cpu().to_vector();
    }

} // namespace

TEST_F(PipelinedImageLoaderTest, LoadsRealImageAndMaskWithExpectedContract) {
    PipelinedImageLoader loader(config());
    loader.prefetch({request(7, 128)});
    const auto ready = loader.get();

    EXPECT_TRUE(ready.error.empty()) << ready.error;
    EXPECT_EQ(ready.sequence_id, 7u);
    ASSERT_TRUE(ready.tensor.is_valid());
    EXPECT_EQ(ready.tensor.device(), Device::CUDA);
    EXPECT_EQ(ready.tensor.dtype(), DataType::Float32);
    ASSERT_EQ(ready.tensor.shape().rank(), 3u);
    EXPECT_EQ(ready.tensor.shape()[0], 3u);
    EXPECT_LE(std::max(ready.tensor.shape()[1], ready.tensor.shape()[2]), 128u);

    ASSERT_TRUE(ready.mask.has_value());
    EXPECT_EQ(ready.mask->device(), Device::CUDA);
    EXPECT_EQ(ready.mask->dtype(), DataType::Float32);
    EXPECT_EQ(ready.mask->shape(),
              TensorShape({ready.tensor.shape()[1], ready.tensor.shape()[2]}));
    EXPECT_GE(ready.mask->min().item<float>(), 0.0f);
    EXPECT_LE(ready.mask->max().item<float>(), 1.0f);
}

TEST_F(PipelinedImageLoaderTest, ResizeAndMaxWidthKeepImageAndMaskAligned) {
    PipelinedImageLoader loader(config());
    loader.prefetch({request(1, 256)});
    const auto large = loader.get();
    loader.prefetch({request(2, 96)});
    const auto small = loader.get();

    ASSERT_TRUE(large.mask.has_value());
    ASSERT_TRUE(small.mask.has_value());
    EXPECT_EQ(large.mask->shape(),
              TensorShape({large.tensor.shape()[1], large.tensor.shape()[2]}));
    EXPECT_EQ(small.mask->shape(),
              TensorShape({small.tensor.shape()[1], small.tensor.shape()[2]}));
    EXPECT_LE(std::max(small.tensor.shape()[1], small.tensor.shape()[2]), 96u);
    EXPECT_LT(small.tensor.shape()[1] * small.tensor.shape()[2],
              large.tensor.shape()[1] * large.tensor.shape()[2]);
}

TEST_F(PipelinedImageLoaderTest, MaskCacheHitPreservesInvertAndThresholdSemantics) {
    PipelinedImageLoader loader(config());

    constexpr float threshold = 0.25f;
    auto threshold_request = request(1, 0);
    threshold_request.mask_params.threshold = threshold;
    loader.prefetch({threshold_request});
    const auto thresholded_cold = loader.get();

    threshold_request.sequence_id = 2;
    loader.prefetch({threshold_request});
    const auto thresholded_hot = loader.get();

    auto normal_request = request(3, 0);
    loader.prefetch({normal_request});
    const auto normal = loader.get();

    auto inverted_request = request(4, 0);
    inverted_request.mask_params.invert = true;
    loader.prefetch({inverted_request});
    const auto inverted = loader.get();

    const auto normal_values = mask_values(normal);
    const auto inverted_values = mask_values(inverted);
    const auto thresholded_cold_values = mask_values(thresholded_cold);
    const auto thresholded_hot_values = mask_values(thresholded_hot);
    ASSERT_EQ(inverted_values.size(), normal_values.size());
    ASSERT_EQ(thresholded_cold_values.size(), normal_values.size());
    ASSERT_EQ(thresholded_hot_values.size(), normal_values.size());

    size_t zeros = 0;
    size_t ones = 0;
    for (size_t i = 0; i < normal_values.size(); ++i) {
        // UINT8 lossless J2K quantization is 1/255.
        EXPECT_NEAR(inverted_values[i], 1.0f - normal_values[i], 0.01f);
        const float expected = normal_values[i] >= threshold ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(thresholded_cold_values[i], expected);
        EXPECT_FLOAT_EQ(thresholded_hot_values[i], expected);
        zeros += thresholded_hot_values[i] == 0.0f;
        ones += thresholded_hot_values[i] == 1.0f;
    }
    EXPECT_GT(zeros, 0u);
    EXPECT_GT(ones, 0u);
}

TEST_F(PipelinedImageLoaderTest, MultipleRequestsPreserveIdsAndOptionalMask) {
    PipelinedImageLoader loader(config());
    loader.prefetch({request(11, 96, false), request(12, 96, true)});

    std::map<size_t, bool> mask_by_sequence;
    for (int i = 0; i < 2; ++i) {
        const auto ready = loader.get();
        EXPECT_TRUE(ready.error.empty()) << ready.error;
        ASSERT_TRUE(ready.tensor.is_valid());
        mask_by_sequence.emplace(ready.sequence_id, ready.mask.has_value());
    }

    EXPECT_EQ(mask_by_sequence,
              (std::map<size_t, bool>{{11u, false}, {12u, true}}));
}

TEST(SidecarResumeSampler, DeterministicCameraStreamContinuesAtCheckpointOffset) {
    constexpr std::uint64_t seed = 0x4c46535f73616d70ULL;
    constexpr size_t dataset_size = 17;
    constexpr size_t checkpoint_iteration = 31;

    lfs::training::InfiniteRandomSampler uninterrupted(dataset_size, seed, 0);
    lfs::training::InfiniteRandomSampler resumed(dataset_size, seed, checkpoint_iteration);

    for (size_t i = 0; i < 40; ++i) {
        const auto expected = uninterrupted.next(1);
        ASSERT_TRUE(expected.has_value());
        if (i < checkpoint_iteration)
            continue;
        const auto actual = resumed.next(1);
        ASSERT_TRUE(actual.has_value());
        EXPECT_EQ(*actual, *expected) << "camera stream diverged at post-resume sample " << i;
    }
}
