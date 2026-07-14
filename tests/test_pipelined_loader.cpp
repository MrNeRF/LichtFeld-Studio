/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <vector>

using namespace lfs::core;
using namespace lfs::io;

namespace {

    class PipelinedImageLoaderTest : public ::testing::Test {
    protected:
        void SetUp() override {
            image_path_ = std::filesystem::path(TEST_DATA_DIR) /
                          "bicycle/images_4/_DSC8744.JPG";
            mask_path_ = std::filesystem::path(TEST_DATA_DIR) /
                         "bicycle/masks/_DSC8744.png";
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
            result.use_filesystem_cache = false;
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

        std::filesystem::path image_path_;
        std::filesystem::path mask_path_;
    };

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

TEST_F(PipelinedImageLoaderTest, CacheKeySeparatesInvertAndThresholdSemantics) {
    PipelinedImageLoader loader(config());

    auto normal_request = request(1, 96);
    loader.prefetch({normal_request});
    const auto normal = loader.get();

    auto inverted_request = request(2, 96);
    inverted_request.mask_params.invert = true;
    loader.prefetch({inverted_request});
    const auto inverted = loader.get();

    auto threshold_request = request(3, 96);
    threshold_request.mask_params.threshold = 0.5f;
    loader.prefetch({threshold_request});
    const auto thresholded = loader.get();

    const auto normal_values = mask_values(normal);
    const auto inverted_values = mask_values(inverted);
    const auto thresholded_values = mask_values(thresholded);
    ASSERT_EQ(inverted_values.size(), normal_values.size());
    ASSERT_EQ(thresholded_values.size(), normal_values.size());

    size_t zeros = 0;
    size_t ones = 0;
    for (size_t i = 0; i < normal_values.size(); ++i) {
        EXPECT_NEAR(inverted_values[i], 1.0f - normal_values[i], 1e-5f);
        const float expected = normal_values[i] >= 0.5f ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(thresholded_values[i], expected);
        zeros += thresholded_values[i] == 0.0f;
        ones += thresholded_values[i] == 1.0f;
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
