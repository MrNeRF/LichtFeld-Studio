/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    struct Moments {
        double mean = 0.0;
        double variance = 0.0;
        float minimum = 0.0f;
        float maximum = 0.0f;
    };

    Moments moments(const std::vector<float>& values) {
        Moments result;
        result.minimum = *std::min_element(values.begin(), values.end());
        result.maximum = *std::max_element(values.begin(), values.end());
        for (const float value : values) {
            result.mean += value;
        }
        result.mean /= static_cast<double>(values.size());
        for (const float value : values) {
            result.variance += (value - result.mean) * (value - result.mean);
        }
        result.variance /= static_cast<double>(values.size());
        return result;
    }

    class TensorVulkanRandom : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }
    };

    TEST_F(TensorVulkanRandom, UniformStaysInsideTheHalfOpenRangeWithExpectedMoments) {
        // Catches a generator whose draws leave [low, high), one that reuses the
        // same block for every element, and a degenerate range that is not held.
        GpuBackendScope scope(GpuBackend::Vulkan);
        constexpr size_t count = 1000003;
        const Tensor draws = Tensor::uniform({count}, -2.0f, 2.0f, Device::GPU);
        EXPECT_EQ(gpu_backend_of(draws), GpuBackend::Vulkan);
        const std::vector<float> values = draws.cpu().to_vector();
        const Moments stats = moments(values);
        EXPECT_GE(stats.minimum, -2.0f);
        EXPECT_LT(stats.maximum, 2.0f);
        EXPECT_NEAR(stats.mean, 0.0, 0.01);
        EXPECT_NEAR(stats.variance, 4.0 / 3.0, 0.02);
        const std::set<float> distinct(values.begin(), values.begin() + 1000);
        EXPECT_GT(distinct.size(), 990u);
        const Tensor again = Tensor::uniform({count}, -2.0f, 2.0f, Device::GPU);
        EXPECT_NE(again.cpu().to_vector(), values) << "consecutive draws must advance the seed";
        const std::vector<float> constant = Tensor::uniform({7}, 3.0f, 3.0f, Device::GPU).cpu().to_vector();
        EXPECT_EQ(constant, std::vector<float>(7, 3.0f));
        Tensor in_place = Tensor::zeros({4099}, Device::GPU);
        in_place.uniform_(5.0f, 6.0f);
        const Moments in_place_stats = moments(in_place.cpu().to_vector());
        EXPECT_GE(in_place_stats.minimum, 5.0f);
        EXPECT_LT(in_place_stats.maximum, 6.0f);
        EXPECT_NEAR(in_place_stats.mean, 5.5, 0.02);
    }
    TEST_F(TensorVulkanRandom, RandintCoversTheHalfOpenRangeUniformly) {
        // Catches an off-by-one at either end of [low, high) and a biased bucket.
        GpuBackendScope scope(GpuBackend::Vulkan);
        constexpr size_t count = 1500000;
        const Tensor draws = Tensor::randint({count}, -7, 8, Device::GPU);
        ASSERT_EQ(draws.dtype(), DataType::Int32);
        const std::vector<float> values = draws.cpu().to(DataType::Float32).to_vector();
        std::array<size_t, 15> buckets{};
        for (const float value : values) {
            const int bucket = static_cast<int>(value) + 7;
            ASSERT_GE(bucket, 0);
            ASSERT_LT(bucket, 15);
            ++buckets[static_cast<size_t>(bucket)];
        }
        for (size_t bucket = 0; bucket < buckets.size(); ++bucket) {
            EXPECT_NEAR(static_cast<double>(buckets[bucket]) / count, 1.0 / 15.0, 0.003) << "bucket " << bucket;
        }
        const std::vector<float> single = Tensor::randint({33}, 4, 5, Device::GPU).cpu().to(DataType::Float32).to_vector();
        EXPECT_EQ(single, std::vector<float>(33, 4.0f));
    }

    TEST_F(TensorVulkanRandom, NormalMatchesMomentsForOddAndEvenCounts) {
        // Catches a Box-Muller with the wrong scale, a log of zero, and the odd
        // count path writing past the tensor or leaving the last element.
        GpuBackendScope scope(GpuBackend::Vulkan);
        for (const size_t count : {size_t{999999}, size_t{1000000}}) {
            Tensor draws = Tensor::zeros({count}, Device::GPU);
            draws.normal_(1.5f, 2.0f);
            const std::vector<float> values = draws.cpu().to_vector();
            ASSERT_EQ(values.size(), count);
            const Moments stats = moments(values);
            EXPECT_NEAR(stats.mean, 1.5, 0.01) << "n=" << count;
            EXPECT_NEAR(std::sqrt(stats.variance), 2.0, 0.01) << "n=" << count;
            size_t within = 0;
            for (const float value : values) {
                ASSERT_TRUE(std::isfinite(value));
                within += std::abs(value - 1.5f) <= 2.0f ? 1 : 0;
            }
            EXPECT_NEAR(static_cast<double>(within) / count, 0.6827, 0.005) << "n=" << count;
        }
        const std::vector<float> standard = Tensor::randn({4099}, Device::GPU).cpu().to_vector();
        const Moments stats = moments(standard);
        EXPECT_NEAR(stats.mean, 0.0, 0.06);
        EXPECT_NEAR(stats.variance, 1.0, 0.08);
    }
} // namespace
