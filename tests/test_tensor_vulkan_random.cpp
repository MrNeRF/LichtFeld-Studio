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
        const Tensor draws = Tensor::uniform({count}, -2.0f, 2.0f, Device::CUDA);
        EXPECT_EQ(gpu_backend_of(draws), GpuBackend::Vulkan);
        const std::vector<float> values = draws.cpu().to_vector();
        const Moments stats = moments(values);
        EXPECT_GE(stats.minimum, -2.0f);
        EXPECT_LT(stats.maximum, 2.0f);
        EXPECT_NEAR(stats.mean, 0.0, 0.01);
        EXPECT_NEAR(stats.variance, 4.0 / 3.0, 0.02);
        const std::set<float> distinct(values.begin(), values.begin() + 1000);
        EXPECT_GT(distinct.size(), 990u);
        const Tensor again = Tensor::uniform({count}, -2.0f, 2.0f, Device::CUDA);
        EXPECT_NE(again.cpu().to_vector(), values) << "consecutive draws must advance the seed";
        const std::vector<float> constant = Tensor::uniform({7}, 3.0f, 3.0f, Device::CUDA).cpu().to_vector();
        EXPECT_EQ(constant, std::vector<float>(7, 3.0f));
        Tensor in_place = Tensor::zeros({4099}, Device::CUDA);
        in_place.uniform_(5.0f, 6.0f);
        const Moments in_place_stats = moments(in_place.cpu().to_vector());
        EXPECT_GE(in_place_stats.minimum, 5.0f);
        EXPECT_LT(in_place_stats.maximum, 6.0f);
        EXPECT_NEAR(in_place_stats.mean, 5.5, 0.02);
    }

    TEST_F(TensorVulkanRandom, BernoulliMatchesTheProbability) {
        // Catches a threshold applied to the wrong operand or a non-binary output.
        GpuBackendScope scope(GpuBackend::Vulkan);
        constexpr size_t count = 1000000;
        const std::vector<float> values = Tensor::bernoulli({count}, 0.35f, Device::CUDA).cpu().to_vector();
        size_t ones = 0;
        for (const float value : values) {
            ASSERT_TRUE(value == 0.0f || value == 1.0f);
            ones += value == 1.0f ? 1 : 0;
        }
        EXPECT_NEAR(static_cast<double>(ones) / count, 0.35, 0.003);
    }

    TEST_F(TensorVulkanRandom, RandintCoversTheHalfOpenRangeUniformly) {
        // Catches an off-by-one at either end of [low, high) and a biased bucket.
        GpuBackendScope scope(GpuBackend::Vulkan);
        constexpr size_t count = 1500000;
        const Tensor draws = Tensor::randint({count}, -7, 8, Device::CUDA);
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
        const std::vector<float> single = Tensor::randint({33}, 4, 5, Device::CUDA).cpu().to(DataType::Float32).to_vector();
        EXPECT_EQ(single, std::vector<float>(33, 4.0f));
    }

    TEST_F(TensorVulkanRandom, NormalMatchesMomentsForOddAndEvenCounts) {
        // Catches a Box-Muller with the wrong scale, a log of zero, and the odd
        // count path writing past the tensor or leaving the last element.
        GpuBackendScope scope(GpuBackend::Vulkan);
        for (const size_t count : {size_t{999999}, size_t{1000000}}) {
            Tensor draws = Tensor::zeros({count}, Device::CUDA);
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
        const std::vector<float> standard = Tensor::randn({4099}, Device::CUDA).cpu().to_vector();
        const Moments stats = moments(standard);
        EXPECT_NEAR(stats.mean, 0.0, 0.06);
        EXPECT_NEAR(stats.variance, 1.0, 0.08);
    }

    TEST_F(TensorVulkanRandom, MultinomialWithReplacementFollowsTheWeights) {
        // Catches a cumulative scan that skips the first category, a draw scaled
        // by the wrong total, and a zero-weight category that is still drawn.
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor weights = Tensor::from_vector({0.1f, 0.0f, 0.2f, 0.3f, 0.4f}, {5}, Device::CPU).to(Device::CUDA);
        constexpr int samples = 400000;
        const Tensor draws = Tensor::multinomial(weights, samples, true);
        ASSERT_EQ(draws.dtype(), DataType::Int64);
        const std::vector<float> values = draws.cpu().to(DataType::Float32).to_vector();
        std::array<size_t, 5> counts{};
        for (const float value : values) {
            const int category = static_cast<int>(value);
            ASSERT_GE(category, 0);
            ASSERT_LT(category, 5);
            ++counts[static_cast<size_t>(category)];
        }
        EXPECT_EQ(counts[1], 0u);
        const std::array expected{0.1, 0.0, 0.2, 0.3, 0.4};
        for (size_t category = 0; category < 5; ++category) {
            EXPECT_NEAR(static_cast<double>(counts[category]) / samples, expected[category], 0.004) << "category " << category;
        }
    }

    TEST_F(TensorVulkanRandom, MultinomialWithoutReplacementIsAPermutationFavouringHeavyWeights) {
        // Catches duplicate categories in a draw without replacement, a rank
        // selection that writes out of range, and keys that ignore the weights.
        GpuBackendScope scope(GpuBackend::Vulkan);
        constexpr size_t categories = 100;
        std::vector<float> host_weights(categories);
        for (size_t i = 0; i < categories; ++i) {
            host_weights[i] = static_cast<float>(i + 1);
        }
        const Tensor weights = Tensor::from_vector(host_weights, {categories}, Device::CPU).to(Device::CUDA);
        const std::vector<float> full = Tensor::multinomial(weights, categories, false).cpu().to(DataType::Float32).to_vector();
        const std::set<float> distinct(full.begin(), full.end());
        EXPECT_EQ(distinct.size(), categories);
        EXPECT_EQ(*distinct.begin(), 0.0f);
        EXPECT_EQ(*distinct.rbegin(), static_cast<float>(categories - 1));
        double heaviest_position = 0.0;
        double lightest_position = 0.0;
        constexpr int trials = 100;
        for (int trial = 0; trial < trials; ++trial) {
            const std::vector<float> draw = Tensor::multinomial(weights, 10, false).cpu().to(DataType::Float32).to_vector();
            ASSERT_EQ(draw.size(), 10u);
            EXPECT_EQ(std::set<float>(draw.begin(), draw.end()).size(), 10u) << "trial " << trial;
            heaviest_position += std::count_if(draw.begin(), draw.end(), [](const float v) { return v >= 90.0f; });
            lightest_position += std::count_if(draw.begin(), draw.end(), [](const float v) { return v < 10.0f; });
        }
        EXPECT_GT(heaviest_position, 4.0 * lightest_position + 1.0);
        const Tensor negative = Tensor::from_vector({1.0f, -1.0f, 2.0f}, {3}, Device::CPU).to(Device::CUDA);
        EXPECT_THROW(static_cast<void>(Tensor::multinomial(negative, 2, true)), std::exception);
    }

} // namespace
