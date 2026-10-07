/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

using namespace lfs::core;

TEST(TensorAdvancedTest, LinspaceIncludesEndpointsAndZeroStepsAreEmpty) {
    const auto values = Tensor::linspace(-1.0f, 1.0f, 5, Device::GPU).cpu().to_vector();
    EXPECT_EQ(values, (std::vector<float>{-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}));
    const auto empty = Tensor::linspace(0.0f, 1.0f, 0, Device::GPU);
    EXPECT_EQ(empty.shape(), TensorShape({0}));
    EXPECT_TRUE(empty.is_empty());
}

TEST(TensorAdvancedTest, ArangeCreatesOnRequestedDevice) {
    const auto host = Tensor::arange(0.0f, 2.0f, 0.5f, Device::CPU);
    EXPECT_EQ(host.device(), Device::CPU);
    EXPECT_EQ(host.to_vector(), (std::vector<float>{0.0f, 0.5f, 1.0f, 1.5f}));

    const auto empty_host = Tensor::arange(3.0f, -2.0f, 1.0f, Device::CPU);
    EXPECT_EQ(empty_host.device(), Device::CPU);
    EXPECT_EQ(empty_host.shape(), TensorShape({0}));

    const auto device = Tensor::arange(0.0f, 2.0f, 0.5f, Device::GPU);
    EXPECT_EQ(device.device(), Device::GPU);
    EXPECT_EQ(device.cpu().to_vector(), host.to_vector());
}

TEST(TensorAdvancedTest, PositiveSteppedViewsPreserveStorageAndAliasing) {
    for (const auto device : {Device::CPU, Device::GPU}) {
        auto source = Tensor::arange(0.0f, 24.0f, 1.0f, device).reshape({2, 3, 4});
        auto view = source.slice(1, 0, 3, 2).slice(2, 1, 4, 2);
        EXPECT_EQ(view.shape(), TensorShape({2, 2, 2}));
        EXPECT_EQ(view.cpu().to_vector(), (std::vector<float>{1, 3, 9, 11, 13, 15, 21, 23}));
        view.fill_(17.0f);
        const auto values = source.cpu().to_vector();
        for (size_t i = 0; i < values.size(); ++i) {
            const bool selected = ((i / 4) % 3 != 1) && (i % 2 == 1);
            EXPECT_EQ(values[i], selected ? 17.0f : static_cast<float>(i));
        }
        source = Tensor{};
        EXPECT_EQ(view.cpu().to_vector(), std::vector<float>(8, 17.0f));
        const auto empty = view.slice(2, 2, 2, 3);
        EXPECT_EQ(empty.shape(), TensorShape({2, 2, 0}));
        EXPECT_TRUE(empty.is_empty());
    }
}

TEST(TensorAdvancedTest, StackPreservesValuesAndRejectsEmptyInput) {
    const auto first = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f}, {2}, Device::GPU);
    const auto second = Tensor::from_vector(
        std::vector<float>{3.0f, 4.0f}, {2}, Device::GPU);
    const auto result = Tensor::stack({first, second}, 0);

    EXPECT_EQ(result.shape(), TensorShape({2, 2}));
    EXPECT_EQ(result.cpu().to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_THROW(Tensor::stack({}, 0), std::runtime_error);
}

TEST(TensorAdvancedTest, ApplyAndInplaceChainsHaveDistinctOwnership) {
    auto input = Tensor::ones({4}, Device::GPU);
    const auto applied = input.apply([](const Tensor& tensor) { return tensor.add(1.0f); })
                             .apply([](const Tensor& tensor) { return tensor.mul(2.0f); })
                             .apply([](const Tensor& tensor) { return tensor.sub(1.0f); });

    EXPECT_EQ(input.cpu().to_vector(), (std::vector<float>(4, 1.0f)));
    EXPECT_EQ(applied.cpu().to_vector(), (std::vector<float>(4, 3.0f)));

    input.inplace([](Tensor& tensor) { tensor.add_(1.0f); })
        .inplace([](Tensor& tensor) { tensor.mul_(2.0f); })
        .inplace([](Tensor& tensor) { tensor.sub_(1.0f); });
    EXPECT_EQ(input.cpu().to_vector(), (std::vector<float>(4, 3.0f)));
}

TEST(TensorAdvancedTest, SpecialValuesAreDetectedAndClamped) {
    auto tensor = Tensor::from_vector(
        std::vector<float>{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            0.0f},
        {4}, Device::GPU);

    EXPECT_TRUE(tensor.has_nan());
    EXPECT_TRUE(tensor.has_inf());
    EXPECT_THROW(tensor.assert_finite(), TensorError);

    const auto clamped = tensor.clamp(-10.0f, 10.0f);
    EXPECT_TRUE(clamped.has_nan());
    EXPECT_FALSE(clamped.has_inf());
    const auto values = clamped.cpu().to_vector();
    EXPECT_FLOAT_EQ(values[1], 10.0f);
    EXPECT_FLOAT_EQ(values[2], -10.0f);
}

TEST(TensorAdvancedTest, DiagPlacesValuesOnlyOnDiagonal) {
    const auto diagonal = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::GPU);
    const auto matrix = Tensor::diag(diagonal);

    EXPECT_EQ(matrix.shape(), TensorShape({3, 3}));
    EXPECT_EQ(matrix.cpu().to_vector(),
              (std::vector<float>{1.0f, 0.0f, 0.0f,
                                  0.0f, 2.0f, 0.0f,
                                  0.0f, 0.0f, 3.0f}));
}

TEST(TensorAdvancedTest, ProfilingWrapperPreservesResult) {
    struct ProfilingGuard {
        ProfilingGuard() { Tensor::enable_profiling(true); }
        ~ProfilingGuard() { Tensor::enable_profiling(false); }
    } guard;

    const auto input = Tensor::ones({4}, Device::GPU);
    const auto result = input.timed("test_operation", [](const Tensor& tensor) {
        return tensor.add(1.0f).mul(2.0f).sub(1.0f);
    });

    EXPECT_EQ(result.cpu().to_vector(), (std::vector<float>{3.0f, 3.0f, 3.0f, 3.0f}));
}
