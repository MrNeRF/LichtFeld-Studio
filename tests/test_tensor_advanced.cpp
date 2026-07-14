/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

using namespace lfs::core;

TEST(TensorAdvancedTest, LinspaceIncludesEndpointsAndRejectsZeroSteps) {
    const auto values = Tensor::linspace(-1.0f, 1.0f, 5, Device::CUDA).cpu().to_vector();
    EXPECT_EQ(values, (std::vector<float>{-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}));
    EXPECT_THROW(Tensor::linspace(0.0f, 1.0f, 0, Device::CUDA), std::runtime_error);
}

TEST(TensorAdvancedTest, StackPreservesValuesAndRejectsEmptyInput) {
    const auto first = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f}, {2}, Device::CUDA);
    const auto second = Tensor::from_vector(
        std::vector<float>{3.0f, 4.0f}, {2}, Device::CUDA);
    const auto result = Tensor::stack({first, second}, 0);

    EXPECT_EQ(result.shape(), TensorShape({2, 2}));
    EXPECT_EQ(result.cpu().to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_THROW(Tensor::stack({}, 0), std::runtime_error);
}

TEST(TensorAdvancedTest, TryReshapeDistinguishesCompatibleShapes) {
    const auto input = Tensor::arange(12.0f).to(Device::CUDA);

    const auto compatible = input.try_reshape({3, 4});
    const auto incompatible = input.try_reshape({5, 3});

    ASSERT_TRUE(compatible.has_value());
    EXPECT_EQ(compatible->shape(), TensorShape({3, 4}));
    EXPECT_EQ(compatible->cpu().to_vector(), input.cpu().to_vector());
    EXPECT_FALSE(incompatible.has_value());
}

TEST(TensorAdvancedTest, SplitBatchPreservesTailAndValues) {
    const auto input = Tensor::arange(1000.0f).to(Device::CUDA).reshape({100, 10});

    const auto batches = Tensor::split_batch(input, 32);

    ASSERT_EQ(batches.size(), 4u);
    EXPECT_EQ(batches[0].shape(), TensorShape({32, 10}));
    EXPECT_EQ(batches[1].shape(), TensorShape({32, 10}));
    EXPECT_EQ(batches[2].shape(), TensorShape({32, 10}));
    EXPECT_EQ(batches[3].shape(), TensorShape({4, 10}));
    EXPECT_EQ(Tensor::cat(batches, 0).cpu().to_vector(), input.cpu().to_vector());
}

TEST(TensorAdvancedTest, ApplyAndInplaceChainsHaveDistinctOwnership) {
    auto input = Tensor::ones({4}, Device::CUDA);
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
        {4}, Device::CUDA);

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
        std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);
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

    const auto input = Tensor::ones({4}, Device::CUDA);
    const auto result = input.timed("test_operation", [](const Tensor& tensor) {
        return tensor.add(1.0f).mul(2.0f).sub(1.0f);
    });

    EXPECT_EQ(result.cpu().to_vector(), (std::vector<float>{3.0f, 3.0f, 3.0f, 3.0f}));
}

TEST(TensorAdvancedTest, MetadataAssertionsRejectMismatches) {
    auto cuda_float = Tensor::ones({3, 4}, Device::CUDA, DataType::Float32);
    auto cpu_int = Tensor::zeros({3, 4}, Device::CPU, DataType::Int32);

    EXPECT_NO_THROW(cuda_float.assert_shape({3, 4}, "shape"));
    EXPECT_THROW(cuda_float.assert_shape({4, 3}, "shape"), TensorError);
    EXPECT_NO_THROW(cuda_float.assert_device(Device::CUDA));
    EXPECT_THROW(cuda_float.assert_device(Device::CPU), TensorError);
    EXPECT_NO_THROW(cpu_int.assert_dtype(DataType::Int32));
    EXPECT_THROW(cpu_int.assert_dtype(DataType::Float32), TensorError);
}
