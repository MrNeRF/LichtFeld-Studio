/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace lfs::core;

TEST(TensorLazyGatherTest, MatchesTakeForDuplicatesAndNegativeIndices) {
    const auto input = Tensor::from_vector(
        std::vector<float>{-4.0f, 2.0f, -3.0f, 8.0f}, {4}, Device::CUDA);
    const auto indices = Tensor::from_vector(
        std::vector<int>{3, 1, 3, -1, 0}, {5}, Device::CUDA);

    const auto eager = input.take(indices);
    const auto lazy = input.gather_lazy(indices).eval();

    EXPECT_EQ(lazy.shape(), indices.shape());
    EXPECT_EQ(lazy.dtype(), input.dtype());
    EXPECT_EQ(lazy.device(), input.device());
    EXPECT_EQ(lazy.cpu().to_vector(),
              (std::vector<float>{8.0f, 2.0f, 8.0f, 8.0f, -4.0f}));
    EXPECT_TRUE(lazy.all_close(eager));
}

TEST(TensorLazyGatherTest, FusedUnaryMatchesEagerComposition) {
    const auto input = Tensor::from_vector(
        std::vector<float>{-4.0f, 2.0f, -3.0f, 8.0f}, {4}, Device::CUDA);
    const auto indices = Tensor::from_vector(
        std::vector<int>{2, -4, 1, 2}, {2, 2}, Device::CUDA);

    const auto eager = input.take(indices).abs();
    const auto fused = input.gather_lazy(indices).map(ops::abs_op{}).eval();

    EXPECT_EQ(fused.shape(), TensorShape({2, 2}));
    EXPECT_EQ(fused.cpu().to_vector(),
              (std::vector<float>{3.0f, 4.0f, 2.0f, 3.0f}));
    EXPECT_TRUE(fused.all_close(eager));
}

TEST(TensorLazyGatherTest, RejectsInvalidContractsBeforeDereference) {
    const auto input = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);
    const auto valid_indices = Tensor::from_vector(
        std::vector<int>{0}, {1}, Device::CUDA);
    const auto float_indices = Tensor::from_vector(
        std::vector<float>{0.0f}, {1}, Device::CUDA);
    const auto cpu_indices = valid_indices.cpu();
    const auto too_large = Tensor::from_vector(
        std::vector<int>{3}, {1}, Device::CUDA);
    const auto empty = Tensor::empty({0}, Device::CUDA, DataType::Float32);

    EXPECT_THROW((void)input.gather_lazy(float_indices), std::runtime_error);
    EXPECT_THROW((void)input.gather_lazy(cpu_indices), std::runtime_error);
    EXPECT_THROW(input.gather_lazy(too_large).eval(), std::runtime_error);
    EXPECT_THROW(input.gather_lazy(too_large).map(ops::abs_op{}).eval(), std::runtime_error);
    EXPECT_THROW(empty.gather_lazy(valid_indices).eval(), std::runtime_error);
    EXPECT_THROW(empty.gather_lazy(valid_indices).map(ops::abs_op{}).eval(), std::runtime_error);
}
