/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace lfs::core;

TEST(TensorOrderingTest, Float32SortReturnsValuesAndSourceIndices) {
    const auto input = Tensor::from_vector(
        std::vector<float>{3.0f, 1.0f, 4.0f, 2.0f}, {4}, Device::CUDA);

    const auto [ascending, ascending_indices] = input.sort(0);
    EXPECT_EQ(ascending_indices.dtype(), DataType::Int64);
    EXPECT_EQ(ascending.cpu().to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(ascending_indices.cpu().to_vector_int64(),
              (std::vector<int64_t>{1, 3, 0, 2}));

    const auto [descending, descending_indices] = input.sort(0, true);
    EXPECT_EQ(descending_indices.dtype(), DataType::Int64);
    EXPECT_EQ(descending.cpu().to_vector(), (std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f}));
    EXPECT_EQ(descending_indices.cpu().to_vector_int64(),
              (std::vector<int64_t>{2, 0, 3, 1}));
}

TEST(TensorOrderingTest, CpuSortPreservesInt64IndicesAcrossRanks) {
    const auto matrix = Tensor::from_vector(
        std::vector<float>{5.0f, 2.0f, 8.0f,
                           1.0f, 9.0f, 3.0f,
                           7.0f, 4.0f, 6.0f},
        {3, 3}, Device::CPU);
    const auto [column_values, column_indices] = matrix.sort(0);
    EXPECT_EQ(column_values.to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f,
                                  5.0f, 4.0f, 6.0f,
                                  7.0f, 9.0f, 8.0f}));
    EXPECT_EQ(column_indices.dtype(), DataType::Int64);
    EXPECT_EQ(column_indices.to_vector_int64(),
              (std::vector<int64_t>{1, 0, 1, 0, 2, 2, 2, 1, 0}));

    const auto volume = Tensor::from_vector(
        std::vector<float>{3.0f, 1.0f, 2.0f,
                           6.0f, 4.0f, 5.0f,
                           9.0f, 7.0f, 8.0f,
                           0.0f, -2.0f, -1.0f},
        {2, 2, 3}, Device::CPU);
    const auto [values, indices] = volume.sort(2);
    EXPECT_EQ(values.to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f,
                                  -2.0f, -1.0f, 0.0f}));
    EXPECT_EQ(indices.shape(), volume.shape());
    EXPECT_EQ(indices.dtype(), DataType::Int64);
    EXPECT_EQ(indices.to_vector_int64(),
              (std::vector<int64_t>{1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0}));
}

TEST(TensorOrderingTest, Float32SortAlongSecondDimension) {
    const auto input = Tensor::from_vector(
        std::vector<float>{3.0f, 1.0f, 2.0f, 4.0f, 6.0f, 5.0f},
        {2, 3}, Device::CUDA);

    const auto [values, indices] = input.sort(1, true);
    EXPECT_EQ(values.cpu().to_vector(),
              (std::vector<float>{3.0f, 2.0f, 1.0f, 6.0f, 5.0f, 4.0f}));
    EXPECT_EQ(indices.cpu().to_vector_int64(),
              (std::vector<int64_t>{0, 2, 1, 1, 2, 0}));
}

TEST(TensorOrderingTest, MinMaxWithIndicesReturnValuesAndLocations) {
    for (const auto device : {Device::CPU, Device::CUDA}) {
        const auto input = Tensor::from_vector(
            std::vector<float>{3.0f, 1.0f, 2.0f, 4.0f, 6.0f, 5.0f},
            {2, 3}, device);

        const auto [min_values, min_indices] = input.min_with_indices(1);
        const auto [max_values, max_indices] = input.max_with_indices(1);

        EXPECT_EQ(min_values.cpu().to_vector(), (std::vector<float>{1.0f, 4.0f}));
        EXPECT_EQ(min_indices.dtype(), DataType::Int64);
        EXPECT_EQ(min_indices.cpu().to_vector_int64(), (std::vector<int64_t>{1, 0}));
        EXPECT_EQ(max_values.cpu().to_vector(), (std::vector<float>{3.0f, 6.0f}));
        EXPECT_EQ(max_indices.dtype(), DataType::Int64);
        EXPECT_EQ(max_indices.cpu().to_vector_int64(), (std::vector<int64_t>{0, 1}));
    }
}

TEST(TensorDistanceTest, CdistL1AndL2HaveExactValues) {
    const auto lhs = Tensor::from_vector(
        std::vector<float>{0.0f, 0.0f, 3.0f, 4.0f}, {2, 2}, Device::CUDA);
    const auto rhs = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, -2.0f, 0.0f}, {2, 2}, Device::CUDA);

    const auto l1 = lhs.cdist(rhs, 1.0f).cpu().to_vector();
    EXPECT_EQ(l1, (std::vector<float>{3.0f, 2.0f, 4.0f, 9.0f}));

    const auto l2 = lhs.cdist(rhs, 2.0f).cpu().to_vector();
    ASSERT_EQ(l2.size(), 4u);
    EXPECT_NEAR(l2[0], std::sqrt(5.0f), 1e-5f);
    EXPECT_NEAR(l2[1], 2.0f, 1e-5f);
    EXPECT_NEAR(l2[2], std::sqrt(8.0f), 1e-5f);
    EXPECT_NEAR(l2[3], std::sqrt(41.0f), 1e-5f);
}
