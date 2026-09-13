/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tensor_zero_dimension.cpp
 * @brief Tests for zero-dimension tensor handling (sh-degree 0 bug)
 *
 * Tests tensors with zero-element dimensions like shape [N, 0, 3].
 * Original bug: index_put_ crashed with cudaErrorInvalidDevice.
 */

#include "core/tensor.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

class TensorZeroDimensionTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
    }
};

// Basic zero-dimension tensor creation

TEST_F(TensorZeroDimensionTest, CreateZerosWithZeroDimension) {
    // Shape [10, 0, 3] - has zero elements but valid shape
    auto t = Tensor::zeros({10, 0, 3}, Device::GPU);

    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape()[0], 10);
    EXPECT_EQ(t.shape()[1], 0);
    EXPECT_EQ(t.shape()[2], 3);
    EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, CreateEmptyWithZeroDimension) {
    auto t = Tensor::empty({5, 0, 4}, Device::GPU);

    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape()[0], 5);
    EXPECT_EQ(t.shape()[1], 0);
    EXPECT_EQ(t.shape()[2], 4);
    EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, CreateZeroFirstDimension) {
    auto t = Tensor::zeros({0, 5, 3}, Device::GPU);

    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, CreateZeroLastDimension) {
    auto t = Tensor::zeros({10, 5, 0}, Device::GPU);

    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 0);
}

// index_put_ with zero-element tensors (original crash case)

TEST_F(TensorZeroDimensionTest, IndexPutWithZeroElementValues) {
    // Target tensor: normal shape
    auto target = Tensor::zeros({100, 3}, Device::GPU);

    // Indices: some valid indices
    auto indices = Tensor::from_vector(std::vector<int>{0, 5, 10}, {3}, Device::GPU);

    // Values with zero elements in middle dimension (like ShN at sh-degree 0)
    auto values = Tensor::zeros({3, 0, 3}, Device::GPU);

    // This should be a no-op, not crash
    EXPECT_NO_THROW(target.index_put_(indices, values));
}

TEST_F(TensorZeroDimensionTest, IndexPutWithEmptyIndices) {
    auto target = Tensor::zeros({100, 4}, Device::GPU);

    // Empty indices tensor
    auto indices = Tensor::empty({0}, Device::GPU).to(DataType::Int32);

    // Values with matching shape
    auto values = Tensor::zeros({0, 4}, Device::GPU);

    // Should be a no-op
    EXPECT_NO_THROW(target.index_put_(indices, values));
}

TEST_F(TensorZeroDimensionTest, IndexPutZeroValuesPreservesTarget) {
    // Create target with known values
    std::vector<float> initial = {1, 2, 3, 4, 5, 6};
    auto target = Tensor::from_vector(initial, {2, 3}, Device::GPU);

    // Try to put zero-element values
    auto indices = Tensor::from_vector(std::vector<int>{0}, {1}, Device::GPU);
    auto values = Tensor::zeros({1, 0}, Device::GPU); // 0 elements

    // Should not modify target
    target.index_put_(indices, values);

    // Target should be unchanged
    auto result = target.to_vector();
    for (size_t i = 0; i < initial.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], initial[i]) << "Target modified at index " << i;
    }
}

// Simulating sh-degree 0 scenario (matches default_strategy.cpp)

TEST_F(TensorZeroDimensionTest, ShDegree0ScenarioOptimizerStateReset) {
    // Simulates split()'s reset_optimizer_state_at_indices with ShN shape [N, 0, 3]
    constexpr size_t NUM_GAUSSIANS = 1000;
    constexpr size_t NUM_SPLIT = 50;

    auto exp_avg = Tensor::zeros({NUM_GAUSSIANS, 0, 3}, Device::GPU);
    auto exp_avg_sq = Tensor::zeros({NUM_GAUSSIANS, 0, 3}, Device::GPU);

    std::vector<int> indices_vec(NUM_SPLIT);
    for (size_t i = 0; i < NUM_SPLIT; ++i) {
        indices_vec[i] = static_cast<int>(i * 10);
    }
    auto split_indices = Tensor::from_vector(indices_vec, {NUM_SPLIT}, Device::GPU)
                             .to(DataType::Int64);

    auto zeros = Tensor::zeros({NUM_SPLIT, 0, 3}, Device::GPU);

    EXPECT_NO_THROW({
        exp_avg.index_put_(split_indices, zeros);
        exp_avg_sq.index_put_(split_indices, zeros);
    });
}

TEST_F(TensorZeroDimensionTest, ShDegree0ScenarioDuplicate) {
    constexpr size_t NUM_GAUSSIANS = 500;
    constexpr size_t NUM_DUPLICATE = 30;

    auto shN = Tensor::zeros({NUM_GAUSSIANS, 0, 3}, Device::GPU);

    std::vector<int> src_indices_vec(NUM_DUPLICATE);
    for (size_t i = 0; i < NUM_DUPLICATE; ++i) {
        src_indices_vec[i] = static_cast<int>(i * 5);
    }
    auto src_indices = Tensor::from_vector(src_indices_vec, {NUM_DUPLICATE}, Device::GPU)
                           .to(DataType::Int64);

    EXPECT_NO_THROW({
        auto selected = shN.index_select(0, src_indices);
        EXPECT_EQ(selected.shape()[0], NUM_DUPLICATE);
        EXPECT_EQ(selected.shape()[1], 0);
        EXPECT_EQ(selected.shape()[2], 3);
        EXPECT_EQ(selected.numel(), 0);
    });
}

TEST_F(TensorZeroDimensionTest, ShDegree0ScenarioFillFreeSlots) {
    constexpr size_t NUM_GAUSSIANS = 1000;
    constexpr size_t NUM_FILL = 20;

    auto shN_target = Tensor::zeros({NUM_GAUSSIANS, 0, 3}, Device::GPU);

    std::vector<int> target_indices_vec(NUM_FILL);
    for (size_t i = 0; i < NUM_FILL; ++i) {
        target_indices_vec[i] = static_cast<int>(i * 10);
    }
    auto target_indices = Tensor::from_vector(target_indices_vec, {NUM_FILL}, Device::GPU)
                              .to(DataType::Int64);

    auto values = Tensor::zeros({NUM_FILL, 0, 3}, Device::GPU);

    EXPECT_NO_THROW(shN_target.index_put_(target_indices, values));
}

TEST_F(TensorZeroDimensionTest, ShDegree0ScenarioRemove) {
    constexpr size_t NUM_GAUSSIANS = 800;
    constexpr size_t NUM_PRUNE = 15;

    auto exp_avg = Tensor::zeros({NUM_GAUSSIANS, 0, 3}, Device::GPU);
    auto exp_avg_sq = Tensor::zeros({NUM_GAUSSIANS, 0, 3}, Device::GPU);

    std::vector<int> prune_indices_vec(NUM_PRUNE);
    for (size_t i = 0; i < NUM_PRUNE; ++i) {
        prune_indices_vec[i] = static_cast<int>(i * 20);
    }
    auto prune_indices = Tensor::from_vector(prune_indices_vec, {NUM_PRUNE}, Device::GPU)
                             .to(DataType::Int64);

    auto zeros = Tensor::zeros({NUM_PRUNE, 0, 3}, Device::GPU);

    EXPECT_NO_THROW({
        exp_avg.index_put_(prune_indices, zeros);
        exp_avg_sq.index_put_(prune_indices, zeros);
    });
}

// Edge cases

TEST_F(TensorZeroDimensionTest, MultipleZeroDimensions) {
    auto t = Tensor::zeros({0, 0, 0}, Device::GPU);
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, ZeroDimensionSlice) {
    auto t = Tensor::zeros({10, 0, 3}, Device::GPU);

    // Slicing should work
    auto sliced = t.slice(0, 0, 5);
    EXPECT_EQ(sliced.shape()[0], 5);
    EXPECT_EQ(sliced.shape()[1], 0);
    EXPECT_EQ(sliced.shape()[2], 3);
    EXPECT_EQ(sliced.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, ZeroDimensionReshape) {
    auto t = Tensor::zeros({10, 0, 3}, Device::GPU);

    // Reshape to different zero-element shape
    auto reshaped = t.reshape({5, 0, 6});
    EXPECT_EQ(reshaped.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, ZeroDimensionCat) {
    auto t1 = Tensor::zeros({5, 0, 3}, Device::GPU);
    auto t2 = Tensor::zeros({3, 0, 3}, Device::GPU);

    // Concatenating zero-element tensors
    auto cat_result = t1.cat(t2, 0);
    EXPECT_EQ(cat_result.shape()[0], 8);
    EXPECT_EQ(cat_result.shape()[1], 0);
    EXPECT_EQ(cat_result.shape()[2], 3);
    EXPECT_EQ(cat_result.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, ZeroDimensionAppendZeros) {
    auto t = Tensor::zeros({10, 0, 3}, Device::GPU);
    t.reserve(20);

    // Append zeros to zero-element tensor
    EXPECT_NO_THROW(t.append_zeros(5));

    EXPECT_EQ(t.shape()[0], 15);
    EXPECT_EQ(t.shape()[1], 0);
    EXPECT_EQ(t.shape()[2], 3);
    EXPECT_EQ(t.numel(), 0);
}

// Arithmetic operations on zero-element tensors

TEST_F(TensorZeroDimensionTest, ZeroDimensionArithmetic) {
    auto t1 = Tensor::zeros({10, 0, 3}, Device::GPU);
    auto t2 = Tensor::ones({10, 0, 3}, Device::GPU);

    // These should all be no-ops returning zero-element tensors
    EXPECT_NO_THROW({
        auto sum = t1 + t2;
        EXPECT_EQ(sum.numel(), 0);

        auto diff = t1 - t2;
        EXPECT_EQ(diff.numel(), 0);

        auto prod = t1 * t2;
        EXPECT_EQ(prod.numel(), 0);
    });
}

TEST_F(TensorZeroDimensionTest, ZeroDimensionReduction) {
    auto t = Tensor::zeros({10, 0, 3}, Device::GPU);

    // Sum of zero elements should be 0
    EXPECT_NO_THROW({
        auto sum = t.sum();
        // Result depends on implementation - could be 0 or empty tensor
    });
}

// Vector overload of index_put_

TEST_F(TensorZeroDimensionTest, IndexPutVectorOverloadZeroElements) {
    auto target = Tensor::zeros({100, 50}, Device::GPU);

    // Row and column indices
    auto row_idx = Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::GPU);
    auto col_idx = Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::GPU);

    // Zero-element values
    auto values = Tensor::zeros({0}, Device::GPU);

    std::vector<Tensor> indices = {row_idx, col_idx};

    // Should be a no-op with zero-element values
    EXPECT_NO_THROW(target.index_put_(indices, values));
}

// CPU device tests

TEST_F(TensorZeroDimensionTest, CPUZeroDimensionIndexPut) {
    auto target = Tensor::zeros({100, 3}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 5, 10}, {3}, Device::CPU);
    auto values = Tensor::zeros({3, 0}, Device::CPU);

    EXPECT_NO_THROW(target.index_put_(indices, values));
}

// Mixed operations

TEST_F(TensorZeroDimensionTest, ZeroDimensionToDevice) {
    auto t_cuda = Tensor::zeros({10, 0, 3}, Device::GPU);
    auto t_cpu = t_cuda.to(Device::CPU);

    EXPECT_EQ(t_cpu.device(), Device::CPU);
    EXPECT_EQ(t_cpu.numel(), 0);

    auto t_back = t_cpu.to(Device::GPU);
    EXPECT_EQ(t_back.device(), Device::GPU);
    EXPECT_EQ(t_back.numel(), 0);
}

TEST_F(TensorZeroDimensionTest, ZeroDimensionClone) {
    auto t = Tensor::zeros({10, 0, 3}, Device::GPU);
    auto cloned = t.clone();

    EXPECT_EQ(cloned.shape()[0], 10);
    EXPECT_EQ(cloned.shape()[1], 0);
    EXPECT_EQ(cloned.shape()[2], 3);
    EXPECT_EQ(cloned.numel(), 0);
}

// Stress test

TEST_F(TensorZeroDimensionTest, StressTestManyOperations) {
    constexpr int NUM_ITERATIONS = 100;
    constexpr size_t BASE_SIZE = 1000;
    constexpr size_t NUM_UPDATE = 20;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        const size_t size = BASE_SIZE + iter * 10;
        auto exp_avg = Tensor::zeros({size, 0, 3}, Device::GPU);
        auto exp_avg_sq = Tensor::zeros({size, 0, 3}, Device::GPU);

        std::vector<int> indices_vec(NUM_UPDATE);
        for (size_t i = 0; i < NUM_UPDATE; ++i) {
            indices_vec[i] = static_cast<int>((i * 7) % size);
        }
        auto indices = Tensor::from_vector(indices_vec, {NUM_UPDATE}, Device::GPU)
                           .to(DataType::Int64);

        auto zeros = Tensor::zeros({NUM_UPDATE, 0, 3}, Device::GPU);

        ASSERT_NO_THROW({
            exp_avg.index_put_(indices, zeros);
            exp_avg_sq.index_put_(indices, zeros);
        }) << "Failed at iteration "
           << iter;
    }
}
