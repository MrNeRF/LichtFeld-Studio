/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Tests for stride-related bugs in non-contiguous tensor operations.
 * These tests verify that operations correctly handle sliced/view tensors
 * that have non-contiguous memory layouts.
 */

#include "core/tensor.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

class StrideBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 4x4 tensor with sequential values 1-16
        std::vector<float> data(16);
        for (int i = 0; i < 16; i++) {
            data[i] = static_cast<float>(i + 1);
        }
        base_tensor_ = Tensor::from_vector(data, {4, 4}, Device::CPU);

        // Create a 3x3 slice (non-contiguous view)
        // Original:     Slice (3x3):
        // 1  2  3  4    1  2  3
        // 5  6  7  8    5  6  7
        // 9 10 11 12    9 10 11
        // 13 14 15 16
        sliced_tensor_ = base_tensor_.slice(0, 0, 3).slice(1, 0, 3);
    }

    Tensor base_tensor_;
    Tensor sliced_tensor_;
};

// ============= set_bool / get_bool Tests =============

TEST_F(StrideBugTest, SetBool_NonContiguousSlice) {
    // Create a 4x4 bool tensor, all false
    auto bool_tensor = Tensor::zeros({4, 4}, Device::CPU, DataType::Bool);

    // Slice to 3x3
    auto slice = bool_tensor.slice(0, 0, 3).slice(1, 0, 3);
    ASSERT_FALSE(slice.is_contiguous());

    // Set element [1][1] to true via the slice
    // In the 4x4 tensor, this should be position [1][1] = element 5 (0-indexed: row*4+col = 1*4+1 = 5)
    slice.set_bool({1, 1}, true);

    // Verify via get_bool on the slice
    EXPECT_TRUE(slice.get_bool({1, 1})) << "set_bool/get_bool failed on non-contiguous tensor";

    // Verify the underlying data is correct
    // Element [1][1] in 4x4 should be true
    EXPECT_TRUE(bool_tensor.get_bool({1, 1})) << "Underlying tensor should have [1][1] set to true";

    // Element [1][3] should NOT be affected (this is what happens with wrong strides)
    EXPECT_FALSE(bool_tensor.get_bool({1, 3})) << "Element [1][3] should NOT be affected";
}

TEST_F(StrideBugTest, GetBool_NonContiguousSlice) {
    // Create a 4x4 bool tensor
    auto bool_tensor = Tensor::zeros({4, 4}, Device::CPU, DataType::Bool);

    // Set specific elements in the original tensor
    bool_tensor.set_bool({0, 0}, true);  // Linear idx 0
    bool_tensor.set_bool({1, 0}, true);  // Linear idx 4
    bool_tensor.set_bool({1, 1}, true);  // Linear idx 5
    bool_tensor.set_bool({2, 2}, true);  // Linear idx 10

    // Create a 3x3 slice
    auto slice = bool_tensor.slice(0, 0, 3).slice(1, 0, 3);
    ASSERT_FALSE(slice.is_contiguous());

    // Verify get_bool returns correct values on slice
    EXPECT_TRUE(slice.get_bool({0, 0}));   // Maps to [0][0]
    EXPECT_TRUE(slice.get_bool({1, 0}));   // Maps to [1][0]
    EXPECT_TRUE(slice.get_bool({1, 1}));   // Maps to [1][1]
    EXPECT_TRUE(slice.get_bool({2, 2}));   // Maps to [2][2]
    EXPECT_FALSE(slice.get_bool({0, 1}));  // Maps to [0][1]
    EXPECT_FALSE(slice.get_bool({2, 0}));  // Maps to [2][0]
}

// ============= index_put_ Tests =============

TEST_F(StrideBugTest, IndexPut_NonContiguousSlice) {
    // Create a 4x4 tensor initialized to zeros
    auto tensor = Tensor::zeros({4, 4}, Device::CPU, DataType::Float32);

    // Create a 3x3 slice
    auto slice = tensor.slice(0, 0, 3).slice(1, 0, 3);
    ASSERT_FALSE(slice.is_contiguous());

    // Use index_put_ to set values in the slice
    auto row_indices = Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::CPU);
    auto col_indices = Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{100.0f, 200.0f, 300.0f}, {3}, Device::CPU);

    slice.index_put_({row_indices, col_indices}, values);

    // Verify via direct access on the original tensor
    EXPECT_FLOAT_EQ(tensor[0][0], 100.0f) << "index_put_ should set [0][0] in slice -> [0][0] in original";
    EXPECT_FLOAT_EQ(tensor[1][1], 200.0f) << "index_put_ should set [1][1] in slice -> [1][1] in original";
    EXPECT_FLOAT_EQ(tensor[2][2], 300.0f) << "index_put_ should set [2][2] in slice -> [2][2] in original";

    // Verify that wrong positions are NOT affected
    // With bug: [1][1] in slice with strides [3,1] would compute offset 1*3+1=4, but correct is 1*4+1=5
    EXPECT_FLOAT_EQ(tensor[0][3], 0.0f) << "Element [0][3] should not be affected";
    EXPECT_FLOAT_EQ(tensor[1][3], 0.0f) << "Element [1][3] should not be affected";
}

// ============= nonzero Tests =============

TEST_F(StrideBugTest, Nonzero_NonContiguousSlice) {
    // Create a 4x4 tensor
    std::vector<float> data = {
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 4
    };
    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);

    // Create 3x3 slice - should contain:
    // 1, 0, 0
    // 0, 2, 0
    // 0, 0, 3
    auto slice = tensor.slice(0, 0, 3).slice(1, 0, 3);
    ASSERT_FALSE(slice.is_contiguous());

    // Find nonzero elements
    auto result = slice.nonzero();

    // Should find 3 nonzero elements at positions [0,0], [1,1], [2,2]
    ASSERT_EQ(result.shape()[0], 3) << "Should find 3 nonzero elements in slice";
    ASSERT_EQ(result.shape()[1], 2) << "Result should have 2 columns (row, col)";

    // Verify coordinates - should be diagonal elements
    // nonzero returns Int64, convert to float for easy access
    auto result_float = result.to(DataType::Float32).to(Device::CPU);

    // Check that nonzero found the correct positions
    std::vector<std::pair<float, float>> expected = {{0, 0}, {1, 1}, {2, 2}};
    for (size_t i = 0; i < 3; i++) {
        float row = result_float[i][0];
        float col = result_float[i][1];
        EXPECT_FLOAT_EQ(row, expected[i].first) << "Row mismatch at index " << i;
        EXPECT_FLOAT_EQ(col, expected[i].second) << "Col mismatch at index " << i;
    }
}

// ============= gather Tests =============

TEST_F(StrideBugTest, Gather_NonContiguousSlice) {
    // Create a 4x4 tensor with sequential values
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }
    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);

    // Create 3x3 slice:
    // 1  2  3
    // 5  6  7
    // 9 10 11
    auto slice = tensor.slice(0, 0, 3).slice(1, 0, 3);
    ASSERT_FALSE(slice.is_contiguous());

    // Gather along dimension 1 (columns) with indices [2, 1, 0] for each row
    auto indices = Tensor::from_vector(std::vector<int>{2, 1, 0, 2, 1, 0, 2, 1, 0}, {3, 3}, Device::CPU);

    auto result = slice.gather(1, indices);

    // Expected result after gather along dim 1:
    // Row 0: indices [2,1,0] from [1,2,3] -> [3,2,1]
    // Row 1: indices [2,1,0] from [5,6,7] -> [7,6,5]
    // Row 2: indices [2,1,0] from [9,10,11] -> [11,10,9]

    EXPECT_FLOAT_EQ(result[0][0], 3.0f);
    EXPECT_FLOAT_EQ(result[0][1], 2.0f);
    EXPECT_FLOAT_EQ(result[0][2], 1.0f);
    EXPECT_FLOAT_EQ(result[1][0], 7.0f);
    EXPECT_FLOAT_EQ(result[1][1], 6.0f);
    EXPECT_FLOAT_EQ(result[1][2], 5.0f);
    EXPECT_FLOAT_EQ(result[2][0], 11.0f);
    EXPECT_FLOAT_EQ(result[2][1], 10.0f);
    EXPECT_FLOAT_EQ(result[2][2], 9.0f);
}

// ============= Linear iteration bug tests =============

TEST_F(StrideBugTest, Any_NonContiguousSlice) {
    // Create 4x4 tensor of zeros
    auto tensor = Tensor::zeros({4, 4}, Device::CPU, DataType::Bool);

    // Set element at [2][2] to true
    tensor.set_bool({2, 2}, true);

    // Create 3x3 slice that includes [2][2]
    auto slice = tensor.slice(0, 0, 3).slice(1, 0, 3);
    ASSERT_FALSE(slice.is_contiguous());

    // any() should return true because [2][2] is in the slice
    EXPECT_TRUE(static_cast<bool>(slice.any().item())) << "any() should find the true value in sliced tensor";
}

TEST_F(StrideBugTest, All_NonContiguousSlice) {
    // Create 4x4 tensor of ones
    auto tensor = Tensor::ones({4, 4}, Device::CPU, DataType::Bool);

    // Create 3x3 slice - all should be true
    auto slice = tensor.slice(0, 0, 3).slice(1, 0, 3);
    ASSERT_FALSE(slice.is_contiguous());

    EXPECT_TRUE(static_cast<bool>(slice.all().item())) << "all() should return true for slice of all-ones tensor";

    // Now set one element in the slice to false
    tensor.set_bool({1, 1}, false);

    EXPECT_FALSE(static_cast<bool>(slice.all().item())) << "all() should return false after setting one element to false";
}

TEST_F(StrideBugTest, Sum_NonContiguousSlice) {
    // Use the pre-created sliced tensor
    // Slice values: 1,2,3,5,6,7,9,10,11
    // Sum = 1+2+3+5+6+7+9+10+11 = 54
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    auto sum_result = sliced_tensor_.sum();
    EXPECT_FLOAT_EQ(sum_result.item(), 54.0f) << "Sum of slice should be 54";
}

TEST_F(StrideBugTest, Mean_NonContiguousSlice) {
    // Slice values: 1,2,3,5,6,7,9,10,11
    // Mean = 54 / 9 = 6
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    auto mean_result = sliced_tensor_.mean();
    EXPECT_FLOAT_EQ(mean_result.item(), 6.0f) << "Mean of slice should be 6";
}

TEST_F(StrideBugTest, Max_NonContiguousSlice) {
    // Slice values: 1,2,3,5,6,7,9,10,11
    // Max = 11
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    auto max_result = sliced_tensor_.max();
    EXPECT_FLOAT_EQ(max_result.item(), 11.0f) << "Max of slice should be 11";
}

TEST_F(StrideBugTest, Min_NonContiguousSlice) {
    // Slice values: 1,2,3,5,6,7,9,10,11
    // Min = 1
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    auto min_result = sliced_tensor_.min();
    EXPECT_FLOAT_EQ(min_result.item(), 1.0f) << "Min of slice should be 1";
}

// ============= CUDA versions =============

TEST_F(StrideBugTest, CUDA_DirectAccess_NonContiguousSlice) {
    auto cuda_tensor = base_tensor_.to(Device::CUDA);
    auto slice = cuda_tensor.slice(0, 0, 3).slice(1, 0, 3);

    ASSERT_FALSE(slice.is_contiguous());

    // Test element access
    float val_0_0 = slice[0][0];
    float val_1_0 = slice[1][0];
    float val_1_1 = slice[1][1];
    float val_2_2 = slice[2][2];

    EXPECT_FLOAT_EQ(val_0_0, 1.0f);
    EXPECT_FLOAT_EQ(val_1_0, 5.0f);
    EXPECT_FLOAT_EQ(val_1_1, 6.0f);
    EXPECT_FLOAT_EQ(val_2_2, 11.0f);
}

TEST_F(StrideBugTest, CUDA_Sum_NonContiguousSlice) {
    auto cuda_tensor = base_tensor_.to(Device::CUDA);
    auto slice = cuda_tensor.slice(0, 0, 3).slice(1, 0, 3);

    ASSERT_FALSE(slice.is_contiguous());

    auto sum_result = slice.sum();
    EXPECT_FLOAT_EQ(sum_result.item(), 54.0f) << "CUDA sum of slice should be 54";
}

// ============= New stride bug tests (Round 2) =============

TEST_F(StrideBugTest, DebugValues_NonContiguousSlice) {
    // debug_values() iterates linearly - test that it handles non-contiguous tensors
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    // Get first 9 values from the slice
    auto values = sliced_tensor_.debug_values(9);

    // Should get the actual slice values: 1, 2, 3, 5, 6, 7, 9, 10, 11
    // Not the contiguous memory: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ASSERT_EQ(values.size(), 9);

    std::vector<float> expected = {1, 2, 3, 5, 6, 7, 9, 10, 11};
    for (size_t i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(values[i], expected[i])
            << "debug_values() mismatch at index " << i
            << " (expected " << expected[i] << ", got " << values[i] << ")";
    }
}

TEST_F(StrideBugTest, BroadcastTo_NonContiguousSource) {
    // broadcast_to() should handle non-contiguous source tensors
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    // Slice shape is [3, 3], broadcast to [2, 3, 3]
    auto result = sliced_tensor_.broadcast_to({2, 3, 3});

    ASSERT_EQ(result.shape()[0], 2);
    ASSERT_EQ(result.shape()[1], 3);
    ASSERT_EQ(result.shape()[2], 3);

    // Both copies should have the same values as the original slice
    for (size_t batch = 0; batch < 2; batch++) {
        auto batch_slice = result.slice(0, batch, batch + 1).squeeze(0);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[0][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[0][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[1][0]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[1][1]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[1][2]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[2][0]), 9.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[2][1]), 10.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[2][2]), 11.0f);
    }
}

TEST_F(StrideBugTest, Pad_NonContiguousInput) {
    // Pad() should handle non-contiguous input tensors
    // Pad is accessed via movement(MovementOp::Pad, ...)
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    // Pad the 3x3 slice by 1 on all sides -> 5x5 result
    MovementArgs pad_args;
    pad_args.args = std::vector<std::pair<int, int>>{{1, 1}, {1, 1}};
    auto result = sliced_tensor_.movement(MovementOp::Pad, pad_args);

    ASSERT_EQ(result.shape()[0], 5);
    ASSERT_EQ(result.shape()[1], 5);

    // Check that padding is zeros and original values are in the right place
    // Corners should be 0
    EXPECT_FLOAT_EQ(static_cast<float>(result[0][0]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result[0][4]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result[4][0]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result[4][4]), 0.0f);

    // Original data should be at [1:4, 1:4]
    EXPECT_FLOAT_EQ(static_cast<float>(result[1][1]), 1.0f);   // slice[0][0]
    EXPECT_FLOAT_EQ(static_cast<float>(result[1][2]), 2.0f);   // slice[0][1]
    EXPECT_FLOAT_EQ(static_cast<float>(result[1][3]), 3.0f);   // slice[0][2]
    EXPECT_FLOAT_EQ(static_cast<float>(result[2][1]), 5.0f);   // slice[1][0]
    EXPECT_FLOAT_EQ(static_cast<float>(result[2][2]), 6.0f);   // slice[1][1]
    EXPECT_FLOAT_EQ(static_cast<float>(result[2][3]), 7.0f);   // slice[1][2]
    EXPECT_FLOAT_EQ(static_cast<float>(result[3][1]), 9.0f);   // slice[2][0]
    EXPECT_FLOAT_EQ(static_cast<float>(result[3][2]), 10.0f);  // slice[2][1]
    EXPECT_FLOAT_EQ(static_cast<float>(result[3][3]), 11.0f);  // slice[2][2]
}

TEST_F(StrideBugTest, CalculateOffset_NonContiguousSlice) {
    // calculate_offset() is used internally by many operations
    // Test it indirectly by taking another slice of our non-contiguous slice
    ASSERT_FALSE(sliced_tensor_.is_contiguous());

    // Take another slice to get top-left 2x2 of the 3x3 slice
    auto sub_slice = sliced_tensor_.slice(0, 0, 2).slice(1, 0, 2);

    // Should get top-left 2x2 of the slice: [[1, 2], [5, 6]]
    EXPECT_FLOAT_EQ(static_cast<float>(sub_slice[0][0]), 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(sub_slice[0][1]), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(sub_slice[1][0]), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(sub_slice[1][1]), 6.0f);

    // Verify sum to ensure calculate_offset is used correctly in operations
    auto sum = sub_slice.sum();
    EXPECT_FLOAT_EQ(sum.item(), 1.0f + 2.0f + 5.0f + 6.0f);  // 14.0f
}

TEST_F(StrideBugTest, CUDA_BroadcastTo_NonContiguousSource) {
    auto cuda_tensor = base_tensor_.to(Device::CUDA);
    auto slice = cuda_tensor.slice(0, 0, 3).slice(1, 0, 3);

    ASSERT_FALSE(slice.is_contiguous());

    // Broadcast to [2, 3, 3]
    auto result = slice.broadcast_to({2, 3, 3});

    ASSERT_EQ(result.shape()[0], 2);
    ASSERT_EQ(result.shape()[1], 3);
    ASSERT_EQ(result.shape()[2], 3);

    // Verify values - use slice to get 2D views, then index into those
    for (size_t batch = 0; batch < 2; batch++) {
        auto batch_slice = result.slice(0, batch, batch + 1).squeeze(0);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[1][1]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(batch_slice[2][2]), 11.0f);
    }
}

TEST_F(StrideBugTest, CUDA_Pad_NonContiguousInput) {
    // Test that CUDA Pad works directly on non-contiguous input (no CPU fallback)
    auto cuda_tensor = base_tensor_.to(Device::CUDA);
    auto slice = cuda_tensor.slice(0, 0, 3).slice(1, 0, 3);

    ASSERT_FALSE(slice.is_contiguous());
    ASSERT_EQ(slice.device(), Device::CUDA);

    // Pad the 3x3 slice by 1 on all sides -> 5x5 result
    MovementArgs pad_args;
    pad_args.args = std::vector<std::pair<int, int>>{{1, 1}, {1, 1}};
    auto result = slice.movement(MovementOp::Pad, pad_args);

    ASSERT_EQ(result.shape()[0], 5);
    ASSERT_EQ(result.shape()[1], 5);
    ASSERT_EQ(result.device(), Device::CUDA);

    // Check corners are 0 (padding)
    EXPECT_FLOAT_EQ(static_cast<float>(result[0][0]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result[0][4]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result[4][0]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result[4][4]), 0.0f);

    // Check original data is in the right place
    EXPECT_FLOAT_EQ(static_cast<float>(result[1][1]), 1.0f);   // slice[0][0]
    EXPECT_FLOAT_EQ(static_cast<float>(result[2][2]), 6.0f);   // slice[1][1]
    EXPECT_FLOAT_EQ(static_cast<float>(result[3][3]), 11.0f);  // slice[2][2]
}

