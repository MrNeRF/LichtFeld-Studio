/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tensor_inplace_capacity.cpp
 * @brief Comprehensive tests for in-place tensor operations preserving capacity
 *
 * This test file verifies that ALL in-place tensor operations (methods ending with _)
 * properly preserve tensor properties like capacity, which is critical for in-place
 * growth operations used in training strategies.
 *
 * BUG CONTEXT: The index_put_() method was using "*this = cpu_tensor.to(device_)"
 * which REPLACED the entire tensor instead of copying data in-place, destroying
 * the pre-allocated capacity. This caused training failures during densification.
 */

#include <cmath>
#include <core/tensor.hpp>
#include <gtest/gtest.h>

using namespace lfs::core;

class TensorInplaceCapacityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a tensor with pre-allocated capacity
        capacity_ = 1000;
        initial_size_ = 100;

        // Create tensor with capacity using zeros_direct
        tensor_1d_ = Tensor::zeros_direct({initial_size_}, capacity_, Device::CUDA);
        tensor_2d_ = Tensor::zeros_direct({initial_size_, 3}, capacity_, Device::CUDA);
        tensor_3d_ = Tensor::zeros_direct({initial_size_, 4, 3}, capacity_, Device::CUDA);

        // Initialize with some values for testing
        tensor_1d_.fill_(1.0f);
        tensor_2d_.fill_(1.0f);
        tensor_3d_.fill_(1.0f);
    }

    void TearDown() override {
        // Clean up CUDA memory
        cudaDeviceSynchronize();
    }

    // Helper to verify capacity is preserved after an operation
    void verifyCapacityPreserved(const Tensor& tensor, size_t expected_capacity,
                                 const std::string& operation_name) {
        EXPECT_EQ(tensor.capacity(), expected_capacity)
            << operation_name << " destroyed capacity! "
            << "Expected: " << expected_capacity << ", Got: " << tensor.capacity();
    }

    // Helper to verify tensor is still valid and usable
    void verifyTensorValid(const Tensor& tensor, const std::string& operation_name) {
        EXPECT_TRUE(tensor.is_valid()) << operation_name << " invalidated tensor!";
        EXPECT_GT(tensor.numel(), 0) << operation_name << " resulted in empty tensor!";
    }

    size_t capacity_;
    size_t initial_size_;
    Tensor tensor_1d_;
    Tensor tensor_2d_;
    Tensor tensor_3d_;
};

// ============================================================================
// Basic in-place arithmetic operations (using Tensor overloads, not scalar)
// ============================================================================

TEST_F(TensorInplaceCapacityTest, AddInplaceTensor_PreservesCapacity) {
    auto other = Tensor::full({initial_size_, 3}, 2.0f, Device::CUDA);
    tensor_2d_.add_(other);
    verifyCapacityPreserved(tensor_2d_, capacity_, "add_(Tensor)");
    verifyTensorValid(tensor_2d_, "add_(Tensor)");

    float expected = 1.0f + 2.0f;
    EXPECT_NEAR(tensor_2d_.mean_scalar(), expected, 1e-5f);
}

TEST_F(TensorInplaceCapacityTest, SubInplaceTensor_PreservesCapacity) {
    auto other = Tensor::full({initial_size_, 3}, 0.5f, Device::CUDA);
    tensor_2d_.sub_(other);
    verifyCapacityPreserved(tensor_2d_, capacity_, "sub_(Tensor)");
    verifyTensorValid(tensor_2d_, "sub_(Tensor)");
}

TEST_F(TensorInplaceCapacityTest, MulInplaceTensor_PreservesCapacity) {
    auto other = Tensor::full({initial_size_, 3}, 2.0f, Device::CUDA);
    tensor_2d_.mul_(other);
    verifyCapacityPreserved(tensor_2d_, capacity_, "mul_(Tensor)");
    verifyTensorValid(tensor_2d_, "mul_(Tensor)");
}

TEST_F(TensorInplaceCapacityTest, DivInplaceTensor_PreservesCapacity) {
    auto other = Tensor::full({initial_size_, 3}, 2.0f, Device::CUDA);
    tensor_2d_.div_(other);
    verifyCapacityPreserved(tensor_2d_, capacity_, "div_(Tensor)");
    verifyTensorValid(tensor_2d_, "div_(Tensor)");
}

// ============================================================================
// Fill and zero operations
// ============================================================================

TEST_F(TensorInplaceCapacityTest, ZeroInplace_PreservesCapacity) {
    tensor_2d_.zero_();
    verifyCapacityPreserved(tensor_2d_, capacity_, "zero_()");
    verifyTensorValid(tensor_2d_, "zero_()");

    EXPECT_NEAR(tensor_2d_.sum_scalar(), 0.0f, 1e-5f);
}

TEST_F(TensorInplaceCapacityTest, FillInplace_PreservesCapacity) {
    tensor_2d_.fill_(42.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "fill_(value)");
    verifyTensorValid(tensor_2d_, "fill_(value)");

    EXPECT_NEAR(tensor_2d_.mean_scalar(), 42.0f, 1e-5f);
}

// ============================================================================
// Copy operations
// ============================================================================

TEST_F(TensorInplaceCapacityTest, CopyFrom_PreservesCapacity) {
    auto src = Tensor::full({initial_size_, 3}, 99.0f, Device::CUDA);
    tensor_2d_.copy_from(src);
    verifyCapacityPreserved(tensor_2d_, capacity_, "copy_from()");
    verifyTensorValid(tensor_2d_, "copy_from()");

    EXPECT_NEAR(tensor_2d_.mean_scalar(), 99.0f, 1e-5f);
}

TEST_F(TensorInplaceCapacityTest, CopyInplace_PreservesCapacity) {
    auto src = Tensor::full({initial_size_, 3}, 77.0f, Device::CUDA);
    tensor_2d_.copy_(src);
    verifyCapacityPreserved(tensor_2d_, capacity_, "copy_()");
    verifyTensorValid(tensor_2d_, "copy_()");

    EXPECT_NEAR(tensor_2d_.mean_scalar(), 77.0f, 1e-5f);
}

// ============================================================================
// Random operations
// ============================================================================

TEST_F(TensorInplaceCapacityTest, UniformInplace_PreservesCapacity) {
    tensor_2d_.uniform_(0.0f, 1.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "uniform_()");
    verifyTensorValid(tensor_2d_, "uniform_()");

    // Values should be in [0, 1]
    EXPECT_GE(tensor_2d_.min_scalar(), 0.0f);
    EXPECT_LE(tensor_2d_.max_scalar(), 1.0f);
}

TEST_F(TensorInplaceCapacityTest, NormalInplace_PreservesCapacity) {
    tensor_2d_.normal_(0.0f, 1.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "normal_()");
    verifyTensorValid(tensor_2d_, "normal_()");
}

// ============================================================================
// Clamping operations
// ============================================================================

TEST_F(TensorInplaceCapacityTest, ClampInplace_PreservesCapacity) {
    tensor_2d_.uniform_(-10.0f, 10.0f);
    tensor_2d_.clamp_(-1.0f, 1.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "clamp_()");
    verifyTensorValid(tensor_2d_, "clamp_()");

    EXPECT_GE(tensor_2d_.min_scalar(), -1.0f);
    EXPECT_LE(tensor_2d_.max_scalar(), 1.0f);
}

TEST_F(TensorInplaceCapacityTest, ClampMinInplace_PreservesCapacity) {
    tensor_2d_.uniform_(-10.0f, 10.0f);
    tensor_2d_.clamp_min_(0.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "clamp_min_()");
    verifyTensorValid(tensor_2d_, "clamp_min_()");

    EXPECT_GE(tensor_2d_.min_scalar(), 0.0f);
}

TEST_F(TensorInplaceCapacityTest, ClampMaxInplace_PreservesCapacity) {
    tensor_2d_.uniform_(-10.0f, 10.0f);
    tensor_2d_.clamp_max_(0.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "clamp_max_()");
    verifyTensorValid(tensor_2d_, "clamp_max_()");

    EXPECT_LE(tensor_2d_.max_scalar(), 0.0f);
}

// ============================================================================
// Masking operations
// ============================================================================

TEST_F(TensorInplaceCapacityTest, MaskedFillInplace_PreservesCapacity) {
    // Create a mask (first half true, second half false)
    auto mask = Tensor::zeros_bool({initial_size_, 3}, Device::CUDA);
    auto cpu_mask = mask.cpu();
    for (size_t i = 0; i < initial_size_ / 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            cpu_mask.set_bool({i, j}, true);
        }
    }
    mask = cpu_mask.cuda();

    tensor_2d_.masked_fill_(mask, 999.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "masked_fill_()");
    verifyTensorValid(tensor_2d_, "masked_fill_()");
}

// ============================================================================
// Indexing operations - THE CRITICAL ONES (index_put_ was the original bug)
// ============================================================================

TEST_F(TensorInplaceCapacityTest, IndexPutInplace_PreservesCapacity) {
    // This was the original bug - index_put_() was replacing the tensor
    auto indices = Tensor::from_vector({0, 1, 2, 3, 4}, {5}, Device::CUDA).to(DataType::Int32);
    auto values = Tensor::full({5, 3}, 123.0f, Device::CUDA);

    tensor_2d_.index_put_(indices, values);
    verifyCapacityPreserved(tensor_2d_, capacity_, "index_put_(indices, values)");
    verifyTensorValid(tensor_2d_, "index_put_(indices, values)");

    // Verify the values were actually set
    auto first_row = tensor_2d_.slice(0, 0, 1).cpu();
    EXPECT_NEAR(first_row.mean_scalar(), 123.0f, 1e-5f);
}

TEST_F(TensorInplaceCapacityTest, IndexPutMultiDim_PreservesCapacity) {
    // 2D indexing with row and column indices
    auto tensor = Tensor::zeros_direct({100, 50}, 1000, Device::CUDA);
    tensor.fill_(1.0f);

    auto row_idx = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);
    auto col_idx = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);
    auto values = Tensor::full({3}, 999.0f, Device::CUDA);

    tensor.index_put_({row_idx, col_idx}, values);
    verifyCapacityPreserved(tensor, 1000, "index_put_(vector<Tensor>, values)");
    verifyTensorValid(tensor, "index_put_(vector<Tensor>, values)");
}

TEST_F(TensorInplaceCapacityTest, ScatterInplace_PreservesCapacity) {
    auto indices = Tensor::from_vector({0, 5, 10}, {3}, Device::CUDA).to(DataType::Int32);
    auto src = Tensor::full({3, 3}, 777.0f, Device::CUDA);

    tensor_2d_.scatter_(0, indices, src);
    verifyCapacityPreserved(tensor_2d_, capacity_, "scatter_()");
    verifyTensorValid(tensor_2d_, "scatter_()");
}

TEST_F(TensorInplaceCapacityTest, IndexFillInplace_PreservesCapacity) {
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);

    tensor_2d_.index_fill_(0, indices, 555.0f);
    verifyCapacityPreserved(tensor_2d_, capacity_, "index_fill_()");
    verifyTensorValid(tensor_2d_, "index_fill_()");
}

TEST_F(TensorInplaceCapacityTest, IndexCopyInplace_PreservesCapacity) {
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);
    auto src = Tensor::full({3, 3}, 444.0f, Device::CUDA);

    tensor_2d_.index_copy_(0, indices, src);
    verifyCapacityPreserved(tensor_2d_, capacity_, "index_copy_()");
    verifyTensorValid(tensor_2d_, "index_copy_()");
}

TEST_F(TensorInplaceCapacityTest, IndexAddInplace_PreservesCapacity) {
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);
    auto src = Tensor::full({3, 3}, 333.0f, Device::CUDA);

    tensor_2d_.index_add_(0, indices, src);
    verifyCapacityPreserved(tensor_2d_, capacity_, "index_add_()");
    verifyTensorValid(tensor_2d_, "index_add_()");
}

// ============================================================================
// Growth operations that depend on capacity
// ============================================================================

TEST_F(TensorInplaceCapacityTest, AppendGather_PreservesCapacity) {
    auto indices = Tensor::from_vector({0, 1, 2, 3, 4}, {5}, Device::CUDA).to(DataType::Int32);

    size_t old_size = tensor_2d_.shape()[0];
    tensor_2d_.append_gather(indices);

    verifyCapacityPreserved(tensor_2d_, capacity_, "append_gather()");
    verifyTensorValid(tensor_2d_, "append_gather()");
    EXPECT_EQ(tensor_2d_.shape()[0], old_size + 5);
}

TEST_F(TensorInplaceCapacityTest, AppendGatherCopiesSelectedRowsExactly) {
    auto tensor = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f,
                           4.0f, 5.0f, 6.0f},
        {2, 3}, Device::CUDA);
    tensor.reserve(8);
    const auto indices = Tensor::from_vector(
                             std::vector<int>{0, 1, 0}, {3}, Device::CUDA)
                             .to(DataType::Int32);

    tensor.append_gather(indices);

    EXPECT_EQ(tensor.shape(), TensorShape({5, 3}));
    EXPECT_EQ(tensor.cpu().to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  1.0f, 2.0f, 3.0f}));
}

TEST_F(TensorInplaceCapacityTest, AppendZeros_PreservesCapacity) {
    size_t old_size = tensor_2d_.shape()[0];
    tensor_2d_.append_zeros(10);

    verifyCapacityPreserved(tensor_2d_, capacity_, "append_zeros()");
    verifyTensorValid(tensor_2d_, "append_zeros()");
    EXPECT_EQ(tensor_2d_.shape()[0], old_size + 10);
}

// ============================================================================
// Chain of operations - real world usage pattern
// ============================================================================

TEST_F(TensorInplaceCapacityTest, ChainedOperations_PreserveCapacity) {
    // This simulates what happens in training: multiple in-place operations
    // followed by growth operations

    // Step 1: Various in-place operations
    tensor_2d_.fill_(1.0f);
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "fill_ broke capacity";

    auto mult = Tensor::full({initial_size_, 3}, 2.0f, Device::CUDA);
    tensor_2d_.mul_(mult);
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "mul_ broke capacity";

    auto add = Tensor::full({initial_size_, 3}, 0.5f, Device::CUDA);
    tensor_2d_.add_(add);
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "add_ broke capacity";

    // Step 2: Indexed operations (the ones that historically had bugs)
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);
    auto values = Tensor::full({3, 3}, 10.0f, Device::CUDA);

    tensor_2d_.index_put_(indices, values);
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "index_put_ broke capacity";

    // Step 3: Growth operations that depend on capacity
    tensor_2d_.append_zeros(50);
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "append_zeros broke capacity";

    auto more_indices = Tensor::from_vector({0, 1, 2, 3, 4}, {5}, Device::CUDA).to(DataType::Int32);
    tensor_2d_.append_gather(more_indices);
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "append_gather broke capacity";

    // Final verification
    EXPECT_EQ(tensor_2d_.shape()[0], initial_size_ + 50 + 5);
    EXPECT_TRUE(tensor_2d_.is_valid());
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(TensorInplaceCapacityTest, IndexPut_WithInt64Indices_PreservesCapacity) {
    // Create Int32 indices first, then convert to Int64
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int64);
    auto values = Tensor::full({3, 3}, 888.0f, Device::CUDA);

    tensor_2d_.index_put_(indices, values);
    verifyCapacityPreserved(tensor_2d_, capacity_, "index_put_(Int64 indices)");
    verifyTensorValid(tensor_2d_, "index_put_(Int64 indices)");
}

TEST_F(TensorInplaceCapacityTest, MultipleIndexPuts_PreservesCapacity) {
    // Repeatedly calling index_put_ should preserve capacity each time
    for (int i = 0; i < 10; i++) {
        auto indices = Tensor::from_vector({i * 5, i * 5 + 1, i * 5 + 2}, {3}, Device::CUDA).to(DataType::Int32);
        auto values = Tensor::full({3, 3}, static_cast<float>(i), Device::CUDA);

        tensor_2d_.index_put_(indices, values);
        verifyCapacityPreserved(tensor_2d_, capacity_, "index_put_ iteration " + std::to_string(i));
    }
}

// ============================================================================
// 3D tensor tests (different memory layout)
// ============================================================================

TEST_F(TensorInplaceCapacityTest, IndexPut3D_PreservesCapacity) {
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);
    auto values = Tensor::full({3, 4, 3}, 111.0f, Device::CUDA);

    tensor_3d_.index_put_(indices, values);
    verifyCapacityPreserved(tensor_3d_, capacity_, "index_put_() on 3D tensor");
    verifyTensorValid(tensor_3d_, "index_put_() on 3D tensor");
}

TEST_F(TensorInplaceCapacityTest, AppendGather3D_PreservesCapacity) {
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);

    tensor_3d_.append_gather(indices);
    verifyCapacityPreserved(tensor_3d_, capacity_, "append_gather() on 3D tensor");
    verifyTensorValid(tensor_3d_, "append_gather() on 3D tensor");
}

// ============================================================================
// CPU tensor tests - Note: zeros_direct() only supports CUDA, so CPU tensors
// don't have capacity. These tests verify CPU in-place operations work correctly
// but don't check capacity (since it's N/A for CPU tensors).
// ============================================================================

TEST_F(TensorInplaceCapacityTest, CPU_IndexPut_Works) {
    // CPU tensors don't support capacity, just verify in-place operations work
    auto cpu_tensor = Tensor::zeros({initial_size_, 3}, Device::CPU);
    cpu_tensor.fill_(1.0f);

    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CPU).to(DataType::Int32);
    auto values = Tensor::full({3, 3}, 222.0f, Device::CPU);

    cpu_tensor.index_put_(indices, values);
    verifyTensorValid(cpu_tensor, "CPU index_put_()");

    // Verify the operation worked
    auto first_row = cpu_tensor.slice(0, 0, 1);
    EXPECT_NEAR(first_row.mean_scalar(), 222.0f, 1e-5f);
}

TEST_F(TensorInplaceCapacityTest, CPU_AllInplaceOps_Work) {
    // CPU tensors don't support capacity, just verify operations work
    auto cpu_tensor = Tensor::zeros({initial_size_, 3}, Device::CPU);
    cpu_tensor.fill_(1.0f);

    auto add_t = Tensor::full({initial_size_, 3}, 1.0f, Device::CPU);
    cpu_tensor.add_(add_t);
    EXPECT_NEAR(cpu_tensor.mean_scalar(), 2.0f, 1e-5f);

    auto mul_t = Tensor::full({initial_size_, 3}, 2.0f, Device::CPU);
    cpu_tensor.mul_(mul_t);
    EXPECT_NEAR(cpu_tensor.mean_scalar(), 4.0f, 1e-5f);

    cpu_tensor.zero_();
    EXPECT_NEAR(cpu_tensor.sum_scalar(), 0.0f, 1e-5f);

    cpu_tensor.fill_(5.0f);
    EXPECT_NEAR(cpu_tensor.mean_scalar(), 5.0f, 1e-5f);

    cpu_tensor.clamp_(0.0f, 3.0f);
    EXPECT_NEAR(cpu_tensor.max_scalar(), 3.0f, 1e-5f);
}

// ============================================================================
// Regression test for the original bug
// ============================================================================

// ============================================================================
// View operations - THESE SHOULD NOT PRESERVE CAPACITY (expected behavior)
// These tests document the expected behavior for views
// ============================================================================

TEST_F(TensorInplaceCapacityTest, ViewOperations_DoNotPreserveCapacity) {
    // Views share data but don't have capacity - this is EXPECTED behavior
    // These tests document this so users know to be careful

    // slice() creates a view without capacity
    auto slice_view = tensor_2d_.slice(0, 0, 50);
    EXPECT_EQ(slice_view.capacity(), 0) << "slice() should NOT preserve capacity (expected)";

    // But original tensor should still have capacity
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "Original tensor should keep capacity after creating slice view";

    // squeeze() creates a view without capacity
    auto tensor_with_dim1 = Tensor::zeros_direct({initial_size_, 1, 3}, capacity_, Device::CUDA);
    auto squeezed = tensor_with_dim1.squeeze(1);
    EXPECT_EQ(squeezed.capacity(), 0) << "squeeze() should NOT preserve capacity (expected)";
    EXPECT_EQ(tensor_with_dim1.capacity(), capacity_) << "Original tensor should keep capacity";

    // unsqueeze() creates a view without capacity
    auto unsqueezed = tensor_2d_.unsqueeze(0);
    EXPECT_EQ(unsqueezed.capacity(), 0) << "unsqueeze() should NOT preserve capacity (expected)";
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "Original tensor should keep capacity";
}

TEST_F(TensorInplaceCapacityTest, DANGER_SelfAssignmentThroughView_LosesCapacity) {
    // This test documents a DANGEROUS pattern that loses capacity
    // Users should NEVER do: tensor = tensor.some_view_operation()

    auto test_tensor = Tensor::zeros_direct({initial_size_, 3}, capacity_, Device::CUDA);
    test_tensor.fill_(1.0f);
    EXPECT_EQ(test_tensor.capacity(), capacity_);

    // DANGEROUS: Self-assignment through a view loses capacity!
    // This is what caused the reset_opacity() bug
    // test_tensor = test_tensor.clamp(0.0f, 0.5f);  // DON'T DO THIS!

    // CORRECT: Use in-place version to preserve capacity
    test_tensor.clamp_(0.0f, 0.5f);
    EXPECT_EQ(test_tensor.capacity(), capacity_) << "In-place clamp_ should preserve capacity";
}

TEST_F(TensorInplaceCapacityTest, DANGER_NonInplaceOperations_CreateNewTensors) {
    // Non-in-place operations create NEW tensors without capacity
    // Users must use in-place versions when capacity matters

    // clamp() returns new tensor without capacity
    auto clamped = tensor_2d_.clamp(0.0f, 0.5f);
    EXPECT_EQ(clamped.capacity(), 0) << "clamp() returns new tensor without capacity";
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "Original should keep capacity";

    // add() returns new tensor without capacity
    auto added = tensor_2d_.add(1.0f);
    EXPECT_EQ(added.capacity(), 0) << "add() returns new tensor without capacity";
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "Original should keep capacity";

    // index_select() returns new tensor without capacity
    auto indices = Tensor::from_vector({0, 1, 2}, {3}, Device::CUDA).to(DataType::Int32);
    auto selected = tensor_2d_.index_select(0, indices);
    EXPECT_EQ(selected.capacity(), 0) << "index_select() returns new tensor without capacity";
    EXPECT_EQ(tensor_2d_.capacity(), capacity_) << "Original should keep capacity";

    // contiguous() on a view returns new tensor without capacity
    auto slice_view = tensor_2d_.slice(0, 0, 50);
    auto contig = slice_view.contiguous();
    EXPECT_EQ(contig.capacity(), 0) << "contiguous() returns new tensor without capacity";
}

TEST_F(TensorInplaceCapacityTest, SAFE_CopyFromPreservesCapacity) {
    // copy_from() is the SAFE way to update a tensor's data while preserving capacity
    auto test_tensor = Tensor::zeros_direct({initial_size_, 3}, capacity_, Device::CUDA);

    // Create source data (new tensor, no capacity)
    auto source = Tensor::full({initial_size_, 3}, 42.0f, Device::CUDA);
    EXPECT_EQ(source.capacity(), 0);

    // copy_from() preserves destination's capacity
    test_tensor.copy_from(source);
    EXPECT_EQ(test_tensor.capacity(), capacity_) << "copy_from() should preserve capacity";
    EXPECT_NEAR(test_tensor.mean_scalar(), 42.0f, 1e-5f);
}

// ============================================================================
// Regression test for the original bug
// ============================================================================

TEST_F(TensorInplaceCapacityTest, REGRESSION_IndexPut_ThenAppendZeros) {
    // This is the exact sequence that exposed the original bug:
    // 1. Create tensor with capacity
    // 2. Use index_put_() to fill free slots
    // 3. Attempt to grow with append_zeros() - FAILED because capacity was 0

    size_t n_gaussians = 100;
    size_t max_capacity = 1000;

    auto means = Tensor::zeros_direct({n_gaussians, 3}, max_capacity, Device::CUDA);
    means.fill_(1.0f);

    // Simulate fill_free_slots_with_data() calling index_put_()
    auto free_indices = Tensor::from_vector({50, 51, 52, 53, 54}, {5}, Device::CUDA).to(DataType::Int64);
    auto new_data = Tensor::full({5, 3}, 99.0f, Device::CUDA);

    means.index_put_(free_indices, new_data);

    // CRITICAL: Capacity must be preserved for append_zeros to work
    ASSERT_EQ(means.capacity(), max_capacity)
        << "REGRESSION: index_put_() destroyed capacity! "
        << "This was the original bug that broke densification.";

    // Now append_zeros should work
    EXPECT_NO_THROW(means.append_zeros(10))
        << "append_zeros() failed after index_put_() - capacity was likely destroyed";

    EXPECT_EQ(means.shape()[0], n_gaussians + 10);
    EXPECT_EQ(means.capacity(), max_capacity);
}
