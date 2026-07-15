/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

/**
 * @brief Test suite to verify Tensor slicing is safe for pre-allocation strategy
 *
 * Tests verify:
 * 1. Slicing doesn't create copies (zero-copy)
 * 2. .ptr() returns correct pointer (base or offset based on implementation)
 * 3. .shape()[0] returns slice size, not full tensor size
 * 4. Modifications through slice affect original tensor
 * 5. Multiple slices share same underlying buffer
 */

class TensorSlicingSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

/**
 * Test 1: Verify slicing is zero-copy (doesn't allocate new memory)
 */
TEST_F(TensorSlicingSafetyTest, SliceIsZeroCopy) {
    const size_t N = 1000;
    Tensor full = Tensor::zeros({N, 3}, Device::CUDA);

    void* full_ptr = full.data_ptr();

    Tensor slice = full.slice(0, 0, 500);

    // Verify slice shares same base pointer
    void* slice_ptr = slice.data_ptr();
    EXPECT_EQ(slice_ptr, full_ptr)
        << "Slice pointer differs from original! Not zero-copy.";
}

/**
 * Test 2: Verify slice.shape()[0] returns slice size, not full size
 */
TEST_F(TensorSlicingSafetyTest, SliceShapeReturnsLogicalSize) {
    const size_t N = 1000;
    const size_t SLICE_SIZE = 500;

    Tensor full = Tensor::zeros({N, 3}, Device::CUDA);
    Tensor slice = full.slice(0, 0, SLICE_SIZE);

    EXPECT_EQ(full.shape()[0], N) << "Full tensor shape changed!";
    EXPECT_EQ(slice.shape()[0], SLICE_SIZE) << "Slice shape incorrect!";
    EXPECT_EQ(slice.shape()[1], 3) << "Slice other dims incorrect!";
    EXPECT_EQ(slice.numel(), SLICE_SIZE * 3) << "Slice numel incorrect!";
}

/**
 * Test 3: Verify .ptr<T>() returns correct pointer
 * CRITICAL: Rasterizer uses .ptr<float>() to get GPU pointers
 * This test documents the actual behavior - whether ptr() returns base or offset pointer
 */
TEST_F(TensorSlicingSafetyTest, SlicePtrBehavior) {
    const size_t N = 1000;
    Tensor full = Tensor::zeros({N, 3}, Device::CUDA);

    float* full_ptr = full.ptr<float>();

    // Create slice starting at different offsets
    Tensor slice1 = full.slice(0, 0, 500);   // [0:500]
    Tensor slice2 = full.slice(0, 100, 600); // [100:600]

    float* slice1_ptr = slice1.ptr<float>();
    float* slice2_ptr = slice2.ptr<float>();

    // Document the actual behavior
    // Option A: ptr() returns base pointer (slice1_ptr == full_ptr)
    // Option B: ptr() returns offset pointer (slice2_ptr == full_ptr + offset)

    // CRITICAL FINDING: Tensor.ptr() returns OFFSET pointer!
    // This means slicing is compatible with pre-allocation strategy
    //
    // Example: If we pre-allocate storage[0:4M] and create slice[0:54k],
    // then slice.ptr() points to storage[0], which is correct for rasterizer.
    //
    // The rasterizer will use slice.shape()[0] for the count and slice.ptr() for data,
    // which gives it exactly the first 54k elements.

    std::cout << "Tensor.ptr() returns OFFSET pointer (slice starts at offset)" << std::endl;
    EXPECT_EQ(slice1_ptr, full_ptr);             // Slice at offset 0 points to base
    EXPECT_EQ(slice2_ptr, full_ptr + (100 * 3)); // Slice at offset 100 points to base+offset
}

/**
 * Test 4: Verify modifications through slice affect original tensor
 */
TEST_F(TensorSlicingSafetyTest, SliceModifiesOriginalTensor) {
    const size_t N = 1000;
    Tensor full = Tensor::zeros({N, 3}, Device::CUDA);

    // Modify through slice
    Tensor slice = full.slice(0, 0, 500);
    slice.fill_(42.0f);

    // Check that original tensor's first 500 rows are modified
    Tensor first_500 = full.slice(0, 0, 500);
    Tensor last_500 = full.slice(0, 500, 1000);

    // Copy first and last values to CPU to check
    float first_val, last_val;
    cudaMemcpy(&first_val, first_500.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_val, last_500.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_FLOAT_EQ(first_val, 42.0f)
        << "Slice modification didn't affect original tensor!";
    EXPECT_FLOAT_EQ(last_val, 0.0f)
        << "Slice modification affected wrong part of tensor!";
}

/**
 * Test 5: Multiple slices can coexist and share buffer
 */
TEST_F(TensorSlicingSafetyTest, MultipleSlicesShareBuffer) {
    const size_t N = 1000;
    Tensor full = Tensor::zeros({N, 3}, Device::CUDA);

    // Create multiple overlapping slices
    Tensor slice1 = full.slice(0, 0, 500);    // [0:500]
    Tensor slice2 = full.slice(0, 250, 750);  // [250:750]
    Tensor slice3 = full.slice(0, 500, 1000); // [500:1000]

    // All should share same underlying storage
    void* base_ptr = full.storage_ptr();
    EXPECT_EQ(slice1.storage_ptr(), base_ptr);
    EXPECT_EQ(slice2.storage_ptr(), base_ptr);
    EXPECT_EQ(slice3.storage_ptr(), base_ptr);

    // Modify through slice1
    slice1.fill_(10.0f);

    // Verify slice2's overlapping region [250:500] is modified
    Tensor overlap = slice2.slice(0, 0, 250); // First 250 of slice2 = [250:500] of full
    float overlap_val;
    cudaMemcpy(&overlap_val, overlap.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_FLOAT_EQ(overlap_val, 10.0f)
        << "Multiple slices don't share buffer!";
}

/**
 * Test 6: Verify slicing with dim=0 works for different tensor shapes
 */
TEST_F(TensorSlicingSafetyTest, SlicingDifferentShapes) {
    // Test various shapes used in Gaussian Splatting
    std::vector<std::vector<size_t>> shapes = {
        {1000, 3},     // means: [N, 3]
        {1000, 1, 3},  // sh0: [N, 1, 3]
        {1000, 15, 3}, // shN: [N, 15, 3]
        {1000, 4},     // rotations: [N, 4]
        {1000, 1},     // opacity: [N, 1]
    };

    for (const auto& shape : shapes) {
        Tensor full = Tensor::zeros(TensorShape(shape), Device::CUDA);
        Tensor slice = full.slice(0, 0, 500);

        // Verify slice has correct shape
        EXPECT_EQ(slice.shape()[0], 500);
        for (size_t i = 1; i < shape.size(); i++) {
            EXPECT_EQ(slice.shape()[i], shape[i]);
        }

        // Verify zero-copy
        EXPECT_EQ(slice.data_ptr(), full.data_ptr());
    }
}

/**
 * Test 7: Verify slice with end > size doesn't crash
 */
TEST_F(TensorSlicingSafetyTest, SliceOutOfBounds) {
    const size_t N = 1000;
    Tensor full = Tensor::zeros({N, 3}, Device::CUDA);

    EXPECT_THROW(full.slice(0, 0, 2000), std::runtime_error);
}

/**
 * Test 10: Integration test - simulate MCMC growth pattern
 */
TEST_F(TensorSlicingSafetyTest, SimulateMCMCGrowth) {
    const size_t MAX_CAP = 4096;
    const size_t START_SIZE = 64;

    // Pre-allocate full capacity
    Tensor storage = Tensor::zeros({MAX_CAP, 3}, Device::CUDA);
    size_t current_size = START_SIZE;

    const auto* storage_ptr = storage.storage_ptr();

    // Simulate MCMC growth (5% per step)
    while (current_size < MAX_CAP) {
        // Get current view
        Tensor current = storage.slice(0, 0, current_size);

        // Verify shape is correct
        EXPECT_EQ(current.shape()[0], current_size);
        EXPECT_EQ(current.storage_ptr(), storage_ptr);

        // Simulate adding new Gaussians (5% growth)
        size_t n_new = std::min(
            std::max<size_t>(1, static_cast<size_t>(current_size * 0.05)),
            MAX_CAP - current_size);
        current_size += n_new;

        // Create new view with grown size
        Tensor grown = storage.slice(0, 0, current_size);
        EXPECT_EQ(grown.shape()[0], current_size);
        EXPECT_EQ(grown.storage_ptr(), storage_ptr);
    }
}
