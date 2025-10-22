/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

class MinimalSortDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        lfs::core::Tensor::manual_seed(42);
    }
};

TEST_F(MinimalSortDebugTest, Step1_BasicSort) {
    std::cout << "\n=== STEP 1: Basic Sort Call ===" << std::endl;

    auto data = lfs::core::Tensor::from_vector({3.0f, 1.0f, 4.0f}, {3}, lfs::core::Device::CUDA);
    std::cout << "Input created" << std::endl;

    try {
        std::cout << "Calling sort(0)..." << std::endl;
        auto result = data.sort(0);
        std::cout << "✓ sort() returned successfully" << std::endl;

        std::cout << "Accessing result.first..." << std::endl;
        auto vals = result.first;
        std::cout << "✓ Got values tensor" << std::endl;
        std::cout << "  Values shape: [" << vals.shape()[0] << "]" << std::endl;
        std::cout << "  Values dtype: " << lfs::core::dtype_name(vals.dtype()) << std::endl;

        std::cout << "Accessing result.second..." << std::endl;
        auto idx = result.second;
        std::cout << "✓ Got indices tensor" << std::endl;
        std::cout << "  Indices shape: [" << idx.shape()[0] << "]" << std::endl;
        std::cout << "  Indices dtype: " << lfs::core::dtype_name(idx.dtype()) << std::endl;

        SUCCEED();
    } catch (const std::exception& e) {
        std::cout << "✗ Exception: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(MinimalSortDebugTest, Step2_ReadSortedValues) {
    std::cout << "\n=== STEP 2: Read Sorted Values ===" << std::endl;

    auto data = lfs::core::Tensor::from_vector({3.0f, 1.0f, 4.0f}, {3}, lfs::core::Device::CUDA);
    auto [vals, idx] = data.sort(0);

    std::cout << "Moving values to CPU..." << std::endl;
    try {
        auto vals_cpu = vals.cpu();
        std::cout << "✓ Moved to CPU" << std::endl;

        std::cout << "Converting to vector..." << std::endl;
        auto vals_vec = vals_cpu.to_vector();
        std::cout << "✓ Converted to vector, size: " << vals_vec.size() << std::endl;

        std::cout << "Values: ";
        for (auto v : vals_vec) {
            std::cout << v << " ";
        }
        std::cout << std::endl;

        EXPECT_EQ(vals_vec.size(), 3);
        EXPECT_FLOAT_EQ(vals_vec[0], 1.0f);
        EXPECT_FLOAT_EQ(vals_vec[1], 3.0f);
        EXPECT_FLOAT_EQ(vals_vec[2], 4.0f);

    } catch (const std::exception& e) {
        std::cout << "✗ Exception: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(MinimalSortDebugTest, Step3_CheckIndicesDtype) {
    std::cout << "\n=== STEP 3: Check Indices Dtype ===" << std::endl;

    auto data = lfs::core::Tensor::from_vector({3.0f, 1.0f, 4.0f}, {3}, lfs::core::Device::CUDA);
    auto [vals, idx] = data.sort(0);

    std::cout << "Indices dtype: " << lfs::core::dtype_name(idx.dtype()) << std::endl;

    if (idx.dtype() == lfs::core::DataType::Int64) {
        std::cout << "✓ Indices are Int64" << std::endl;
    } else if (idx.dtype() == lfs::core::DataType::Int32) {
        std::cout << "✓ Indices are Int32" << std::endl;
    } else {
        std::cout << "✗ Unexpected dtype!" << std::endl;
        FAIL();
    }

    std::cout << "Device: " << lfs::core::device_name(idx.device()) << std::endl;
    std::cout << "Numel: " << idx.numel() << std::endl;
    std::cout << "Is valid: " << idx.is_valid() << std::endl;

    SUCCEED();
}

TEST_F(MinimalSortDebugTest, Step4_MoveIndicesToCPU) {
    std::cout << "\n=== STEP 4: Move Indices to CPU ===" << std::endl;

    auto data = lfs::core::Tensor::from_vector({3.0f, 1.0f, 4.0f}, {3}, lfs::core::Device::CUDA);
    auto [vals, idx] = data.sort(0);

    std::cout << "Before cpu(): dtype=" << lfs::core::dtype_name(idx.dtype())
              << " device=" << lfs::core::device_name(idx.device()) << std::endl;

    try {
        std::cout << "Calling idx.cpu()..." << std::endl;
        auto idx_cpu = idx.cpu();
        std::cout << "✓ Moved to CPU" << std::endl;

        std::cout << "After cpu(): dtype=" << lfs::core::dtype_name(idx_cpu.dtype())
                  << " device=" << lfs::core::device_name(idx_cpu.device()) << std::endl;

        EXPECT_EQ(idx_cpu.device(), lfs::core::Device::CPU);

    } catch (const std::exception& e) {
        std::cout << "✗ Exception during cpu(): " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(MinimalSortDebugTest, Step5_ReadIndicesAsInt64) {
    std::cout << "\n=== STEP 5: Read Indices as Int64 ===" << std::endl;

    auto data = lfs::core::Tensor::from_vector({3.0f, 1.0f, 4.0f}, {3}, lfs::core::Device::CUDA);
    auto [vals, idx] = data.sort(0);

    auto idx_cpu = idx.cpu();

    std::cout << "Checking if to_vector_int64() exists..." << std::endl;

    try {
        std::cout << "Calling to_vector_int64()..." << std::endl;
        auto idx_vec = idx_cpu.to_vector_int64();
        std::cout << "✓ Got int64 vector, size: " << idx_vec.size() << std::endl;

        std::cout << "Indices: ";
        for (auto v : idx_vec) {
            std::cout << v << " ";
        }
        std::cout << std::endl;

        EXPECT_EQ(idx_vec.size(), 3);
        // Indices should be [1, 0, 2] for sorting [3, 1, 4]

    } catch (const std::exception& e) {
        std::cout << "✗ to_vector_int64() failed: " << e.what() << std::endl;
        std::cout << "Trying to_vector_int() instead..." << std::endl;

        try {
            auto idx_vec = idx_cpu.to_vector_int();
            std::cout << "✓ Got int32 vector, size: " << idx_vec.size() << std::endl;

            std::cout << "Indices: ";
            for (auto v : idx_vec) {
                std::cout << v << " ";
            }
            std::cout << std::endl;

        } catch (const std::exception& e2) {
            std::cout << "✗ to_vector_int() also failed: " << e2.what() << std::endl;
            FAIL();
        }
    }
}

TEST_F(MinimalSortDebugTest, Step6_DirectPointerAccess) {
    std::cout << "\n=== STEP 6: Direct Pointer Access ===" << std::endl;

    auto data = lfs::core::Tensor::from_vector({3.0f, 1.0f, 4.0f}, {3}, lfs::core::Device::CUDA);
    auto [vals, idx] = data.sort(0);

    auto idx_cpu = idx.cpu();

    std::cout << "Dtype: " << lfs::core::dtype_name(idx_cpu.dtype()) << std::endl;

    if (idx_cpu.dtype() == lfs::core::DataType::Int64) {
        std::cout << "Attempting ptr<int64_t>()..." << std::endl;
        try {
            auto* ptr = idx_cpu.ptr<int64_t>();
            if (ptr == nullptr) {
                std::cout << "✗ ptr is nullptr!" << std::endl;
                FAIL();
            }
            std::cout << "✓ Got pointer" << std::endl;

            std::cout << "Reading values: ";
            for (size_t i = 0; i < 3; ++i) {
                std::cout << ptr[i] << " ";
            }
            std::cout << std::endl;

        } catch (const std::exception& e) {
            std::cout << "✗ Exception: " << e.what() << std::endl;
            FAIL();
        }
    } else if (idx_cpu.dtype() == lfs::core::DataType::Int32) {
        std::cout << "Attempting ptr<int>()..." << std::endl;
        try {
            auto* ptr = idx_cpu.ptr<int>();
            if (ptr == nullptr) {
                std::cout << "✗ ptr is nullptr!" << std::endl;
                FAIL();
            }
            std::cout << "✓ Got pointer" << std::endl;

            std::cout << "Reading values: ";
            for (size_t i = 0; i < 3; ++i) {
                std::cout << ptr[i] << " ";
            }
            std::cout << std::endl;

        } catch (const std::exception& e) {
            std::cout << "✗ Exception: " << e.what() << std::endl;
            FAIL();
        }
    }
}

TEST_F(MinimalSortDebugTest, Step7_CheckIndicesValidity) {
    std::cout << "\n=== STEP 7: Validate Indices Content ===" << std::endl;

    // Use a simple known case
    auto data = lfs::core::Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f}, {4}, lfs::core::Device::CUDA);
    std::cout << "Input: [5.0, 2.0, 8.0, 1.0]" << std::endl;
    std::cout << "Expected sorted: [1.0, 2.0, 5.0, 8.0]" << std::endl;
    std::cout << "Expected indices: [3, 1, 0, 2]" << std::endl;

    auto [vals, idx] = data.sort(0);

    // Check values
    auto vals_cpu = vals.cpu().to_vector();
    std::cout << "\nActual sorted values: ";
    for (auto v : vals_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    // Check indices
    auto idx_cpu = idx.cpu();

    std::cout << "\nAttempting to read indices..." << std::endl;
    std::cout << "Dtype: " << lfs::core::dtype_name(idx_cpu.dtype()) << std::endl;
    std::cout << "Device: " << lfs::core::device_name(idx_cpu.device()) << std::endl;
    std::cout << "Valid: " << idx_cpu.is_valid() << std::endl;
    std::cout << "Numel: " << idx_cpu.numel() << std::endl;
    std::cout << "Raw ptr: " << idx_cpu.raw_ptr() << std::endl;

    if (idx_cpu.raw_ptr() == nullptr) {
        std::cout << "✗ ERROR: Raw pointer is null!" << std::endl;
        FAIL();
    }

    try {
        if (idx_cpu.dtype() == lfs::core::DataType::Int64) {
            auto idx_vec = idx_cpu.to_vector_int64();
            std::cout << "Actual indices (int64): ";
            for (auto v : idx_vec)
                std::cout << v << " ";
            std::cout << std::endl;
        } else if (idx_cpu.dtype() == lfs::core::DataType::Int32) {
            auto idx_vec = idx_cpu.to_vector_int();
            std::cout << "Actual indices (int32): ";
            for (auto v : idx_vec)
                std::cout << v << " ";
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Exception reading indices: " << e.what() << std::endl;
        FAIL();
    }
}
