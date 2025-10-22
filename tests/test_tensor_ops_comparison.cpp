/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "core_new/tensor.hpp"

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-4f;

    // Helper to convert torch::Tensor to lfs::core::Tensor
    lfs::core::Tensor torch_to_tensor(const torch::Tensor& torch_tensor) {
        auto cpu_tensor = torch_tensor.cpu().contiguous();
        std::vector<size_t> shape;
        for (int i = 0; i < torch_tensor.dim(); ++i) {
            shape.push_back(torch_tensor.size(i));
        }

        if (torch_tensor.scalar_type() == torch::kFloat32) {
            std::vector<float> data(cpu_tensor.data_ptr<float>(),
                                    cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
            return lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
        } else if (torch_tensor.scalar_type() == torch::kInt32) {
            std::vector<int> data(cpu_tensor.data_ptr<int>(),
                                  cpu_tensor.data_ptr<int>() + cpu_tensor.numel());
            return lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
        } else if (torch_tensor.scalar_type() == torch::kBool) {
            std::vector<bool> data;
            auto bool_ptr = cpu_tensor.data_ptr<bool>();
            for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
                data.push_back(bool_ptr[i]);
            }
            return lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
        }

        return lfs::core::Tensor();
    }

    // Helper to convert lfs::core::Tensor to torch::Tensor
    torch::Tensor tensor_to_torch(const lfs::core::Tensor& gs_tensor) {
        auto cpu_tensor = gs_tensor.cpu();
        std::vector<int64_t> shape;
        for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
            shape.push_back(cpu_tensor.shape()[i]);
        }

        if (gs_tensor.dtype() == lfs::core::DataType::Float32) {
            auto data = cpu_tensor.to_vector();
            auto torch_tensor = torch::from_blob(data.data(), shape, torch::kFloat32).clone();
            return torch_tensor.cuda();
        } else if (gs_tensor.dtype() == lfs::core::DataType::Int32) {
            auto data = cpu_tensor.to_vector_int();
            auto torch_tensor = torch::from_blob(data.data(), shape, torch::kInt32).clone();
            return torch_tensor.cuda();
        } else if (gs_tensor.dtype() == lfs::core::DataType::Bool) {
            auto data = cpu_tensor.to_vector_bool();
            std::vector<uint8_t> uint8_data(data.begin(), data.end());
            auto torch_tensor = torch::from_blob(uint8_data.data(), shape, torch::kUInt8).clone().to(torch::kBool);
            return torch_tensor.cuda();
        }

        return torch::Tensor();
    }

    // Helper to compare tensors
    bool tensors_close(const lfs::core::Tensor& a, const torch::Tensor& b, float tol = FLOAT_TOLERANCE) {
        if (a.numel() != b.numel()) {
            std::cout << "Size mismatch: " << a.numel() << " vs " << b.numel() << std::endl;
            return false;
        }

        auto a_cpu = a.cpu().to_vector();
        auto b_cpu = b.cpu();
        auto b_ptr = b_cpu.data_ptr<float>();

        for (size_t i = 0; i < a.numel(); ++i) {
            float diff = std::abs(a_cpu[i] - b_ptr[i]);
            if (diff > tol) {
                std::cout << "Mismatch at index " << i << ": " << a_cpu[i] << " vs " << b_ptr[i]
                          << " (diff: " << diff << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    // Helper for bool tensor comparison
    bool bool_tensors_equal(const lfs::core::Tensor& a, const torch::Tensor& b) {
        if (a.numel() != b.numel()) {
            std::cout << "Size mismatch: " << a.numel() << " vs " << b.numel() << std::endl;
            return false;
        }

        auto a_cpu = a.cpu().to_vector_bool();
        auto b_cpu = b.cpu();
        auto b_ptr = b_cpu.data_ptr<bool>();

        for (size_t i = 0; i < a.numel(); ++i) {
            if (a_cpu[i] != b_ptr[i]) {
                std::cout << "Mismatch at index " << i << ": " << a_cpu[i] << " vs " << b_ptr[i] << std::endl;
                return false;
            }
        }
        return true;
    }

} // anonymous namespace

class TensorBugHuntingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        lfs::core::Tensor::manual_seed(42);
    }
};

// ============================================================================
// CRITICAL BUG TESTS - CPU ↔ CUDA Transfer
// ============================================================================

TEST_F(TensorBugHuntingTest, CriticalCPUToGPUTransfer) {
    // The critical test mentioned in the requirements
    auto t = lfs::core::Tensor::ones({2, 3}, lfs::core::Device::CPU);
    t.ptr<float>()[0] = 99.0f;

    auto gpu = t.cuda();
    auto back = gpu.cpu();

    EXPECT_FLOAT_EQ(back.ptr<float>()[0], 99.0f) << "CPU→GPU→CPU transfer corrupted data!";

    // Also test via operator[]
    EXPECT_FLOAT_EQ(back[0][0], 99.0f) << "Operator[] access shows corrupted data!";
}

TEST_F(TensorBugHuntingTest, CPUToGPUTransferMultipleValues) {
    auto t = lfs::core::Tensor::zeros({5, 4}, lfs::core::Device::CPU);
    float* data = t.ptr<float>();

    // Set unique values
    for (int i = 0; i < 20; ++i) {
        data[i] = static_cast<float>(i * 10);
    }

    auto gpu = t.cuda();
    auto back = gpu.cpu();
    float* back_data = back.ptr<float>();

    // Verify all values
    for (int i = 0; i < 20; ++i) {
        EXPECT_FLOAT_EQ(back_data[i], static_cast<float>(i * 10))
            << "Value corrupted at index " << i;
    }
}

TEST_F(TensorBugHuntingTest, GPUToCPUTransferPreservesData) {
    // Create directly on GPU, transfer to CPU
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto gs_cpu = gs_data.cpu();
    auto torch_cpu = torch_data.cpu();

    EXPECT_TRUE(tensors_close(gs_cpu, torch_cpu));
}

TEST_F(TensorBugHuntingTest, RepeatedCPUGPUTransfers) {
    auto t = lfs::core::Tensor::ones({3, 3}, lfs::core::Device::CPU);
    t.ptr<float>()[4] = 123.456f; // Center element

    // Multiple round trips
    for (int i = 0; i < 5; ++i) {
        auto gpu = t.cuda();
        t = gpu.cpu();

        EXPECT_FLOAT_EQ(t.ptr<float>()[4], 123.456f)
            << "Data corrupted on round trip " << i;
    }
}

// ============================================================================
// CLONE TESTS - Deep Copy Verification
// ============================================================================

TEST_F(TensorBugHuntingTest, CloneIsDeepCopy) {
    auto original = lfs::core::Tensor::ones({5, 5}, lfs::core::Device::CUDA);
    auto cloned = original.clone();

    // Modify original
    auto orig_cpu = original.cpu();
    orig_cpu.ptr<float>()[0] = 999.0f;
    original = orig_cpu.cuda();

    // Cloned should be unchanged
    auto clone_cpu = cloned.cpu();
    EXPECT_FLOAT_EQ(clone_cpu.ptr<float>()[0], 1.0f)
        << "Clone shares memory with original!";
}

TEST_F(TensorBugHuntingTest, CloneCPUTensor) {
    auto original = lfs::core::Tensor::ones({5, 5}, lfs::core::Device::CPU);
    original.ptr<float>()[0] = 42.0f;

    auto cloned = original.clone();

    // Modify original
    original.ptr<float>()[0] = 999.0f;

    // Cloned should be unchanged
    EXPECT_FLOAT_EQ(cloned.ptr<float>()[0], 42.0f)
        << "CPU clone shares memory with original!";
}

TEST_F(TensorBugHuntingTest, ClonePreservesAllValues) {
    auto torch_data = torch::randn({10, 10}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto gs_cloned = gs_data.clone();

    EXPECT_TRUE(tensors_close(gs_cloned, torch_data));
}

// ============================================================================
// TENSORROWPROXY ASSIGNMENT TESTS - Multi-dimensional Tensor Assignment
// ============================================================================

TEST_F(TensorBugHuntingTest, TensorRowProxyAssignment2D) {
    auto torch_dest = torch::zeros({5, 3}, torch::kCUDA);
    auto torch_src = torch::ones({3}, torch::kCUDA) * 42.0f;

    auto gs_dest = lfs::core::Tensor::zeros({5, 3}, lfs::core::Device::CUDA);
    auto gs_src = lfs::core::Tensor::ones({3}, lfs::core::Device::CUDA).mul(42.0f);

    // Assign to row 2
    torch_dest[2] = torch_src;
    gs_dest[2] = gs_src;

    EXPECT_TRUE(tensors_close(gs_dest, torch_dest))
        << "TensorRowProxy assignment failed for 2D tensor!";
}

TEST_F(TensorBugHuntingTest, TensorRowProxyAssignmentFromAnotherRowDebug) {
    std::cout << "\n=== DEBUGGING row-to-row assignment ===" << std::endl;

    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // Print ALL data before operation
    std::cout << "\n--- BEFORE operation ---" << std::endl;
    for (int row = 0; row < 5; ++row) {
        std::vector<float> torch_row(3);
        std::vector<float> gs_row(3);

        for (int col = 0; col < 3; ++col) {
            torch_row[col] = torch_data[row][col].item<float>();
        }

        cudaMemcpy(gs_row.data(), gs_data.ptr<float>() + row * 3,
                   3 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Row " << row << ":" << std::endl;
        std::cout << "  PyTorch: [" << torch_row[0] << ", " << torch_row[1] << ", " << torch_row[2] << "]" << std::endl;
        std::cout << "  GS:      [" << gs_row[0] << ", " << gs_row[1] << ", " << gs_row[2] << "]" << std::endl;
    }

    // Perform operation
    std::cout << "\n--- Extracting row 0 ---" << std::endl;
    auto torch_row0 = torch_data[0].clone();
    lfs::core::Tensor gs_row0_tensor = gs_data[0]; // TensorRowProxy → Tensor conversion

    std::vector<float> torch_row0_check(3);
    for (int i = 0; i < 3; ++i) {
        torch_row0_check[i] = torch_row0[i].item<float>();
    }

    // Convert to CPU and get values
    auto gs_row0_cpu = gs_row0_tensor.cpu();
    auto gs_row0_check = gs_row0_cpu.to_vector();

    std::cout << "PyTorch row 0 copy: [" << torch_row0_check[0] << ", " << torch_row0_check[1] << ", " << torch_row0_check[2] << "]" << std::endl;
    std::cout << "GS row 0 copy:      [" << gs_row0_check[0] << ", " << gs_row0_check[1] << ", " << gs_row0_check[2] << "]" << std::endl;

    std::cout << "\n--- Assigning to row 4 ---" << std::endl;
    torch_data[4] = torch_row0;
    gs_data[4] = gs_row0_tensor;

    // Print ALL data after operation
    std::cout << "\n--- AFTER operation ---" << std::endl;
    for (int row = 0; row < 5; ++row) {
        std::vector<float> torch_row(3);
        std::vector<float> gs_row(3);

        for (int col = 0; col < 3; ++col) {
            torch_row[col] = torch_data[row][col].item<float>();
        }

        cudaMemcpy(gs_row.data(), gs_data.ptr<float>() + row * 3,
                   3 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Row " << row << ":" << std::endl;
        std::cout << "  PyTorch: [" << torch_row[0] << ", " << torch_row[1] << ", " << torch_row[2] << "]" << std::endl;
        std::cout << "  GS:      [" << gs_row[0] << ", " << gs_row[1] << ", " << gs_row[2] << "]" << std::endl;

        if (row == 4) {
            // Compare row 4 with original row 0
            bool torch_match = true, gs_match = true;
            for (int i = 0; i < 3; ++i) {
                if (std::abs(torch_row[i] - torch_row0_check[i]) > 0.0001f)
                    torch_match = false;
                if (std::abs(gs_row[i] - gs_row0_check[i]) > 0.0001f)
                    gs_match = false;
            }
            std::cout << "  PyTorch row 4 matches row 0? " << (torch_match ? "YES" : "NO") << std::endl;
            std::cout << "  GS row 4 matches row 0? " << (gs_match ? "YES" : "NO") << std::endl;
        }
    }
}

TEST_F(TensorBugHuntingTest, DiagnosticTensorsCloseFunction) {
    std::cout << "\n=== Testing tensors_close() function ===" << std::endl;

    // Create identical tensors
    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // They should be close (identical)
    bool close_before = tensors_close(gs_data, torch_data);
    std::cout << "Initial tensors are close? " << (close_before ? "YES" : "NO") << std::endl;

    if (!close_before) {
        std::cout << "ERROR: Initial conversion doesn't preserve values!" << std::endl;

        // Print differences
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 3; ++col) {
                float torch_val = torch_data[row][col].item<float>();

                std::vector<float> gs_val(1);
                cudaMemcpy(gs_val.data(), gs_data.ptr<float>() + row * 3 + col,
                           sizeof(float), cudaMemcpyDeviceToHost);

                float diff = std::abs(torch_val - gs_val[0]);
                if (diff > 1e-5) {
                    std::cout << "  Mismatch at [" << row << "," << col << "]: "
                              << "PyTorch=" << torch_val << ", GS=" << gs_val[0]
                              << ", diff=" << diff << std::endl;
                }
            }
        }
    }

    EXPECT_TRUE(close_before) << "Initial tensors should be identical!";
}

TEST_F(TensorBugHuntingTest, TensorRowProxyAssignmentFromAnotherRowVerbose) {
    std::cout << "\n=== Original test with verbose output ===" << std::endl;

    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    std::cout << "Initial tensors_close: " << (tensors_close(gs_data, torch_data) ? "PASS" : "FAIL") << std::endl;

    // Copy row 0 to row 4
    auto torch_row0 = torch_data[0].clone();
    auto gs_row0_tensor = gs_data[0];

    torch_data[4] = torch_row0;
    gs_data[4] = gs_row0_tensor;

    std::cout << "After assignment tensors_close: " << (tensors_close(gs_data, torch_data) ? "PASS" : "FAIL") << std::endl;

    // Manual comparison
    std::cout << "\nManual element-by-element comparison:" << std::endl;
    bool all_match = true;
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            float torch_val = torch_data[row][col].item<float>();

            std::vector<float> gs_val(1);
            cudaMemcpy(gs_val.data(), gs_data.ptr<float>() + row * 3 + col,
                       sizeof(float), cudaMemcpyDeviceToHost);

            float diff = std::abs(torch_val - gs_val[0]);
            if (diff > 1e-3) {
                std::cout << "  MISMATCH at [" << row << "," << col << "]: "
                          << "PyTorch=" << torch_val << ", GS=" << gs_val[0]
                          << ", diff=" << diff << std::endl;
                all_match = false;
            }
        }
    }

    if (all_match) {
        std::cout << "✓ All elements match!" << std::endl;
    }

    EXPECT_TRUE(all_match);
}

TEST_F(TensorBugHuntingTest, DiagnosticAutoVsExplicitType) {
    std::cout << "\n=== Testing auto vs explicit type ===" << std::endl;

    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    std::cout << "Test 1: Using auto (stays as proxy?)" << std::endl;
    {
        auto torch_row0 = torch_data[0].clone();
        auto gs_row0 = gs_data[0]; // This might be TensorRowProxy!

        std::cout << "  Type of gs_row0: " << typeid(gs_row0).name() << std::endl;

        torch_data[4] = torch_row0;
        gs_data[4] = gs_row0;

        std::vector<float> gs_row4(3);
        cudaMemcpy(gs_row4.data(), gs_data.ptr<float>() + 12,
                   3 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "  GS row 4 after auto: [" << gs_row4[0] << ", " << gs_row4[1] << ", " << gs_row4[2] << "]" << std::endl;
    }

    // Reset
    gs_data = torch_to_tensor(torch_data);

    std::cout << "\nTest 2: Using explicit Tensor type (forces conversion)" << std::endl;
    {
        auto torch_row0 = torch_data[0].clone();
        lfs::core::Tensor gs_row0 = gs_data[0]; // Explicit type forces conversion!

        std::cout << "  Type of gs_row0: " << typeid(gs_row0).name() << std::endl;

        torch_data[4] = torch_row0;
        gs_data[4] = gs_row0;

        std::vector<float> gs_row4(3);
        cudaMemcpy(gs_row4.data(), gs_data.ptr<float>() + 12,
                   3 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "  GS row 4 after explicit: [" << gs_row4[0] << ", " << gs_row4[1] << ", " << gs_row4[2] << "]" << std::endl;
    }
}

TEST_F(TensorBugHuntingTest, TensorRowProxyScalarAssignment) {
    auto torch_data = torch::zeros({5, 3}, torch::kCUDA);
    auto gs_data = lfs::core::Tensor::zeros({5, 3}, lfs::core::Device::CUDA);

    // Assign scalar to a row (should broadcast)
    auto torch_scalar = torch::full({3}, 99.0f, torch::kCUDA);
    auto gs_scalar = lfs::core::Tensor::full({3}, 99.0f, lfs::core::Device::CUDA);

    torch_data[2] = torch_scalar;
    gs_data[2] = gs_scalar;

    EXPECT_TRUE(tensors_close(gs_data, torch_data));
}

TEST_F(TensorBugHuntingTest, TensorRowProxy1DAccess) {
    auto torch_1d = torch::randn({10}, torch::kCUDA);
    auto gs_1d = torch_to_tensor(torch_1d);

    // Access single element
    float torch_val = torch_1d[5].item<float>();
    float gs_val = gs_1d[5]; // Should convert proxy to float

    EXPECT_FLOAT_EQ(gs_val, torch_val)
        << "1D TensorRowProxy conversion to float failed!";
}

// ============================================================================
// EXPAND/BROADCAST TESTS - Memory Layout Issues
// ============================================================================

TEST_F(TensorBugHuntingTest, ExpandNarrowingConversion) {
    auto torch_data = torch::randn({1, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // Expand from {1, 5} to {10, 5}
    auto torch_expanded = torch_data.expand({10, 5});
    auto gs_expanded = gs_data.expand({10, 5});

    EXPECT_TRUE(tensors_close(gs_expanded, torch_expanded))
        << "Expand with narrowing conversion failed!";
}

TEST_F(TensorBugHuntingTest, ExpandWithSingletonDimensions) {
    auto torch_data = torch::randn({1, 1, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_expanded = torch_data.expand({10, 8, 5});
    auto gs_expanded = gs_data.expand({10, 8, 5});

    EXPECT_TRUE(tensors_close(gs_expanded, torch_expanded));
}

TEST_F(TensorBugHuntingTest, ExpandThenOperation) {
    auto torch_a = torch::ones({1, 5}, torch::kCUDA);
    auto torch_b = torch::randn({10, 5}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    auto torch_expanded = torch_a.expand({10, 5});
    auto gs_expanded = gs_a.expand({10, 5});

    auto torch_result = torch_expanded + torch_b;
    auto gs_result = gs_expanded + gs_b;

    EXPECT_TRUE(tensors_close(gs_result, torch_result))
        << "Operation on expanded tensor failed!";
}

TEST_F(TensorBugHuntingTest, BroadcastAdd) {
    auto torch_a = torch::randn({10, 1}, torch::kCUDA);
    auto torch_b = torch::randn({1, 5}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    auto torch_result = torch_a + torch_b; // Should broadcast to {10, 5}
    auto gs_result = gs_a + gs_b;

    EXPECT_TRUE(tensors_close(gs_result, torch_result))
        << "Broadcasting addition failed!";
}

// ============================================================================
// MASKED SELECT TESTS - Expanded Masks
// ============================================================================

TEST_F(TensorBugHuntingTest, MaskedSelectSimple) {
    auto torch_data = torch::tensor({1.0f, 0.0f, 2.0f, 0.0f, 3.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_mask = torch_data > 0.0f;
    auto gs_mask = gs_data > 0.0f;

    auto torch_selected = torch_data.masked_select(torch_mask);
    auto gs_selected = gs_data.masked_select(gs_mask);

    EXPECT_EQ(gs_selected.numel(), torch_selected.numel());
    EXPECT_TRUE(tensors_close(gs_selected, torch_selected));
}

TEST_F(TensorBugHuntingTest, MaskedSelectExpandedMask) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // Create mask from column 0, then expand
    auto torch_col0 = torch_data.select(1, 0);
    auto torch_mask_1d = torch_col0 > 0.0f;
    auto torch_mask = torch_mask_1d.unsqueeze(1).expand({10, 5});

    auto gs_col0 = gs_data.slice(1, 0, 1).squeeze(1);
    auto gs_mask_1d = gs_col0 > 0.0f;
    auto gs_mask = gs_mask_1d.unsqueeze(1).expand({10, 5});

    auto torch_selected = torch_data.masked_select(torch_mask);
    auto gs_selected = gs_data.masked_select(gs_mask);

    EXPECT_EQ(gs_selected.numel(), torch_selected.numel())
        << "Masked select with expanded mask failed!";
}

TEST_F(TensorBugHuntingTest, MaskedFill) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_mask = torch_data > 0.0f;
    auto gs_mask = gs_data > 0.0f;

    auto torch_result = torch_data.clone();
    torch_result.masked_fill_(torch_mask, -999.0f);

    auto gs_result = gs_data.clone();
    gs_result.masked_fill_(gs_mask, -999.0f);

    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

// ============================================================================
// POINTER ACCESS TESTS - Row-Major Layout
// ============================================================================

TEST_F(TensorBugHuntingTest, PointerAccessRowMajor) {
    auto torch_data = torch::randn({5, 4}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // Access element at [2, 3] via pointer arithmetic
    size_t idx = 2 * 4 + 3; // Row-major: row * cols + col

    std::vector<float> torch_val(1), gs_val(1);
    cudaMemcpy(torch_val.data(), torch_data.data_ptr<float>() + idx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_val.data(), gs_data.ptr<float>() + idx, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(gs_val[0], torch_val[0])
        << "Pointer access doesn't match row-major layout!";
}

TEST_F(TensorBugHuntingTest, PointerAccessAllElements) {
    auto torch_data = torch::randn({3, 4, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    size_t total = 3 * 4 * 5;
    std::vector<float> torch_vec(total), gs_vec(total);

    cudaMemcpy(torch_vec.data(), torch_data.data_ptr<float>(), total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_vec.data(), gs_data.ptr<float>(), total * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < total; ++i) {
        EXPECT_FLOAT_EQ(gs_vec[i], torch_vec[i])
            << "Mismatch at linear index " << i;
    }
}

TEST_F(TensorBugHuntingTest, PointerAccessAfterReshape) {
    auto torch_data = torch::randn({20}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_2d = torch_data.reshape({4, 5});
    auto gs_2d = gs_data.reshape({4, 5});

    // Data should be contiguous and identical
    std::vector<float> torch_vec(20), gs_vec(20);
    cudaMemcpy(torch_vec.data(), torch_2d.data_ptr<float>(), 20 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_vec.data(), gs_2d.ptr<float>(), 20 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 20; ++i) {
        EXPECT_FLOAT_EQ(gs_vec[i], torch_vec[i]);
    }
}

// ============================================================================
// EDGE CASE TESTS - Empty, Zero-Size, Single Element
// ============================================================================

TEST_F(TensorBugHuntingTest, EmptyTensorOperations) {
    auto torch_empty = torch::empty({0}, torch::kCUDA);
    auto gs_empty = lfs::core::Tensor::empty({0}, lfs::core::Device::CUDA);

    EXPECT_EQ(gs_empty.numel(), 0);
    EXPECT_EQ(gs_empty.is_empty(), true);

    // Operations on empty tensors should not crash
    auto gs_sum = gs_empty.sum();
    EXPECT_FLOAT_EQ(gs_sum.item(), 0.0f);
}

TEST_F(TensorBugHuntingTest, SingleElementTensor) {
    auto torch_single = torch::tensor({42.0f}, torch::kCUDA);
    auto gs_single = lfs::core::Tensor::from_vector({42.0f}, {1}, lfs::core::Device::CUDA);

    EXPECT_FLOAT_EQ(gs_single.item(), 42.0f);

    // Test CPU/GPU transfer
    auto cpu = gs_single.cpu();
    EXPECT_FLOAT_EQ(cpu.item(), 42.0f);
}

TEST_F(TensorBugHuntingTest, ZeroInOneDimension) {
    auto torch_zero_dim = torch::empty({0, 5}, torch::kCUDA);
    auto gs_zero_dim = lfs::core::Tensor::empty({0, 5}, lfs::core::Device::CUDA);

    EXPECT_EQ(gs_zero_dim.numel(), 0);

    // Sum should work
    auto gs_sum = gs_zero_dim.sum();
    EXPECT_FLOAT_EQ(gs_sum.item(), 0.0f);
}

// ============================================================================
// DTYPE CONVERSION TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, FloatToIntConversion) {
    auto torch_float = torch::tensor({1.9f, 2.1f, -1.5f}, torch::kCUDA);
    auto gs_float = torch_to_tensor(torch_float);

    auto torch_int = torch_float.to(torch::kInt32);
    auto gs_int = gs_float.to(lfs::core::DataType::Int32);

    auto torch_cpu = torch_int.cpu();
    auto gs_cpu = gs_int.cpu();

    auto torch_ptr = torch_cpu.data_ptr<int>();
    auto gs_vec = gs_cpu.to_vector_int();

    for (size_t i = 0; i < gs_vec.size(); ++i) {
        EXPECT_EQ(gs_vec[i], torch_ptr[i])
            << "Float to Int conversion mismatch at " << i;
    }
}

TEST_F(TensorBugHuntingTest, BoolToFloatConversion) {
    auto torch_bool = torch::tensor({true, false, true}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto gs_bool = torch_to_tensor(torch_bool);

    auto torch_float = torch_bool.to(torch::kFloat32);
    auto gs_float = gs_bool.to(lfs::core::DataType::Float32);

    EXPECT_TRUE(tensors_close(gs_float, torch_float));
}

TEST_F(TensorBugHuntingTest, FloatToBoolConversion) {
    auto torch_float = torch::tensor({0.0f, 1.0f, -1.0f, 0.5f}, torch::kCUDA);
    auto gs_float = torch_to_tensor(torch_float);

    auto torch_bool = torch_float.to(torch::kBool);
    auto gs_bool = gs_float.to(lfs::core::DataType::Bool);

    EXPECT_TRUE(bool_tensors_equal(gs_bool, torch_bool));
}

// ============================================================================
// COMPARISON OPERATION TESTS - Int32 Support
// ============================================================================

TEST_F(TensorBugHuntingTest, Int32Comparison) {
    auto torch_a = torch::randint(0, 100, {10}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_a = torch_to_tensor(torch_a);

    auto torch_result = torch_a > 50;
    auto gs_result = gs_a > 50;

    EXPECT_TRUE(bool_tensors_equal(gs_result, torch_result))
        << "Int32 comparison failed!";
}

TEST_F(TensorBugHuntingTest, Int32TensorComparison) {
    auto torch_a = torch::randint(0, 100, {10}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto torch_b = torch::randint(0, 100, {10}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    auto torch_result = torch_a < torch_b;
    auto gs_result = gs_a < gs_b;

    EXPECT_TRUE(bool_tensors_equal(gs_result, torch_result))
        << "Int32 tensor-to-tensor comparison failed!";
}

TEST_F(TensorBugHuntingTest, Int32Equality) {
    auto torch_a = torch::randint(0, 5, {20}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_a = torch_to_tensor(torch_a);

    auto torch_result = torch_a == 2;
    auto gs_result = gs_a == 2;

    EXPECT_TRUE(bool_tensors_equal(gs_result, torch_result))
        << "Int32 equality comparison failed!";
}

// ============================================================================
// REDUCTION WITH KEEPDIM TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, SumWithKeepdim) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_sum = torch_data.sum(1, true);
    auto gs_sum = gs_data.sum(1, true);

    EXPECT_EQ(torch_sum.size(0), gs_sum.shape()[0]);
    EXPECT_EQ(torch_sum.size(1), gs_sum.shape()[1]);
    EXPECT_TRUE(tensors_close(gs_sum, torch_sum, 1e-3f));
}

TEST_F(TensorBugHuntingTest, MeanMultipleDims) {
    auto torch_data = torch::randn({5, 4, 3}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_mean = torch_data.mean(std::vector<int64_t>{0, 2}, true);
    auto gs_mean = gs_data.mean(std::vector<int>{0, 2}, true);

    EXPECT_TRUE(tensors_close(gs_mean, torch_mean));
}

// ============================================================================
// VARIANCE AND STD TESTS - Biased vs Unbiased
// ============================================================================

TEST_F(TensorBugHuntingTest, VarianceScalar) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // PyTorch: var(unbiased=True) is default
    // Full signature: torch.var(input, dim=None, unbiased=True, keepdim=False)
    float torch_var = torch_data.var(/*unbiased=*/true).item<float>();
    float gs_var = gs_data.var_scalar(/*unbiased=*/true);

    EXPECT_NEAR(gs_var, torch_var, 1e-3f)
        << "Scalar variance calculation mismatch!";
}

TEST_F(TensorBugHuntingTest, StdDimensional) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // PyTorch: std(dim, unbiased=True, keepdim=False)
    auto torch_std = torch_data.std(/*dim=*/1, /*unbiased=*/true, /*keepdim=*/true);
    auto gs_std = gs_data.std(/*dim=*/1, /*keepdim=*/true, /*unbiased=*/true);

    EXPECT_TRUE(tensors_close(gs_std, torch_std, 1e-3f))
        << "Dimensional std calculation mismatch!";
}

// ============================================================================
// MIN/MAX WITH INDICES TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, MinWithIndices1D) {
    auto torch_data = torch::randn({10}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto [torch_val, torch_idx] = torch_data.min(0);
    auto [gs_val, gs_idx] = gs_data.min_with_indices(0);

    EXPECT_FLOAT_EQ(gs_val.item(), torch_val.item<float>());
    EXPECT_EQ(gs_idx.cpu().item<int64_t>(), torch_idx.cpu().item<int64_t>());
}

TEST_F(TensorBugHuntingTest, MaxWithIndices2D) {
    auto torch_data = torch::randn({5, 8}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto [torch_val, torch_idx] = torch_data.max(1);
    auto [gs_val, gs_idx] = gs_data.max_with_indices(1);

    EXPECT_TRUE(tensors_close(gs_val, torch_val));

    // Check indices
    auto torch_idx_cpu = torch_idx.cpu();
    auto gs_idx_cpu = gs_idx.cpu();
    auto torch_idx_ptr = torch_idx_cpu.data_ptr<int64_t>();
    auto gs_idx_vec = gs_idx_cpu.to_vector_int64();

    for (size_t i = 0; i < gs_idx_vec.size(); ++i) {
        EXPECT_EQ(gs_idx_vec[i], torch_idx_ptr[i])
            << "Index mismatch at position " << i;
    }
}

// ============================================================================
// SORT TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, Sort1D) {
    auto torch_data = torch::randn({20}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto [torch_sorted, torch_indices] = torch_data.sort();
    auto [gs_sorted, gs_indices] = gs_data.sort();

    EXPECT_TRUE(tensors_close(gs_sorted, torch_sorted));
}

TEST_F(TensorBugHuntingTest, Sort2DDescending) {
    auto torch_data = torch::randn({5, 10}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto [torch_sorted, torch_indices] = torch_data.sort(1, true);
    auto [gs_sorted, gs_indices] = gs_data.sort(1, true);

    EXPECT_TRUE(tensors_close(gs_sorted, torch_sorted));
}

// ============================================================================
// NONZERO TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, Nonzero2D) {
    auto torch_data = torch::tensor({{0.0f, 1.0f, 0.0f},
                                     {2.0f, 0.0f, 3.0f}},
                                    torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_nz = torch_data.nonzero();
    auto gs_nz = gs_data.nonzero();

    EXPECT_EQ(torch_nz.size(0), gs_nz.shape()[0]);
    EXPECT_EQ(torch_nz.size(1), gs_nz.shape()[1]);
}

// ============================================================================
// CDIST (PAIRWISE DISTANCE) TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, CDistL2) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto torch_b = torch::randn({8, 5}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    auto torch_dist = torch::cdist(torch_a, torch_b, 2.0);
    auto gs_dist = gs_a.cdist(gs_b, 2.0f);

    EXPECT_TRUE(tensors_close(gs_dist, torch_dist, 1e-3f))
        << "cdist L2 distance mismatch!";
}

TEST_F(TensorBugHuntingTest, CDistL1) {
    auto torch_a = torch::randn({5, 3}, torch::kCUDA);
    auto torch_b = torch::randn({7, 3}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    auto torch_dist = torch::cdist(torch_a, torch_b, 1.0);
    auto gs_dist = gs_a.cdist(gs_b, 1.0f);

    EXPECT_TRUE(tensors_close(gs_dist, torch_dist, 1e-3f));
}

// ============================================================================
// BROADCAST WITH OPERATIONS TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, BroadcastMultiplication) {
    auto torch_a = torch::randn({5, 1, 3}, torch::kCUDA);
    auto torch_b = torch::randn({1, 4, 3}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    auto torch_result = torch_a * torch_b;
    auto gs_result = gs_a * gs_b;

    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorBugHuntingTest, BroadcastComparison) {
    auto torch_a = torch::randn({10, 1}, torch::kCUDA);
    auto torch_b = torch::randn({1, 5}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    auto torch_result = torch_a > torch_b;
    auto gs_result = gs_a > gs_b;

    EXPECT_TRUE(bool_tensors_equal(gs_result, torch_result));
}

// ============================================================================
// ACCESSOR TESTS
// ============================================================================

TEST_F(TensorBugHuntingTest, Accessor2D) {
    auto torch_data = torch::randn({5, 4}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto gs_cpu = gs_data.cpu();
    auto accessor = gs_cpu.accessor<float, 2>();

    auto torch_cpu = torch_data.cpu();
    auto torch_accessor = torch_cpu.accessor<float, 2>();

    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            float gs_val = accessor(i, j);
            float torch_val = torch_accessor[i][j]; // Direct float access
            EXPECT_FLOAT_EQ(gs_val, torch_val)
                << "Mismatch at [" << i << ", " << j << "]";
        }
    }
}

// ============================================================================
// MEMORY LEAK TESTS (Basic)
// ============================================================================

TEST_F(TensorBugHuntingTest, NoMemoryLeakOnMultipleAllocations) {
    // Note: This test validates that memory pool doesn't continuously grow
    // The pool will cache allocations (by design), but should stabilize

    size_t initial_free, initial_total;
    cudaMemGetInfo(&initial_free, &initial_total);

    // Warm up the memory pool - first allocations will be cached
    for (int i = 0; i < 10; ++i) {
        auto t = lfs::core::Tensor::randn({1000, 1000}, lfs::core::Device::CUDA);
    }
    cudaDeviceSynchronize();

    size_t after_warmup_free, after_warmup_total;
    cudaMemGetInfo(&after_warmup_free, &after_warmup_total);

    // Now create and destroy many more tensors - pool should reuse cached memory
    for (int i = 0; i < 100; ++i) {
        auto t = lfs::core::Tensor::randn({1000, 1000}, lfs::core::Device::CUDA);
        // t goes out of scope, returns to pool
    }

    cudaDeviceSynchronize();

    size_t final_free, final_total;
    cudaMemGetInfo(&final_free, &final_total);

    // After warmup, memory usage should stabilize (within 50MB for pool overhead)
    // We allow more slack because the pool caches multiple sizes
    size_t additional_usage = after_warmup_free > final_free ? after_warmup_free - final_free : 0;
    EXPECT_LT(additional_usage, 50 * 1024 * 1024)
        << "Memory pool growing unexpectedly: " << (additional_usage / 1024 / 1024) << " MB after warmup";
}

// ============================================================================
// DIAGNOSTIC TESTS - Compare GS vs PyTorch Behavior
// ============================================================================

TEST_F(TensorBugHuntingTest, DiagnosticCompareSliceIsView) {
    std::cout << "\n=== Comparing slice() behavior: GS vs PyTorch ===" << std::endl;

    // PyTorch
    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    void* torch_orig_ptr = torch_data.data_ptr<float>();
    auto torch_slice = torch_data.slice(0, 0, 1);
    void* torch_slice_ptr = torch_slice.data_ptr<float>();

    bool torch_is_view = (torch_slice_ptr == torch_orig_ptr);
    std::cout << "PyTorch: slice is " << (torch_is_view ? "VIEW" : "COPY") << std::endl;

    // GS
    auto gs_data = lfs::core::Tensor::randn({5, 3}, lfs::core::Device::CUDA);
    void* gs_orig_ptr = gs_data.raw_ptr();
    auto gs_slice = gs_data.slice(0, 0, 1);
    void* gs_slice_ptr = gs_slice.raw_ptr();

    bool gs_is_view = (gs_slice_ptr == gs_orig_ptr);
    std::cout << "GS: slice is " << (gs_is_view ? "VIEW" : "COPY") << std::endl;

    EXPECT_EQ(gs_is_view, torch_is_view) << "GS slice behavior doesn't match PyTorch!";
}

TEST_F(TensorBugHuntingTest, DiagnosticCompareCloneIsCopy) {
    std::cout << "\n=== Comparing clone() behavior: GS vs PyTorch ===" << std::endl;

    // PyTorch
    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    void* torch_orig_ptr = torch_data.data_ptr<float>();
    auto torch_clone = torch_data.clone();
    void* torch_clone_ptr = torch_clone.data_ptr<float>();

    bool torch_is_copy = (torch_clone_ptr != torch_orig_ptr);
    std::cout << "PyTorch: clone is " << (torch_is_copy ? "COPY" : "VIEW") << std::endl;

    // GS
    auto gs_data = lfs::core::Tensor::randn({5, 3}, lfs::core::Device::CUDA);
    void* gs_orig_ptr = gs_data.raw_ptr();
    auto gs_clone = gs_data.clone();
    void* gs_clone_ptr = gs_clone.raw_ptr();

    bool gs_is_copy = (gs_clone_ptr != gs_orig_ptr);
    std::cout << "GS: clone is " << (gs_is_copy ? "COPY" : "VIEW") << std::endl;

    EXPECT_EQ(gs_is_copy, torch_is_copy) << "GS clone behavior doesn't match PyTorch!";
}

TEST_F(TensorBugHuntingTest, DiagnosticCompareSqueezeIsView) {
    std::cout << "\n=== Comparing squeeze() behavior: GS vs PyTorch ===" << std::endl;

    // PyTorch
    auto torch_data = torch::randn({1, 5}, torch::kCUDA);
    void* torch_orig_ptr = torch_data.data_ptr<float>();
    auto torch_squeezed = torch_data.squeeze(0);
    void* torch_squeeze_ptr = torch_squeezed.data_ptr<float>();

    bool torch_is_view = (torch_squeeze_ptr == torch_orig_ptr);
    std::cout << "PyTorch: squeeze is " << (torch_is_view ? "VIEW" : "COPY") << std::endl;

    // GS
    auto gs_data = lfs::core::Tensor::randn({1, 5}, lfs::core::Device::CUDA);
    void* gs_orig_ptr = gs_data.raw_ptr();
    auto gs_squeezed = gs_data.squeeze(0);
    void* gs_squeeze_ptr = gs_squeezed.raw_ptr();

    bool gs_is_view = (gs_squeeze_ptr == gs_orig_ptr);
    std::cout << "GS: squeeze is " << (gs_is_view ? "VIEW" : "COPY") << std::endl;

    EXPECT_EQ(gs_is_view, torch_is_view) << "GS squeeze behavior doesn't match PyTorch!";
}

TEST_F(TensorBugHuntingTest, DiagnosticCompareReshapeIsView) {
    std::cout << "\n=== Comparing reshape() behavior: GS vs PyTorch ===" << std::endl;

    // PyTorch
    auto torch_data = torch::randn({6}, torch::kCUDA);
    void* torch_orig_ptr = torch_data.data_ptr<float>();
    auto torch_reshaped = torch_data.reshape({2, 3});
    void* torch_reshape_ptr = torch_reshaped.data_ptr<float>();

    bool torch_is_view = (torch_reshape_ptr == torch_orig_ptr);
    std::cout << "PyTorch: reshape is " << (torch_is_view ? "VIEW" : "COPY") << std::endl;

    // GS
    auto gs_data = lfs::core::Tensor::randn({6}, lfs::core::Device::CUDA);
    void* gs_orig_ptr = gs_data.raw_ptr();
    auto gs_reshaped = gs_data.reshape({2, 3});
    void* gs_reshape_ptr = gs_reshaped.raw_ptr();

    bool gs_is_view = (gs_reshape_ptr == gs_orig_ptr);
    std::cout << "GS: reshape is " << (gs_is_view ? "VIEW" : "COPY") << std::endl;

    EXPECT_EQ(gs_is_view, torch_is_view) << "GS reshape behavior doesn't match PyTorch!";
}

TEST_F(TensorBugHuntingTest, DiagnosticCompareSliceSqueezeClone) {
    std::cout << "\n=== Comparing slice()->squeeze()->clone() chain: GS vs PyTorch ===" << std::endl;

    // PyTorch
    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    void* torch_orig_ptr = torch_data.data_ptr<float>();

    auto torch_sliced = torch_data.slice(0, 0, 1);
    auto torch_squeezed = torch_sliced.squeeze(0);
    auto torch_cloned = torch_squeezed.clone();

    void* torch_clone_ptr = torch_cloned.data_ptr<float>();
    bool torch_clone_independent = (torch_clone_ptr != torch_orig_ptr);

    std::cout << "PyTorch: final result is "
              << (torch_clone_independent ? "INDEPENDENT" : "SHARED MEMORY") << std::endl;

    // GS
    auto gs_data = lfs::core::Tensor::randn({5, 3}, lfs::core::Device::CUDA);
    void* gs_orig_ptr = gs_data.raw_ptr();

    auto gs_sliced = gs_data.slice(0, 0, 1);
    auto gs_squeezed = gs_sliced.squeeze(0);
    auto gs_cloned = gs_squeezed.clone();

    void* gs_clone_ptr = gs_cloned.raw_ptr();
    bool gs_clone_independent = (gs_clone_ptr != gs_orig_ptr);

    std::cout << "GS: final result is "
              << (gs_clone_independent ? "INDEPENDENT" : "SHARED MEMORY") << std::endl;

    EXPECT_EQ(gs_clone_independent, torch_clone_independent)
        << "GS chain behavior doesn't match PyTorch!";
}

TEST_F(TensorBugHuntingTest, DiagnosticCompareModificationPropagation) {
    std::cout << "\n=== Testing modification propagation: GS vs PyTorch ===" << std::endl;

    // PyTorch - test that views see modifications
    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    auto torch_slice = torch_data.slice(0, 0, 1);

    // Save original value
    float torch_orig = torch_data.index({0, 0}).item<float>();

    // Modify using fill_ (safe way to modify in PyTorch)
    torch_data.index({0, 0}).fill_(999.0f);

    // Check if slice sees the change
    float torch_slice_val = torch_slice.index({0, 0}).item<float>();
    bool torch_view_sees_change = (torch_slice_val == 999.0f);

    std::cout << "PyTorch: slice " << (torch_view_sees_change ? "SEES" : "DOESN'T SEE")
              << " modification" << std::endl;

    // GS - same test
    auto gs_data = lfs::core::Tensor::randn({5, 3}, lfs::core::Device::CUDA);
    auto gs_slice = gs_data.slice(0, 0, 1);

    // Save original value
    std::vector<float> gs_orig(1);
    cudaMemcpy(gs_orig.data(), gs_data.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);

    // Modify
    float new_val = 999.0f;
    cudaMemcpy(gs_data.ptr<float>(), &new_val, sizeof(float), cudaMemcpyHostToDevice);

    // Check if slice sees the change
    std::vector<float> gs_slice_val(1);
    cudaMemcpy(gs_slice_val.data(), gs_slice.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    bool gs_view_sees_change = (gs_slice_val[0] == 999.0f);

    std::cout << "GS: slice " << (gs_view_sees_change ? "SEES" : "DOESN'T SEE")
              << " modification" << std::endl;

    EXPECT_EQ(gs_view_sees_change, torch_view_sees_change)
        << "GS view behavior doesn't match PyTorch!";
}

TEST_F(TensorBugHuntingTest, DiagnosticCompareCloneIndependence) {
    std::cout << "\n=== Testing clone independence: GS vs PyTorch ===" << std::endl;

    // PyTorch - test that clones DON'T see modifications
    auto torch_data = torch::randn({5, 3}, torch::kCUDA);
    auto torch_clone = torch_data.clone();

    // Save clone's value
    float torch_clone_orig = torch_clone.index({0, 0}).item<float>();

    // Modify original
    torch_data.index({0, 0}).fill_(999.0f);

    // Check if clone is unaffected
    float torch_clone_after = torch_clone.index({0, 0}).item<float>();
    bool torch_clone_unaffected = (torch_clone_after == torch_clone_orig) && (torch_clone_after != 999.0f);

    std::cout << "PyTorch: clone is " << (torch_clone_unaffected ? "INDEPENDENT" : "AFFECTED")
              << std::endl;

    // GS - same test
    auto gs_data = lfs::core::Tensor::randn({5, 3}, lfs::core::Device::CUDA);
    auto gs_clone = gs_data.clone();

    // Save clone's value
    std::vector<float> gs_clone_orig(1);
    cudaMemcpy(gs_clone_orig.data(), gs_clone.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);

    // Modify original
    float new_val = 999.0f;
    cudaMemcpy(gs_data.ptr<float>(), &new_val, sizeof(float), cudaMemcpyHostToDevice);

    // Check if clone is unaffected
    std::vector<float> gs_clone_after(1);
    cudaMemcpy(gs_clone_after.data(), gs_clone.ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    bool gs_clone_unaffected = (gs_clone_after[0] == gs_clone_orig[0]) && (gs_clone_after[0] != 999.0f);

    std::cout << "GS: clone is " << (gs_clone_unaffected ? "INDEPENDENT" : "AFFECTED")
              << std::endl;

    EXPECT_EQ(gs_clone_unaffected, torch_clone_unaffected)
        << "GS clone independence doesn't match PyTorch!";
}