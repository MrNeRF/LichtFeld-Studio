/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>
#include <vector>

#include "core_new/tensor.hpp"

namespace {

    constexpr float TOLERANCE = 1e-5f;

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
        }

        return lfs::core::Tensor();
    }

    // Helper to compare tensors
    bool tensors_equal(const torch::Tensor& torch_tensor,
                       const lfs::core::Tensor& gs_tensor,
                       float tolerance = TOLERANCE) {
        if (torch_tensor.dim() != static_cast<int64_t>(gs_tensor.ndim())) {
            std::cerr << "Dimension mismatch: torch=" << torch_tensor.dim()
                      << " gs=" << gs_tensor.ndim() << std::endl;
            return false;
        }

        for (int i = 0; i < torch_tensor.dim(); ++i) {
            if (torch_tensor.size(i) != static_cast<int64_t>(gs_tensor.shape()[i])) {
                std::cerr << "Shape mismatch at dim " << i << ": torch="
                          << torch_tensor.size(i) << " gs=" << gs_tensor.shape()[i] << std::endl;
                return false;
            }
        }

        auto torch_cpu = torch_tensor.cpu().contiguous();
        auto gs_cpu = gs_tensor.cpu();

        // Float32 comparison
        if (torch_tensor.scalar_type() == torch::kFloat32 &&
            gs_tensor.dtype() == lfs::core::DataType::Float32) {
            auto torch_data = torch_cpu.data_ptr<float>();
            auto gs_data = gs_cpu.to_vector();

            if (torch_cpu.numel() != static_cast<int64_t>(gs_data.size())) {
                std::cerr << "Element count mismatch" << std::endl;
                return false;
            }

            for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
                float diff = std::abs(torch_data[i] - gs_data[i]);
                if (diff > tolerance) {
                    std::cerr << "Value mismatch at index " << i << ": torch="
                              << torch_data[i] << " gs=" << gs_data[i]
                              << " diff=" << diff << std::endl;
                    return false;
                }
            }
            return true;
        }

        // Int32 comparison
        if (torch_tensor.scalar_type() == torch::kInt32 &&
            gs_tensor.dtype() == lfs::core::DataType::Int32) {
            auto torch_data = torch_cpu.data_ptr<int>();
            auto gs_data = gs_cpu.to_vector_int();

            if (torch_cpu.numel() != static_cast<int64_t>(gs_data.size())) {
                return false;
            }

            for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
                if (torch_data[i] != gs_data[i]) {
                    std::cerr << "Value mismatch at index " << i << ": torch="
                              << torch_data[i] << " gs=" << gs_data[i] << std::endl;
                    return false;
                }
            }
            return true;
        }

        // ADDED: Int64 comparison (for nonzero() output)
        if (torch_tensor.scalar_type() == torch::kInt64 &&
            gs_tensor.dtype() == lfs::core::DataType::Int64) {
            auto torch_data = torch_cpu.data_ptr<int64_t>();
            auto gs_data = gs_cpu.to_vector_int64();

            if (torch_cpu.numel() != static_cast<int64_t>(gs_data.size())) {
                std::cerr << "Element count mismatch: torch=" << torch_cpu.numel()
                          << " gs=" << gs_data.size() << std::endl;
                return false;
            }

            for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
                if (torch_data[i] != gs_data[i]) {
                    std::cerr << "Value mismatch at index " << i << ": torch="
                              << torch_data[i] << " gs=" << gs_data[i] << std::endl;
                    return false;
                }
            }
            return true;
        }

        // ADDED: Bool comparison or cross-dtype comparison (Bool vs Float32)
        // Handle the case where torch is Float32 and gs is Bool (or vice versa)
        if ((torch_tensor.scalar_type() == torch::kFloat32 && gs_tensor.dtype() == lfs::core::DataType::Bool) ||
            (torch_tensor.scalar_type() == torch::kBool && gs_tensor.dtype() == lfs::core::DataType::Float32)) {

            // Convert both to Float32 for comparison
            auto torch_float = torch_cpu.to(torch::kFloat32);
            auto gs_float = (gs_tensor.dtype() == lfs::core::DataType::Bool)
                                ? gs_cpu.to(lfs::core::DataType::Float32)
                                : gs_cpu;

            auto torch_data = torch_float.data_ptr<float>();
            auto gs_data = gs_float.to_vector();

            if (torch_float.numel() != static_cast<int64_t>(gs_data.size())) {
                return false;
            }

            for (int64_t i = 0; i < torch_float.numel(); ++i) {
                float diff = std::abs(torch_data[i] - gs_data[i]);
                if (diff > tolerance) {
                    std::cerr << "Value mismatch at index " << i << ": torch="
                              << torch_data[i] << " gs=" << gs_data[i]
                              << " diff=" << diff << std::endl;
                    return false;
                }
            }
            return true;
        }

        // ADDED: Direct Bool comparison
        if (torch_tensor.scalar_type() == torch::kBool &&
            gs_tensor.dtype() == lfs::core::DataType::Bool) {
            auto torch_data = torch_cpu.data_ptr<bool>();
            auto gs_data = gs_cpu.to_vector_bool();

            if (torch_cpu.numel() != static_cast<int64_t>(gs_data.size())) {
                return false;
            }

            for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
                if (torch_data[i] != gs_data[i]) {
                    std::cerr << "Bool value mismatch at index " << i << ": torch="
                              << torch_data[i] << " gs=" << gs_data[i] << std::endl;
                    return false;
                }
            }
            return true;
        }

        std::cerr << "Unsupported dtype combination: torch="
                  << torch_tensor.scalar_type() << " gs=" << (int)gs_tensor.dtype() << std::endl;
        return false;
    }

    void print_tensor_comparison(const std::string& name,
                                 const torch::Tensor& torch_tensor,
                                 const lfs::core::Tensor& gs_tensor) {
        std::cout << "\n=== " << name << " ===" << std::endl;

        auto torch_cpu = torch_tensor.cpu().contiguous();
        std::cout << "Torch shape: [";
        for (int i = 0; i < torch_tensor.dim(); ++i) {
            std::cout << torch_tensor.size(i);
            if (i < torch_tensor.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "GS shape: [";
        for (size_t i = 0; i < gs_tensor.ndim(); ++i) {
            std::cout << gs_tensor.shape()[i];
            if (i < gs_tensor.ndim() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        if (torch_tensor.scalar_type() == torch::kFloat32) {
            auto torch_data = torch_cpu.data_ptr<float>();
            auto gs_cpu = gs_tensor.cpu();
            auto gs_data = gs_cpu.to_vector();

            size_t n = std::min(static_cast<size_t>(torch_cpu.numel()), size_t(20));
            std::cout << "First " << n << " values:" << std::endl;
            std::cout << "Torch: ";
            for (size_t i = 0; i < n; ++i) {
                std::cout << std::setw(10) << torch_data[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "GS:    ";
            for (size_t i = 0; i < n; ++i) {
                std::cout << std::setw(10) << gs_data[i] << " ";
            }
            std::cout << std::endl;
        }
    }

} // anonymous namespace

class TensorVsTorchTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        lfs::core::Tensor::manual_seed(42);
    }
};

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

TEST_F(TensorVsTorchTest, BasicCreation) {
    // Test zeros
    auto torch_zeros = torch::zeros({10, 5}, torch::kCUDA);
    auto gs_zeros = lfs::core::Tensor::zeros({10, 5}, lfs::core::Device::CUDA);
    EXPECT_TRUE(tensors_equal(torch_zeros, gs_zeros));

    // Test ones
    auto torch_ones = torch::ones({10, 5}, torch::kCUDA);
    auto gs_ones = lfs::core::Tensor::ones({10, 5}, lfs::core::Device::CUDA);
    EXPECT_TRUE(tensors_equal(torch_ones, gs_ones));

    // Test full
    auto torch_full = torch::full({10, 5}, 3.14f, torch::kCUDA);
    auto gs_full = lfs::core::Tensor::full({10, 5}, 3.14f, lfs::core::Device::CUDA);
    EXPECT_TRUE(tensors_equal(torch_full, gs_full));
}

TEST_F(TensorVsTorchTest, RandInt) {
    // Test randint with same seed
    torch::manual_seed(123);
    lfs::core::Tensor::manual_seed(123);

    auto torch_randint = torch::randint(0, 100, {10}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    auto gs_randint = lfs::core::Tensor::randint({10}, 0, 100, lfs::core::Device::CUDA, lfs::core::DataType::Int32);

    std::cout << "\n=== RandInt Test ===" << std::endl;
    auto torch_cpu = torch_randint.cpu();
    auto gs_cpu = gs_randint.cpu();
    std::cout << "Torch: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << torch_cpu[i].item<int>() << " ";
    }
    std::cout << std::endl;
    std::cout << "GS:    ";
    auto gs_data = gs_cpu.to_vector_int();
    for (int i = 0; i < 10; ++i) {
        std::cout << gs_data[i] << " ";
    }
    std::cout << std::endl;
}

// ============================================================================
// INDEXING AND SLICING
// ============================================================================

TEST_F(TensorVsTorchTest, RowIndexing) {
    std::cout << "\n=== Testing Row Indexing ===" << std::endl;

    // Create test data
    auto torch_data = torch::arange(0, 20, torch::kCUDA).reshape({10, 2}).to(torch::kFloat32);
    auto gs_data = torch_to_tensor(torch_data);

    // Test single row access
    for (int i = 0; i < 10; ++i) {
        auto torch_row = torch_data[i];
        auto gs_row = lfs::core::Tensor(gs_data[i]); // Convert proxy to tensor

        std::cout << "Row " << i << ":" << std::endl;
        print_tensor_comparison("Row Access", torch_row, gs_row);

        EXPECT_TRUE(tensors_equal(torch_row, gs_row))
            << "Row " << i << " mismatch";
    }
}

TEST_F(TensorVsTorchTest, RowAssignment) {
    std::cout << "\n=== Testing Row Assignment ===" << std::endl;

    // Create source and destination tensors
    auto torch_src = torch::arange(0, 20, torch::kCUDA).reshape({10, 2}).to(torch::kFloat32);
    auto torch_dst = torch::zeros({10, 2}, torch::kCUDA);

    auto gs_src = torch_to_tensor(torch_src);
    auto gs_dst = lfs::core::Tensor::zeros({10, 2}, lfs::core::Device::CUDA);

    // Assign each row
    for (int i = 0; i < 10; ++i) {
        torch_dst[i] = torch_src[i];
        gs_dst[i] = gs_src[i];
    }

    std::cout << "After assignment:" << std::endl;
    print_tensor_comparison("Assignment Result", torch_dst, gs_dst);

    EXPECT_TRUE(tensors_equal(torch_dst, gs_dst))
        << "Row assignment produced different results";
}

TEST_F(TensorVsTorchTest, SliceOperation) {
    std::cout << "\n=== Testing Slice Operation ===" << std::endl;

    auto torch_data = torch::arange(0, 30, torch::kCUDA).reshape({10, 3}).to(torch::kFloat32);
    auto gs_data = torch_to_tensor(torch_data);

    // Test slice(dim, start, end)
    auto torch_slice = torch_data.slice(0, 0, 5); // First 5 rows
    auto gs_slice = gs_data.slice(0, 0, 5);

    print_tensor_comparison("Slice [0:5, :]", torch_slice, gs_slice);
    EXPECT_TRUE(tensors_equal(torch_slice, gs_slice));

    // Test slice with different range
    torch_slice = torch_data.slice(0, 2, 7); // Rows 2-6
    gs_slice = gs_data.slice(0, 2, 7);

    print_tensor_comparison("Slice [2:7, :]", torch_slice, gs_slice);
    EXPECT_TRUE(tensors_equal(torch_slice, gs_slice));
}

// ============================================================================
// DISTANCE OPERATIONS
// ============================================================================

TEST_F(TensorVsTorchTest, CdistOperation) {
    std::cout << "\n=== Testing cdist Operation ===" << std::endl;

    // Create two sets of points
    auto torch_a = torch::tensor({{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}}, torch::kCUDA);
    auto torch_b = torch::tensor({{0.5f, 0.5f}, {1.0f, 1.0f}}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    // Compute cdist
    auto torch_dist = torch::cdist(torch_a, torch_b);
    auto gs_dist = gs_a.cdist(gs_b);

    print_tensor_comparison("cdist result", torch_dist, gs_dist);
    EXPECT_TRUE(tensors_equal(torch_dist, gs_dist, 1e-4f));

    // Test with larger data
    torch_a = torch::randn({100, 5}, torch::kCUDA);
    torch_b = torch::randn({20, 5}, torch::kCUDA);
    gs_a = torch_to_tensor(torch_a);
    gs_b = torch_to_tensor(torch_b);

    torch_dist = torch::cdist(torch_a, torch_b);
    gs_dist = gs_a.cdist(gs_b);

    EXPECT_TRUE(tensors_equal(torch_dist, gs_dist, 1e-3f))
        << "cdist failed on larger data";
}

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

TEST_F(TensorVsTorchTest, MinOperation) {
    std::cout << "\n=== Testing min Operation ===" << std::endl;

    auto torch_data = torch::rand({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // Test min along dimension 1
    auto torch_result = std::get<0>(torch::min(torch_data, 1));
    auto gs_result = gs_data.min(1);

    print_tensor_comparison("min(dim=1)", torch_result, gs_result);
    EXPECT_TRUE(tensors_equal(torch_result, gs_result));

    // Test min along dimension 0
    torch_result = std::get<0>(torch::min(torch_data, 0));
    gs_result = gs_data.min(0);

    print_tensor_comparison("min(dim=0)", torch_result, gs_result);
    EXPECT_TRUE(tensors_equal(torch_result, gs_result));
}

TEST_F(TensorVsTorchTest, SumOperation) {
    std::cout << "\n=== Testing sum Operation ===" << std::endl;

    auto torch_data = torch::tensor({{1.0f, 2.0f, 3.0f},
                                     {4.0f, 5.0f, 6.0f}},
                                    torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // Test sum of all elements
    auto torch_sum = torch_data.sum();
    auto gs_sum = gs_data.sum();

    std::cout << "Total sum - Torch: " << torch_sum.item<float>()
              << " GS: " << gs_sum.item() << std::endl;

    EXPECT_NEAR(torch_sum.item<float>(), gs_sum.item(), TOLERANCE);

    // Test sum along dimension
    auto torch_sum_dim = torch_data.sum(1);
    auto gs_sum_dim = gs_data.sum(1);

    print_tensor_comparison("sum(dim=1)", torch_sum_dim, gs_sum_dim);
    EXPECT_TRUE(tensors_equal(torch_sum_dim, gs_sum_dim));
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

TEST_F(TensorVsTorchTest, SquareOperation) {
    std::cout << "\n=== Testing square Operation ===" << std::endl;

    auto torch_data = torch::tensor({1.0f, 2.0f, 3.0f, -2.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_squared = torch_data.pow(2);
    auto gs_squared = gs_data.square();

    print_tensor_comparison("square", torch_squared, gs_squared);
    EXPECT_TRUE(tensors_equal(torch_squared, gs_squared));
}

TEST_F(TensorVsTorchTest, DivisionOperation) {
    std::cout << "\n=== Testing division Operation ===" << std::endl;

    auto torch_num = torch::tensor({10.0f, 20.0f, 30.0f}, torch::kCUDA);
    auto torch_den = torch::tensor({2.0f, 4.0f, 5.0f}, torch::kCUDA);

    auto gs_num = torch_to_tensor(torch_num);
    auto gs_den = torch_to_tensor(torch_den);

    auto torch_div = torch_num / torch_den;
    auto gs_div = gs_num.div(gs_den);

    print_tensor_comparison("division", torch_div, gs_div);
    EXPECT_TRUE(tensors_equal(torch_div, gs_div));

    // Test scalar division
    auto torch_scalar_div = torch_num / 5.0f;
    auto gs_scalar_div = gs_num.div(5.0f);

    print_tensor_comparison("scalar division", torch_scalar_div, gs_scalar_div);
    EXPECT_TRUE(tensors_equal(torch_scalar_div, gs_scalar_div));
}

// ============================================================================
// CUMULATIVE OPERATIONS
// ============================================================================

TEST_F(TensorVsTorchTest, CumsumOperation) {
    std::cout << "\n=== Testing cumsum Operation ===" << std::endl;

    auto torch_data = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_cumsum = torch_data.cumsum(0);
    auto gs_cumsum = gs_data.cumsum(0);

    print_tensor_comparison("cumsum", torch_cumsum, gs_cumsum);
    EXPECT_TRUE(tensors_equal(torch_cumsum, gs_cumsum));

    // Test 2D cumsum
    auto torch_2d = torch::tensor({{1.0f, 2.0f, 3.0f},
                                   {4.0f, 5.0f, 6.0f}},
                                  torch::kCUDA);
    auto gs_2d = torch_to_tensor(torch_2d);

    auto torch_cumsum_0 = torch_2d.cumsum(0);
    auto gs_cumsum_0 = gs_2d.cumsum(0);

    print_tensor_comparison("cumsum dim=0", torch_cumsum_0, gs_cumsum_0);
    EXPECT_TRUE(tensors_equal(torch_cumsum_0, gs_cumsum_0));

    auto torch_cumsum_1 = torch_2d.cumsum(1);
    auto gs_cumsum_1 = gs_2d.cumsum(1);

    print_tensor_comparison("cumsum dim=1", torch_cumsum_1, gs_cumsum_1);
    EXPECT_TRUE(tensors_equal(torch_cumsum_1, gs_cumsum_1));
}

// ============================================================================
// COMPARISON OPERATIONS
// ============================================================================

TEST_F(TensorVsTorchTest, BoolTensorToVector) {
    std::cout << "\n=== Testing Bool Tensor to_vector ===" << std::endl;

    // Create a Bool tensor directly
    auto bool_tensor = lfs::core::Tensor::full_bool({5}, true, lfs::core::Device::CUDA);
    std::cout << "Bool tensor created" << std::endl;
    std::cout << "Valid: " << bool_tensor.is_valid() << std::endl;
    std::cout << "Shape: " << bool_tensor.shape().str() << std::endl;
    std::cout << "Dtype: " << (int)bool_tensor.dtype() << " (5=Bool)" << std::endl;
    std::cout << "Numel: " << bool_tensor.numel() << std::endl;

    auto cpu_tensor = bool_tensor.cpu();
    std::cout << "Moved to CPU" << std::endl;
    std::cout << "CPU Valid: " << cpu_tensor.is_valid() << std::endl;
    std::cout << "CPU Numel: " << cpu_tensor.numel() << std::endl;

    // This exposes the bug - to_vector() returns empty for Bool dtype
    std::cout << "\n*** BUG DETECTED ***" << std::endl;
    auto vec = cpu_tensor.to_vector();
    std::cout << "to_vector() on Bool tensor returned size: " << vec.size() << std::endl;
    std::cout << "Expected size: " << cpu_tensor.numel() << std::endl;

    EXPECT_EQ(vec.size(), cpu_tensor.numel())
        << "BUG: to_vector() doesn't handle Bool dtype - returns empty vector!";

    // Try to_vector_bool instead
    auto bool_vec = cpu_tensor.to_vector_bool();
    std::cout << "\nto_vector_bool() returned size: " << bool_vec.size() << std::endl;

    if (!bool_vec.empty()) {
        std::cout << "Bool values: ";
        for (size_t i = 0; i < bool_vec.size(); ++i) {
            std::cout << bool_vec[i] << " ";
        }
        std::cout << std::endl;
    }
}

TEST_F(TensorVsTorchTest, ComparisonOperationsSimple) {
    std::cout << "\n=== Testing Simple Comparison ===" << std::endl;

    auto torch_simple = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kCUDA);
    auto gs_simple = torch_to_tensor(torch_simple);

    std::cout << "Testing ge(2.0f)..." << std::endl;
    auto torch_result = torch_simple >= 2.0f;
    std::cout << "Torch result dtype: bool" << std::endl;

    auto gs_result = gs_simple.ge(2.0f);
    std::cout << "GS result dtype: " << (int)gs_result.dtype() << " (5=Bool, 0=Float32)" << std::endl;

    std::cout << "\n*** COMPARING BEHAVIOR ***" << std::endl;
    std::cout << "Torch returns: bool tensor" << std::endl;
    std::cout << "GS returns: " << ((gs_result.dtype() == lfs::core::DataType::Bool) ? "Bool tensor" : "Float32 tensor") << std::endl;

    // Torch can convert Bool to Float
    auto torch_as_float = torch_result.to(torch::kFloat32);
    auto torch_cpu = torch_as_float.cpu();
    std::cout << "\nTorch Bool->Float32 values: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << torch_cpu[i].item<float>() << " ";
    }
    std::cout << std::endl;

    // Can GS convert Bool to Float?
    std::cout << "Attempting GS Bool->Float32 conversion..." << std::endl;
    auto gs_as_float = gs_result.to(lfs::core::DataType::Float32);
    auto gs_cpu = gs_as_float.cpu();
    auto gs_vec = gs_cpu.to_vector();
    std::cout << "GS Bool->Float32 values (size=" << gs_vec.size() << "): ";
    for (size_t i = 0; i < gs_vec.size(); ++i) {
        std::cout << gs_vec[i] << " ";
    }
    std::cout << std::endl;
}

TEST_F(TensorVsTorchTest, ComparisonOperations) {
    std::cout << "\n=== Testing Comparison Operations ===" << std::endl;

    auto torch_a = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, torch::kCUDA);
    auto torch_b = torch::tensor({3.0f, 3.0f, 3.0f, 3.0f, 3.0f}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    std::cout << "Input A valid: " << gs_a.is_valid() << " shape: " << gs_a.shape().str() << std::endl;
    std::cout << "Input B valid: " << gs_b.is_valid() << " shape: " << gs_b.shape().str() << std::endl;

    // Test >=
    auto torch_ge = torch_a >= torch_b;
    std::cout << "Torch >= completed, dtype: bool" << std::endl;

    auto gs_ge = gs_a.ge(gs_b);
    std::cout << "GS >= completed, dtype: " << (int)gs_ge.dtype() << " (5=Bool)" << std::endl;

    std::cout << "\nTorch >= result:" << std::endl;
    auto torch_ge_cpu = torch_ge.cpu();
    for (int i = 0; i < 5; ++i) {
        std::cout << torch_ge_cpu[i].item<bool>() << " ";
    }
    std::cout << std::endl;

    std::cout << "\n*** EXPOSING THE BUG ***" << std::endl;
    std::cout << "GS >= result - attempting to move to CPU..." << std::endl;
    auto gs_ge_cpu = gs_ge.cpu();
    std::cout << "GS >= CPU tensor valid: " << gs_ge_cpu.is_valid() << std::endl;
    std::cout << "GS >= CPU tensor numel: " << gs_ge_cpu.numel() << std::endl;
    std::cout << "GS >= CPU tensor dtype: " << (int)gs_ge_cpu.dtype() << " (5=Bool)" << std::endl;

    std::cout << "\nCalling to_vector() on Bool tensor..." << std::endl;
    auto gs_ge_data = gs_ge_cpu.to_vector();
    std::cout << "to_vector() returned size: " << gs_ge_data.size() << std::endl;
    std::cout << "Expected size: " << gs_ge_cpu.numel() << std::endl;

    // This will fail/crash because the vector is empty
    EXPECT_EQ(gs_ge_data.size(), static_cast<size_t>(5))
        << "BUG: to_vector() returns empty vector for Bool dtype!";

    if (gs_ge_data.size() >= 5) {
        std::cout << "GS >= result:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << (gs_ge_data[i] != 0.0f) << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Cannot print values - vector is empty!" << std::endl;
    }

    // This comparison will fail because of the dtype mismatch
    auto torch_ge_float = torch_ge.to(torch::kFloat32);
    // This will fail because gs_ge is Bool, not Float32
    EXPECT_TRUE(tensors_equal(torch_ge_float, gs_ge))
        << "BUG: comparison operations return Bool dtype, but tests expect Float32";
}

// ============================================================================
// NONZERO OPERATION
// ============================================================================

TEST_F(TensorVsTorchTest, NonzeroOperation) {
    std::cout << "\n=== Testing nonzero Operation ===" << std::endl;

    auto torch_data = torch::tensor({0.0f, 1.0f, 0.0f, 3.0f, 0.0f, 5.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_nz = torch_data.nonzero();
    auto gs_nz = gs_data.nonzero();

    print_tensor_comparison("nonzero", torch_nz, gs_nz);

    // Both should have same indices
    auto torch_nz_cpu = torch_nz.cpu();
    auto gs_nz_cpu = gs_nz.cpu();
    auto gs_nz_data = gs_nz_cpu.to_vector();

    std::cout << "Torch nonzero indices: ";
    for (int i = 0; i < torch_nz.size(0); ++i) {
        std::cout << torch_nz_cpu[i][0].item<int64_t>() << " ";
    }
    std::cout << std::endl;

    std::cout << "GS nonzero indices: ";
    for (size_t i = 0; i < gs_nz.shape()[0]; ++i) {
        std::cout << static_cast<int>(gs_nz_data[i]) << " ";
    }
    std::cout << std::endl;
}

// ============================================================================
// ITEM EXTRACTION
// ============================================================================

TEST_F(TensorVsTorchTest, ItemExtraction) {
    std::cout << "\n=== Testing item Extraction ===" << std::endl;

    // Test scalar item extraction
    auto torch_scalar = torch::tensor({42.0f}, torch::kCUDA);
    auto gs_scalar = torch_to_tensor(torch_scalar);

    float torch_item = torch_scalar.item<float>();
    float gs_item = gs_scalar.item();

    std::cout << "Torch item: " << torch_item << std::endl;
    std::cout << "GS item: " << gs_item << std::endl;

    EXPECT_NEAR(torch_item, gs_item, TOLERANCE);

    // Test accessing element from 1D tensor
    auto torch_1d = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kCUDA);
    auto gs_1d = torch_to_tensor(torch_1d);

    for (int i = 0; i < 3; ++i) {
        float torch_val = torch_1d[i].item<float>();
        float gs_val = gs_1d[i].item();

        std::cout << "Index " << i << " - Torch: " << torch_val
                  << " GS: " << gs_val << std::endl;

        EXPECT_NEAR(torch_val, gs_val, TOLERANCE);
    }
}

// ============================================================================
// COMPLETE KMEANS++ INITIALIZATION SIMULATION
// ============================================================================
TEST_F(TensorVsTorchTest, KMeansPlusPlusInitialization) {
    std::cout << "\n=== Testing Complete K-Means++ Initialization ===" << std::endl;

    const int n = 20;
    const int d = 2;
    const int k = 3;

    // Create identical test data
    torch::manual_seed(999);
    lfs::core::Tensor::manual_seed(999);

    auto torch_data = torch::rand({n, d}, torch::kCUDA) * 10.0f;
    auto gs_data = torch_to_tensor(torch_data);

    // Verify input data is identical
    ASSERT_TRUE(tensors_equal(torch_data, gs_data))
        << "Input data mismatch!";

    // TORCH IMPLEMENTATION
    auto torch_centroids = torch::zeros({k, d}, torch::kCUDA);
    auto torch_distances = torch::full({n}, INFINITY, torch::kCUDA);

    // GS IMPLEMENTATION
    auto gs_centroids = lfs::core::Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(d)},
                                                 lfs::core::Device::CUDA);
    auto gs_distances = lfs::core::Tensor::full({static_cast<size_t>(n)}, INFINITY,
                                                lfs::core::Device::CUDA);

    // Choose first centroid using SHARED random value
    torch::manual_seed(123);
    int first_idx = torch::randint(n, {1}, torch::kInt32).item<int>();
    std::cout << "\nShared first centroid index: " << first_idx << std::endl;

    // Use same index for both
    torch_centroids[0] = torch_data[first_idx];
    gs_centroids[0] = gs_data[first_idx];

    // Compare first centroids
    std::cout << "\nFirst centroid comparison:" << std::endl;
    print_tensor_comparison("First Centroid", torch_centroids[0], lfs::core::Tensor(gs_centroids[0]));
    EXPECT_TRUE(tensors_equal(torch_centroids[0], lfs::core::Tensor(gs_centroids[0])))
        << "First centroid mismatch";

    // Iterate for remaining centroids
    for (int c = 1; c < k; ++c) {
        std::cout << "\n--- Iteration " << c << " ---" << std::endl;

        // TORCH
        auto torch_centroid_view = torch_centroids.slice(0, 0, c);
        std::cout << "Torch centroid_view shape: [" << torch_centroid_view.size(0)
                  << ", " << torch_centroid_view.size(1) << "]" << std::endl;

        auto torch_dists = torch::cdist(torch_data, torch_centroid_view);
        std::cout << "Torch dists shape: [" << torch_dists.size(0)
                  << ", " << torch_dists.size(1) << "]" << std::endl;

        torch_distances = std::get<0>(torch::min(torch_dists, 1));
        std::cout << "Torch distances shape: [" << torch_distances.size(0) << "]" << std::endl;

        // GS
        auto gs_centroid_view = gs_centroids.slice(0, 0, c);
        std::cout << "GS centroid_view shape: [" << gs_centroid_view.shape()[0]
                  << ", " << gs_centroid_view.shape()[1] << "]" << std::endl;

        auto gs_dists = gs_data.cdist(gs_centroid_view);
        std::cout << "GS dists shape: [" << gs_dists.shape()[0]
                  << ", " << gs_dists.shape()[1] << "]" << std::endl;

        gs_distances = gs_dists.min(1);
        std::cout << "GS distances shape: [" << gs_distances.shape()[0] << "]" << std::endl;

        // Compare distances
        print_tensor_comparison("Distances", torch_distances, gs_distances);
        EXPECT_TRUE(tensors_equal(torch_distances, gs_distances, 1e-3f))
            << "Distance mismatch at iteration " << c;

        // Compute probabilities
        auto torch_probs = torch_distances.pow(2);
        torch_probs = torch_probs / torch_probs.sum();

        auto gs_probs = gs_distances.square();
        gs_probs = gs_probs.div(gs_probs.sum());

        print_tensor_comparison("Probabilities", torch_probs, gs_probs);
        EXPECT_TRUE(tensors_equal(torch_probs, gs_probs, 1e-3f))
            << "Probability mismatch at iteration " << c;

        // Cumsum
        auto torch_cumsum = torch_probs.cumsum(0);
        auto gs_cumsum = gs_probs.cumsum(0);

        print_tensor_comparison("Cumsum", torch_cumsum, gs_cumsum);
        EXPECT_TRUE(tensors_equal(torch_cumsum, gs_cumsum, 1e-3f))
            << "Cumsum mismatch at iteration " << c;

        // Sample using SHARED random value
        torch::manual_seed(c + 456);
        float shared_random = torch::rand({1}).item<float>();
        std::cout << "Shared random value: " << shared_random << std::endl;

        // Find index for torch
        auto torch_ge = torch_cumsum >= shared_random;
        auto gs_ge = gs_cumsum.ge(shared_random);

        print_tensor_comparison("GE mask", torch_ge.to(torch::kFloat32), gs_ge);

        auto torch_nz = torch_ge.nonzero();
        auto gs_nz = gs_ge.nonzero();

        std::cout << "Torch nonzero shape: [" << torch_nz.size(0) << "]" << std::endl;
        std::cout << "GS nonzero shape: [" << gs_nz.shape()[0] << "]" << std::endl;

        if (torch_nz.size(0) > 0 && gs_nz.shape()[0] > 0) {
            int torch_idx = torch_nz[0][0].item<int64_t>();

            // FIXED: Properly extract Int64 value from 2D tensor
            auto gs_nz_cpu = gs_nz.cpu();
            auto gs_nz_data = gs_nz_cpu.to_vector_int64();
            int gs_idx = static_cast<int>(gs_nz_data[0]);

            std::cout << "Selected index - Torch: " << torch_idx << " GS: " << gs_idx << std::endl;

            // Both should select the same index since they use the same random value
            EXPECT_EQ(torch_idx, gs_idx)
                << "Index selection mismatch at iteration " << c;

            torch_centroids[c] = torch_data[torch_idx];
            gs_centroids[c] = gs_data[gs_idx];

            print_tensor_comparison("New Centroid", torch_centroids[c], lfs::core::Tensor(gs_centroids[c]));
            EXPECT_TRUE(tensors_equal(torch_centroids[c], lfs::core::Tensor(gs_centroids[c])))
                << "New centroid mismatch at iteration " << c;
        }
    }

    // Final comparison
    std::cout << "\n=== Final Centroids Comparison ===" << std::endl;
    print_tensor_comparison("Final Centroids", torch_centroids, gs_centroids);
    EXPECT_TRUE(tensors_equal(torch_centroids, gs_centroids, 1e-3f))
        << "Final centroids mismatch";
}

TEST_F(TensorVsTorchTest, RandomGenerationComparison) {
    std::cout << "\n=== Testing Random Generation ===" << std::endl;

    // Test 1: Same seed produces same sequence?
    std::cout << "\n--- Test 1: Reproducibility ---" << std::endl;

    lfs::core::Tensor::manual_seed(42);
    auto gs_rand1 = lfs::core::Tensor::rand({10}, lfs::core::Device::CUDA);

    lfs::core::Tensor::manual_seed(42);
    auto gs_rand2 = lfs::core::Tensor::rand({10}, lfs::core::Device::CUDA);

    auto gs_rand1_vec = gs_rand1.cpu().to_vector();
    auto gs_rand2_vec = gs_rand2.cpu().to_vector();

    bool same = true;
    for (size_t i = 0; i < gs_rand1_vec.size(); ++i) {
        if (std::abs(gs_rand1_vec[i] - gs_rand2_vec[i]) > 1e-6f) {
            same = false;
            break;
        }
    }
    EXPECT_TRUE(same) << "GS random generation is not reproducible with same seed!";

    // Test 2: Different seeds produce different sequences?
    std::cout << "\n--- Test 2: Different Seeds ---" << std::endl;

    lfs::core::Tensor::manual_seed(42);
    auto gs_rand_a = lfs::core::Tensor::rand({10}, lfs::core::Device::CUDA);

    lfs::core::Tensor::manual_seed(999);
    auto gs_rand_b = lfs::core::Tensor::rand({10}, lfs::core::Device::CUDA);

    auto gs_rand_a_vec = gs_rand_a.cpu().to_vector();
    auto gs_rand_b_vec = gs_rand_b.cpu().to_vector();

    bool different = false;
    for (size_t i = 0; i < gs_rand_a_vec.size(); ++i) {
        if (std::abs(gs_rand_a_vec[i] - gs_rand_b_vec[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Different seeds produce identical sequences!";

    // Test 3: Distribution properties (uniform [0,1])
    std::cout << "\n--- Test 3: Distribution Properties ---" << std::endl;

    lfs::core::Tensor::manual_seed(123);
    auto gs_large = lfs::core::Tensor::rand({10000}, lfs::core::Device::CUDA);

    float mean = gs_large.mean().item();
    float min_val = gs_large.min().item();
    float max_val = gs_large.max().item();
    float std_val = gs_large.std().item();

    std::cout << "Mean: " << mean << " (expected ~0.5)" << std::endl;
    std::cout << "Min:  " << min_val << " (expected ~0.0)" << std::endl;
    std::cout << "Max:  " << max_val << " (expected ~1.0)" << std::endl;
    std::cout << "Std:  " << std_val << " (expected ~0.289)" << std::endl;

    EXPECT_NEAR(mean, 0.5f, 0.05f) << "Mean far from expected 0.5";
    EXPECT_GE(min_val, 0.0f) << "Values below 0";
    EXPECT_LE(max_val, 1.0f) << "Values above 1";
    EXPECT_NEAR(std_val, 0.289f, 0.05f) << "Std dev far from expected";

    // Test 4: randint distribution
    std::cout << "\n--- Test 4: randint Distribution ---" << std::endl;

    lfs::core::Tensor::manual_seed(456);
    auto gs_int = lfs::core::Tensor::randint({1000}, 0, 10, lfs::core::Device::CUDA, lfs::core::DataType::Int32);

    auto gs_int_cpu = gs_int.cpu();
    auto gs_int_vec = gs_int_cpu.to_vector_int();

    std::vector<int> counts(10, 0);
    for (int val : gs_int_vec) {
        if (val >= 0 && val < 10) {
            counts[val]++;
        }
    }

    std::cout << "Distribution:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Value " << i << ": " << counts[i] << " times";
        std::cout << " (" << (counts[i] * 100.0f / 1000.0f) << "%)" << std::endl;
    }

    // Each value should appear roughly 100 times (10% of 1000)
    for (int i = 0; i < 10; ++i) {
        EXPECT_GE(counts[i], 50) << "Value " << i << " appears too rarely";
        EXPECT_LE(counts[i], 150) << "Value " << i << " appears too often";
    }

    // Test 5: randn (normal distribution)
    std::cout << "\n--- Test 5: randn Normal Distribution ---" << std::endl;

    lfs::core::Tensor::manual_seed(789);
    auto gs_normal = lfs::core::Tensor::randn({10000}, lfs::core::Device::CUDA);

    float normal_mean = gs_normal.mean().item();
    float normal_std = gs_normal.std().item();

    std::cout << "Mean: " << normal_mean << " (expected ~0.0)" << std::endl;
    std::cout << "Std:  " << normal_std << " (expected ~1.0)" << std::endl;

    EXPECT_NEAR(normal_mean, 0.0f, 0.05f) << "Normal mean far from 0";
    EXPECT_NEAR(normal_std, 1.0f, 0.1f) << "Normal std far from 1";
}

// ============================================================================
// TENSOR ROW PROXY SCALAR EXTRACTION TESTS
// ============================================================================

TEST_F(TensorVsTorchTest, RowProxyScalarExtraction1D) {
    std::cout << "\n=== Testing Row Proxy Scalar Extraction (1D) ===" << std::endl;

    // Create 1D tensors
    auto torch_1d = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, torch::kCUDA);
    auto gs_1d = torch_to_tensor(torch_1d);

    std::cout << "Testing implicit float conversion for 1D tensors..." << std::endl;

    // Test implicit conversion to float
    for (int i = 0; i < 5; ++i) {
        float torch_val = torch_1d[i].item<float>();
        float gs_val = gs_1d[i]; // Should work with implicit conversion!

        std::cout << "Index " << i << " - Torch: " << torch_val
                  << " GS: " << gs_val << std::endl;

        EXPECT_NEAR(torch_val, gs_val, TOLERANCE)
            << "Scalar extraction mismatch at index " << i;
    }

    // Test explicit item() method
    std::cout << "\nTesting explicit item() method..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        float torch_val = torch_1d[i].item<float>();
        float gs_val = gs_1d[i].item();

        EXPECT_NEAR(torch_val, gs_val, TOLERANCE)
            << "Explicit item() mismatch at index " << i;
    }
}

TEST_F(TensorVsTorchTest, RowProxyInt64Extraction) {
    std::cout << "\n=== Testing Row Proxy Int64 Extraction ===" << std::endl;

    // Create Int64 tensor (like nonzero() output)
    auto torch_int64 = torch::tensor({10, 20, 30, 40, 50},
                                     torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto gs_int64 = lfs::core::Tensor::from_vector({10, 20, 30, 40, 50}, {5}, lfs::core::Device::CUDA)
                        .to(lfs::core::DataType::Int64);

    std::cout << "Testing item_int64() method..." << std::endl;

    for (int i = 0; i < 5; ++i) {
        int64_t torch_val = torch_int64[i].item<int64_t>();
        int64_t gs_val = gs_int64[i].item_int64();

        std::cout << "Index " << i << " - Torch: " << torch_val
                  << " GS: " << gs_val << std::endl;

        EXPECT_EQ(torch_val, gs_val)
            << "Int64 extraction mismatch at index " << i;
    }
}

TEST_F(TensorVsTorchTest, RowProxyInt32Extraction) {
    std::cout << "\n=== Testing Row Proxy Int32 Extraction ===" << std::endl;

    auto torch_int32 = torch::tensor({100, 200, 300},
                                     torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_int32 = lfs::core::Tensor::from_vector({100, 200, 300}, {3}, lfs::core::Device::CUDA)
                        .to(lfs::core::DataType::Int32);

    std::cout << "Testing item_int() method..." << std::endl;

    for (int i = 0; i < 3; ++i) {
        int torch_val = torch_int32[i].item<int>();
        int gs_val = gs_int32[i].item_int();

        std::cout << "Index " << i << " - Torch: " << torch_val
                  << " GS: " << gs_val << std::endl;

        EXPECT_EQ(torch_val, gs_val)
            << "Int32 extraction mismatch at index " << i;
    }
}

TEST_F(TensorVsTorchTest, RowProxy2DNoImplicitConversion) {
    std::cout << "\n=== Testing Row Proxy 2D (No Implicit Conversion) ===" << std::endl;

    auto torch_2d = torch::rand({5, 3}, torch::kCUDA);
    auto gs_2d = torch_to_tensor(torch_2d);

    std::cout << "Testing that 2D tensors don't allow implicit float conversion..." << std::endl;

    // This should work: explicit Tensor conversion
    for (int i = 0; i < 5; ++i) {
        auto torch_row = torch_2d[i];
        lfs::core::Tensor gs_row = gs_2d[i]; // Explicit conversion

        EXPECT_TRUE(tensors_equal(torch_row, gs_row))
            << "2D row extraction mismatch at index " << i;
    }

    // 2D element access should still work
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            float torch_val = torch_2d[i][j].item<float>();
            float gs_val = gs_2d[i][j];

            EXPECT_NEAR(torch_val, gs_val, TOLERANCE)
                << "2D element access mismatch at [" << i << "][" << j << "]";
        }
    }
}

TEST_F(TensorVsTorchTest, RowProxyNonzeroIntegration) {
    std::cout << "\n=== Testing Row Proxy with Nonzero Output ===" << std::endl;

    // Create data with some zeros
    auto torch_data = torch::tensor({0.0f, 1.0f, 0.0f, 3.0f, 0.0f, 5.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // Get nonzero indices
    auto torch_nz = torch_data.nonzero(); // Returns [count, 1] Int64 tensor
    auto gs_nz = gs_data.nonzero();       // Returns [count, 1] Int64 tensor

    std::cout << "Torch nonzero shape: [" << torch_nz.size(0) << ", " << torch_nz.size(1) << "]" << std::endl;
    std::cout << "GS nonzero shape: [" << gs_nz.shape()[0] << ", " << gs_nz.shape()[1] << "]" << std::endl;

    ASSERT_EQ(torch_nz.size(0), static_cast<int64_t>(gs_nz.shape()[0]))
        << "Nonzero count mismatch";

    std::cout << "\nExtracting first nonzero index using row proxy..." << std::endl;

    // Extract first index using row proxy
    int64_t torch_first_idx = torch_nz[0][0].item<int64_t>();
    int64_t gs_first_idx = gs_nz[0].item_int64(); // Use item_int64() on row proxy

    std::cout << "Torch first index: " << torch_first_idx << std::endl;
    std::cout << "GS first index: " << gs_first_idx << std::endl;

    EXPECT_EQ(torch_first_idx, gs_first_idx)
        << "First nonzero index mismatch";

    // Verify all indices match
    std::cout << "\nComparing all indices..." << std::endl;
    for (int i = 0; i < torch_nz.size(0); ++i) {
        int64_t torch_idx = torch_nz[i][0].item<int64_t>();
        int64_t gs_idx = gs_nz[i].item_int64();

        std::cout << "Position " << i << " - Torch: " << torch_idx
                  << " GS: " << gs_idx << std::endl;

        EXPECT_EQ(torch_idx, gs_idx)
            << "Nonzero index mismatch at position " << i;
    }
}

TEST_F(TensorVsTorchTest, KMeansPlusPlusWithRowProxy) {
    std::cout << "\n=== Testing K-Means++ with Row Proxy Extraction ===" << std::endl;

    const int n = 100;
    const int d = 3;
    const int k = 5;

    // Create identical test data
    torch::manual_seed(12345);
    lfs::core::Tensor::manual_seed(12345);

    auto torch_data = torch::rand({n, d}, torch::kCUDA) * 100.0f;
    auto gs_data = torch_to_tensor(torch_data);

    ASSERT_TRUE(tensors_equal(torch_data, gs_data))
        << "Input data mismatch!";

    // Initialize centroids
    auto torch_centroids = torch::zeros({k, d}, torch::kCUDA);
    auto gs_centroids = lfs::core::Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(d)},
                                                 lfs::core::Device::CUDA);

    // Choose first centroid
    torch::manual_seed(999);
    int first_idx = torch::randint(n, {1}, torch::kInt32).item<int>();

    torch_centroids[0] = torch_data[first_idx];
    gs_centroids[0] = gs_data[first_idx];

    // Select remaining centroids
    for (int c = 1; c < k; ++c) {
        std::cout << "\n--- Selecting centroid " << c << " ---" << std::endl;

        // Compute distances
        auto torch_centroid_view = torch_centroids.slice(0, 0, c);
        auto torch_dists = torch::cdist(torch_data, torch_centroid_view);
        auto torch_min_dists = std::get<0>(torch::min(torch_dists, 1));

        auto gs_centroid_view = gs_centroids.slice(0, 0, c);
        auto gs_dists = gs_data.cdist(gs_centroid_view);
        auto gs_min_dists = gs_dists.min(1);

        // Compute probabilities
        auto torch_probs = torch_min_dists.pow(2);
        torch_probs = torch_probs / torch_probs.sum();

        auto gs_probs = gs_min_dists.square();
        gs_probs = gs_probs.div(gs_probs.sum());

        // Cumsum
        auto torch_cumsum = torch_probs.cumsum(0);
        auto gs_cumsum = gs_probs.cumsum(0);

        // Sample with shared random value
        torch::manual_seed(c * 1000);
        float rand_val = torch::rand({1}).item<float>();

        // Find index
        auto torch_ge = torch_cumsum >= rand_val;
        auto gs_ge = gs_cumsum.ge(rand_val);

        auto torch_indices = torch_ge.nonzero();
        auto gs_indices = gs_ge.nonzero();

        if (torch_indices.size(0) > 0 && gs_indices.numel() > 0) {
            // CRITICAL TEST: Use row proxy for clean extraction
            int64_t torch_next_idx = torch_indices[0][0].item<int64_t>();
            int64_t gs_next_idx = gs_indices[0].item_int64(); // Clean row proxy usage!

            std::cout << "Selected index - Torch: " << torch_next_idx
                      << " GS: " << gs_next_idx << std::endl;

            EXPECT_EQ(torch_next_idx, gs_next_idx)
                << "Index selection mismatch at iteration " << c;

            // Assign new centroids
            torch_centroids[c] = torch_data[torch_next_idx];
            gs_centroids[c] = gs_data[gs_next_idx];
        }
    }

    // Final comparison
    std::cout << "\n=== Final Centroids ===" << std::endl;
    print_tensor_comparison("Centroids", torch_centroids, gs_centroids);

    EXPECT_TRUE(tensors_equal(torch_centroids, gs_centroids, 1e-3f))
        << "Final centroids mismatch - row proxy extraction may have failed";
}

TEST_F(TensorVsTorchTest, RowProxyArithmeticOperations) {
    std::cout << "\n=== Testing Row Proxy Arithmetic Operations ===" << std::endl;

    auto torch_data = torch::tensor({2.0f, 4.0f, 6.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    std::cout << "\nTesting arithmetic with scalars..." << std::endl;

    // Test row proxy arithmetic operations
    float torch_val = torch_data[1].item<float>(); // 4.0
    float gs_val = gs_data[1];                     // 4.0

    EXPECT_NEAR(torch_val, gs_val, TOLERANCE);

    // Test arithmetic on proxy
    auto torch_result = torch_data[1] * 2.0f;
    auto gs_result = gs_data[1] * 2.0f; // Row proxy * scalar returns Tensor

    std::cout << "Torch: 4.0 * 2.0 = " << torch_result.item<float>() << std::endl;
    std::cout << "GS: 4.0 * 2.0 = " << gs_result.item() << std::endl;

    EXPECT_NEAR(torch_result.item<float>(), gs_result.item(), TOLERANCE);

    // Test square
    torch_result = torch_data[2].pow(2);
    gs_result = gs_data[2].square();

    std::cout << "Torch: 6.0^2 = " << torch_result.item<float>() << std::endl;
    std::cout << "GS: 6.0^2 = " << gs_result.item() << std::endl;

    EXPECT_NEAR(torch_result.item<float>(), gs_result.item(), TOLERANCE);
}

TEST_F(TensorVsTorchTest, RowProxyComparisonWithOtherProxy) {
    std::cout << "\n=== Testing Row Proxy Comparison with Other Proxy ===" << std::endl;

    auto torch_a = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kCUDA);
    auto torch_b = torch::tensor({2.0f, 2.0f, 2.0f}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    std::cout << "\nTesting proxy-to-proxy subtraction..." << std::endl;

    // Test proxy - proxy
    for (int i = 0; i < 3; ++i) {
        auto torch_diff = torch_a[i] - torch_b[i];
        auto gs_diff = gs_a[i] - gs_b[i];

        float torch_val = torch_diff.item<float>();
        float gs_val = gs_diff.item();

        std::cout << "Index " << i << " - Torch: " << torch_val
                  << " GS: " << gs_val << std::endl;

        EXPECT_NEAR(torch_val, gs_val, TOLERANCE)
            << "Proxy-to-proxy subtraction mismatch at index " << i;
    }
}

TEST_F(TensorVsTorchTest, RowProxyAssignmentFromProxy) {
    std::cout << "\n=== Testing Row Proxy Assignment from Another Proxy ===" << std::endl;

    auto torch_src = torch::arange(0, 10, torch::kCUDA).to(torch::kFloat32);
    auto torch_dst = torch::zeros({10}, torch::kCUDA);

    auto gs_src = torch_to_tensor(torch_src);
    auto gs_dst = lfs::core::Tensor::zeros({10}, lfs::core::Device::CUDA);

    std::cout << "\nTesting dst[4] = src[0] pattern..." << std::endl;

    // Critical test: assign from one proxy to another
    torch_dst[4] = torch_src[0];
    gs_dst[4] = gs_src[0]; // Should properly clone the data

    // Verify
    float torch_val = torch_dst[4].item<float>();
    float gs_val = gs_dst[4];

    std::cout << "After dst[4] = src[0]:" << std::endl;
    std::cout << "Torch dst[4]: " << torch_val << std::endl;
    std::cout << "GS dst[4]: " << gs_val << std::endl;

    EXPECT_NEAR(torch_val, 0.0f, TOLERANCE);
    EXPECT_NEAR(gs_val, 0.0f, TOLERANCE);

    // Verify full tensors match
    EXPECT_TRUE(tensors_equal(torch_dst, gs_dst))
        << "Proxy-to-proxy assignment failed";
}

TEST_F(TensorVsTorchTest, RowProxyEdgeCases) {
    std::cout << "\n=== Testing Row Proxy Edge Cases ===" << std::endl;

    // Test with scalar (0D) tensor
    std::cout << "\n1. Scalar tensor test..." << std::endl;
    auto torch_scalar = torch::tensor(42.0f, torch::kCUDA);
    auto gs_scalar = lfs::core::Tensor::full({1}, 42.0f, lfs::core::Device::CUDA).squeeze();

    float torch_item = torch_scalar.item<float>();
    float gs_item = gs_scalar.item();

    EXPECT_NEAR(torch_item, gs_item, TOLERANCE);

    // Test with empty dimensions
    std::cout << "\n2. Tensor with size-1 dimension..." << std::endl;
    auto torch_single = torch::tensor({5.0f}, torch::kCUDA);
    auto gs_single = torch_to_tensor(torch_single);

    float torch_val = torch_single[0].item<float>();
    float gs_val = gs_single[0]; // Implicit conversion for 1D

    EXPECT_NEAR(torch_val, 5.0f, TOLERANCE);
    EXPECT_NEAR(gs_val, 5.0f, TOLERANCE);

    // Test with large indices
    std::cout << "\n3. Large tensor indexing..." << std::endl;
    auto torch_large = torch::arange(0, 10000, torch::kCUDA).to(torch::kFloat32);
    auto gs_large = torch_to_tensor(torch_large);

    int test_idx = 9999;
    float torch_last = torch_large[test_idx].item<float>();
    float gs_last = gs_large[test_idx];

    std::cout << "Index " << test_idx << " - Torch: " << torch_last
              << " GS: " << gs_last << std::endl;

    EXPECT_NEAR(torch_last, static_cast<float>(test_idx), TOLERANCE);
    EXPECT_NEAR(gs_last, static_cast<float>(test_idx), TOLERANCE);
}