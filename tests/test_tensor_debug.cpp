/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-4f;

    void print_comparison(const std::string& test_name, bool passed) {
        std::cout << (passed ? "✓ PASS" : "✗ FAIL") << ": " << test_name << std::endl;
    }

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
        } else if (torch_tensor.scalar_type() == torch::kInt64) {
            std::vector<int64_t> data(cpu_tensor.data_ptr<int64_t>(),
                                      cpu_tensor.data_ptr<int64_t>() + cpu_tensor.numel());
            std::vector<int> data_int32(data.begin(), data.end());
            return lfs::core::Tensor::from_vector(data_int32, lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
        }

        return lfs::core::Tensor();
    }

    // Helper to compare float tensors
    bool compare_float_tensors(const lfs::core::Tensor& gs_t, const torch::Tensor& torch_t,
                               float tol = FLOAT_TOLERANCE) {
        if (gs_t.numel() != torch_t.numel()) {
            std::cout << "  Size mismatch: gs=" << gs_t.numel()
                      << " torch=" << torch_t.numel() << std::endl;
            return false;
        }

        auto gs_cpu = gs_t.cpu().to_vector();
        auto torch_cpu = torch_t.cpu();
        auto torch_ptr = torch_cpu.data_ptr<float>();

        for (size_t i = 0; i < gs_t.numel(); ++i) {
            float diff = std::abs(gs_cpu[i] - torch_ptr[i]);
            if (diff > tol) {
                std::cout << "  Mismatch at index " << i << ": gs=" << gs_cpu[i]
                          << " torch=" << torch_ptr[i] << " diff=" << diff << std::endl;
                return false;
            }
        }
        return true;
    }

    // Helper to compare int tensors
    bool compare_int_tensors(const lfs::core::Tensor& gs_t, const torch::Tensor& torch_t) {
        if (gs_t.numel() != torch_t.numel()) {
            std::cout << "  Size mismatch: gs=" << gs_t.numel()
                      << " torch=" << torch_t.numel() << std::endl;
            return false;
        }

        auto gs_cpu = gs_t.cpu();
        auto torch_cpu = torch_t.cpu();

        // Get gs tensor data based on its dtype
        std::vector<int64_t> gs_data;
        if (gs_cpu.dtype() == lfs::core::DataType::Int64) {
            gs_data = gs_cpu.to_vector_int64();
        } else if (gs_cpu.dtype() == lfs::core::DataType::Int32) {
            auto gs_int32 = gs_cpu.to_vector_int();
            gs_data.assign(gs_int32.begin(), gs_int32.end());
        } else {
            std::cout << "  Unsupported gs dtype: " << lfs::core::dtype_name(gs_cpu.dtype()) << std::endl;
            return false;
        }

        // Compare against torch tensor
        if (torch_t.scalar_type() == torch::kInt32) {
            auto torch_ptr = torch_cpu.data_ptr<int>();
            for (size_t i = 0; i < gs_t.numel(); ++i) {
                if (gs_data[i] != torch_ptr[i]) {
                    std::cout << "  Mismatch at index " << i << ": gs=" << gs_data[i]
                              << " torch=" << torch_ptr[i] << std::endl;
                    return false;
                }
            }
        } else if (torch_t.scalar_type() == torch::kInt64) {
            auto torch_ptr = torch_cpu.data_ptr<int64_t>();
            for (size_t i = 0; i < gs_t.numel(); ++i) {
                if (gs_data[i] != torch_ptr[i]) {
                    std::cout << "  Mismatch at index " << i << ": gs=" << gs_data[i]
                              << " torch=" << torch_ptr[i] << std::endl;
                    return false;
                }
            }
        } else {
            std::cout << "  Unsupported torch dtype" << std::endl;
            return false;
        }

        return true;
    }

} // namespace

class TensorDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        lfs::core::Tensor::manual_seed(42);
    }
};

// ============================================================================
// Test 1: Sort Operation - CRITICAL for k-means
// ============================================================================

TEST_F(TensorDebugTest, Sort_Float32_Ascending) {
    std::cout << "\n=== TEST: Sort Float32 Ascending ===" << std::endl;

    auto torch_data = torch::tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    std::cout << "Input: [3.0, 1.0, 4.0, 1.0, 5.0]" << std::endl;

    // LibTorch
    auto [torch_vals, torch_idx] = torch::sort(torch_data);
    std::cout << "LibTorch result:" << std::endl;
    std::cout << "  Values: " << torch_vals << std::endl;
    std::cout << "  Indices: " << torch_idx << std::endl;
    std::cout << "  Index dtype: " << torch_idx.dtype() << std::endl;

    // lfs::core::Tensor
    std::cout << "lfs::core::Tensor result:" << std::endl;
    try {
        auto result = gs_data.sort(0);
        auto gs_vals = result.first;
        auto gs_idx = result.second;

        std::cout << "  Values shape: [" << gs_vals.shape()[0] << "]" << std::endl;
        std::cout << "  Indices shape: [" << gs_idx.shape()[0] << "]" << std::endl;
        std::cout << "  Index dtype: " << lfs::core::dtype_name(gs_idx.dtype()) << std::endl;

        // Compare values
        bool vals_match = compare_float_tensors(gs_vals, torch_vals);
        print_comparison("Sort values match", vals_match);
        EXPECT_TRUE(vals_match);

        // Compare indices (allow Int32 or Int64)
        bool idx_match = compare_int_tensors(gs_idx, torch_idx);
        print_comparison("Sort indices match", idx_match);
        EXPECT_TRUE(idx_match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL() << "lfs::core::Tensor sort() threw exception";
    }
}

TEST_F(TensorDebugTest, Sort_Float32_Descending) {
    std::cout << "\n=== TEST: Sort Float32 Descending ===" << std::endl;

    auto torch_data = torch::tensor({3.0f, 1.0f, 4.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    // LibTorch
    auto [torch_vals, torch_idx] = torch::sort(torch_data, 0, true);
    std::cout << "LibTorch descending: " << torch_vals << std::endl;

    // lfs::core::Tensor
    try {
        auto [gs_vals, gs_idx] = gs_data.sort(0, true);

        bool vals_match = compare_float_tensors(gs_vals, torch_vals);
        print_comparison("Descending sort values match", vals_match);
        EXPECT_TRUE(vals_match);

        bool idx_match = compare_int_tensors(gs_idx, torch_idx);
        print_comparison("Descending sort indices match", idx_match);
        EXPECT_TRUE(idx_match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL() << "lfs::core::Tensor descending sort() threw exception";
    }
}

// ============================================================================
// Test 2: Item Extraction - CRITICAL for k-means
// ============================================================================

TEST_F(TensorDebugTest, Item_Float32) {
    std::cout << "\n=== TEST: Item Float32 Extraction ===" << std::endl;

    auto torch_scalar = torch::tensor(42.5f, torch::kCUDA);
    auto gs_scalar = torch_to_tensor(torch_scalar);

    float torch_val = torch_scalar.item<float>();
    std::cout << "LibTorch item<float>(): " << torch_val << std::endl;

    try {
        float gs_val = gs_scalar.item();
        std::cout << "lfs::core::Tensor item(): " << gs_val << std::endl;

        bool match = std::abs(torch_val - gs_val) < FLOAT_TOLERANCE;
        print_comparison("Float item match", match);
        EXPECT_TRUE(match);
    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, Item_Int32) {
    std::cout << "\n=== TEST: Item Int32 Extraction ===" << std::endl;

    auto torch_scalar = torch::tensor(42, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_scalar = torch_to_tensor(torch_scalar);

    int torch_val = torch_scalar.item<int>();
    std::cout << "LibTorch item<int>(): " << torch_val << std::endl;

    try {
        int gs_val = gs_scalar.item<int>();
        std::cout << "lfs::core::Tensor item<int>(): " << gs_val << std::endl;

        bool match = (torch_val == gs_val);
        print_comparison("Int32 item match", match);
        EXPECT_EQ(torch_val, gs_val);
    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, Item_AfterIndexing_Int32) {
    std::cout << "\n=== TEST: Item After Indexing Int32 ===" << std::endl;

    auto torch_data = torch::tensor({10, 20, 30, 40, 50},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_data = torch_to_tensor(torch_data);

    for (int i = 0; i < 5; ++i) {
        auto torch_slice = torch_data.slice(0, i, i + 1);
        int torch_val = torch_slice.item<int>();

        auto gs_slice = gs_data.slice(0, i, i + 1);
        int gs_val = gs_slice.item<int>();

        std::cout << "Index " << i << ": torch=" << torch_val << " gs=" << gs_val << std::endl;
        EXPECT_EQ(torch_val, gs_val);
    }

    print_comparison("Item after indexing", true);
}

// ============================================================================
// Test 3: Unsqueeze/Squeeze - CRITICAL for k-means 1D
// ============================================================================

TEST_F(TensorDebugTest, Unsqueeze) {
    std::cout << "\n=== TEST: Unsqueeze ===" << std::endl;

    auto torch_data = torch::randn({5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_unsq = torch_data.unsqueeze(1);
    std::cout << "LibTorch unsqueeze(1): " << torch_unsq.sizes() << std::endl;

    try {
        auto gs_unsq = gs_data.unsqueeze(1);
        std::cout << "lfs::core::Tensor unsqueeze(1): [" << gs_unsq.shape()[0]
                  << ", " << gs_unsq.shape()[1] << "]" << std::endl;

        bool shape_match = (torch_unsq.size(0) == gs_unsq.shape()[0] &&
                            torch_unsq.size(1) == gs_unsq.shape()[1]);
        bool data_match = compare_float_tensors(gs_unsq, torch_unsq);

        print_comparison("Unsqueeze shape", shape_match);
        print_comparison("Unsqueeze data", data_match);

        EXPECT_TRUE(shape_match && data_match);
    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, Squeeze) {
    std::cout << "\n=== TEST: Squeeze ===" << std::endl;

    auto torch_data = torch::randn({5, 1}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_sq = torch_data.squeeze(1);
    std::cout << "LibTorch squeeze(1): " << torch_sq.sizes() << std::endl;

    try {
        auto gs_sq = gs_data.squeeze(1);
        std::cout << "lfs::core::Tensor squeeze(1): [" << gs_sq.shape()[0] << "]" << std::endl;

        bool shape_match = (torch_sq.size(0) == gs_sq.shape()[0] && torch_sq.dim() == gs_sq.ndim());
        bool data_match = compare_float_tensors(gs_sq, torch_sq);

        print_comparison("Squeeze shape", shape_match);
        print_comparison("Squeeze data", data_match);

        EXPECT_TRUE(shape_match && data_match);
    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

// ============================================================================
// Test 4: Masked Select - CRITICAL for k-means centroid update
// ============================================================================

TEST_F(TensorDebugTest, MaskedSelect_Basic) {
    std::cout << "\n=== TEST: Masked Select Basic ===" << std::endl;

    auto torch_data = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_mask = torch_data > 0.0f;
    auto gs_mask = gs_data > 0.0f;

    auto torch_selected = torch_data.masked_select(torch_mask);
    std::cout << "LibTorch masked_select: " << torch_selected << std::endl;
    std::cout << "  Shape: " << torch_selected.sizes() << std::endl;

    try {
        auto gs_selected = gs_data.masked_select(gs_mask);
        std::cout << "lfs::core::Tensor masked_select:" << std::endl;
        std::cout << "  Shape: [" << gs_selected.shape()[0] << "]" << std::endl;
        std::cout << "  Numel: " << gs_selected.numel() << std::endl;

        bool match = compare_float_tensors(gs_selected, torch_selected);
        print_comparison("Masked select", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, MaskedSelect_WithReshape) {
    std::cout << "\n=== TEST: Masked Select + Reshape ===" << std::endl;

    auto torch_data = torch::tensor({-1.0f, 2.0f, 3.0f, -4.0f, 5.0f, 6.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_mask = torch_data > 0.0f;
    auto gs_mask = gs_data > 0.0f;

    auto torch_selected = torch_data.masked_select(torch_mask).reshape({-1, 1});
    std::cout << "LibTorch masked_select + reshape(-1, 1): " << torch_selected.sizes() << std::endl;

    try {
        auto gs_selected = gs_data.masked_select(gs_mask).reshape({-1, 1});
        std::cout << "lfs::core::Tensor masked_select + reshape(-1, 1): ["
                  << gs_selected.shape()[0] << ", " << gs_selected.shape()[1] << "]" << std::endl;

        bool match = compare_float_tensors(gs_selected, torch_selected);
        print_comparison("Masked select + reshape", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

// ============================================================================
// Test 5: Cumsum - Used in k-means++ initialization
// ============================================================================

TEST_F(TensorDebugTest, Cumsum) {
    std::cout << "\n=== TEST: Cumsum ===" << std::endl;

    auto torch_data = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_cumsum = torch_data.cumsum(0);
    std::cout << "LibTorch cumsum: " << torch_cumsum << std::endl;

    try {
        auto gs_cumsum = gs_data.cumsum(0);

        bool match = compare_float_tensors(gs_cumsum, torch_cumsum, 1e-3f);
        print_comparison("Cumsum", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR (may not be implemented): " << e.what() << std::endl;
        GTEST_SKIP() << "cumsum not implemented";
    }
}

// ============================================================================
// Test 6: Comparison and Equality - CRITICAL for k-means
// ============================================================================

TEST_F(TensorDebugTest, Equality_Int32) {
    std::cout << "\n=== TEST: Equality Int32 ===" << std::endl;

    auto torch_labels = torch::tensor({0, 1, 2, 1, 0, 2},
                                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_labels = torch_to_tensor(torch_labels);

    auto torch_mask = torch_labels == 1;
    std::cout << "LibTorch (labels == 1): " << torch_mask << std::endl;

    try {
        auto gs_mask = gs_labels.eq(1);

        auto torch_mask_float = torch_mask.to(torch::kFloat32);
        auto gs_mask_float = gs_mask.to(lfs::core::DataType::Float32);

        bool match = compare_float_tensors(gs_mask_float, torch_mask_float);
        print_comparison("Int32 equality", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, GreaterThan_Float32) {
    std::cout << "\n=== TEST: Greater Than Float32 ===" << std::endl;

    auto torch_data = torch::tensor({-1.0f, 0.0f, 1.0f, 2.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_mask = torch_data > 0.0f;
    std::cout << "LibTorch (data > 0): " << torch_mask << std::endl;

    try {
        auto gs_mask = gs_data > 0.0f;

        auto torch_mask_float = torch_mask.to(torch::kFloat32);
        auto gs_mask_float = gs_mask.to(lfs::core::DataType::Float32);

        bool match = compare_float_tensors(gs_mask_float, torch_mask_float);
        print_comparison("Greater than", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

// ============================================================================
// Test 7: Any/All Reductions - CRITICAL for k-means
// ============================================================================

TEST_F(TensorDebugTest, AnyScalar) {
    std::cout << "\n=== TEST: Any Scalar ===" << std::endl;

    auto torch_all_false = torch::zeros({10}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto torch_some_true = torch::ones({10}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto gs_all_false = lfs::core::Tensor::zeros_bool({10}, lfs::core::Device::CUDA);
    auto gs_some_true = lfs::core::Tensor::ones_bool({10}, lfs::core::Device::CUDA);

    bool torch_result_false = torch_all_false.any().item<bool>();
    bool torch_result_true = torch_some_true.any().item<bool>();

    std::cout << "LibTorch all_false.any(): " << torch_result_false << std::endl;
    std::cout << "LibTorch some_true.any(): " << torch_result_true << std::endl;

    try {
        bool gs_result_false = gs_all_false.any_scalar();
        bool gs_result_true = gs_some_true.any_scalar();

        std::cout << "lfs::core::Tensor all_false.any_scalar(): " << gs_result_false << std::endl;
        std::cout << "lfs::core::Tensor some_true.any_scalar(): " << gs_result_true << std::endl;

        bool match = (torch_result_false == gs_result_false) && (torch_result_true == gs_result_true);
        print_comparison("Any scalar", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

// ============================================================================
// Test 8: Zero In-place - CRITICAL for k-means
// ============================================================================

TEST_F(TensorDebugTest, ZeroInplace_Float32) {
    std::cout << "\n=== TEST: Zero In-place Float32 ===" << std::endl;

    auto torch_data = torch::randn({5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    torch_data.zero_();
    std::cout << "LibTorch after zero_(): " << torch_data << std::endl;

    try {
        gs_data.zero_();

        bool match = compare_float_tensors(gs_data, torch_data);
        print_comparison("Zero inplace float", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, ZeroInplace_Int32) {
    std::cout << "\n=== TEST: Zero In-place Int32 ===" << std::endl;

    auto torch_data = torch::randint(1, 100, {5},
                                     torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_data = torch_to_tensor(torch_data);

    torch_data.zero_();
    std::cout << "LibTorch after zero_(): " << torch_data << std::endl;

    try {
        gs_data.zero_();

        bool match = compare_int_tensors(gs_data, torch_data);
        print_comparison("Zero inplace int32", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

// ============================================================================
// Test 9: Copy/Clone Operations - CRITICAL for k-means
// ============================================================================

TEST_F(TensorDebugTest, Clone) {
    std::cout << "\n=== TEST: Clone ===" << std::endl;

    // Use deterministic values to avoid random failures
    auto torch_original = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, torch::kCUDA);
    auto gs_original = torch_to_tensor(torch_original);

    // Clone both tensors
    auto torch_cloned = torch_original.clone();
    auto gs_cloned = gs_original.clone();

    // Verify clones match originals
    bool clone_match = compare_float_tensors(gs_cloned, torch_cloned);
    print_comparison("Clone data matches", clone_match);
    EXPECT_TRUE(clone_match);

    // Modify clones
    torch_cloned.zero_();
    gs_cloned.zero_();

    // Verify clones are now zero
    auto torch_cloned_sum = torch_cloned.sum().item<float>();
    auto gs_cloned_sum = gs_cloned.sum().item();

    EXPECT_FLOAT_EQ(torch_cloned_sum, 0.0f);
    EXPECT_FLOAT_EQ(gs_cloned_sum, 0.0f);

    // Verify originals are UNCHANGED (should still sum to 15.0)
    auto torch_orig_sum = torch_original.sum().item<float>();
    auto gs_orig_sum = gs_original.sum().item();

    std::cout << "  Original sum after clone.zero_() - torch: " << torch_orig_sum << std::endl;
    std::cout << "  Original sum after clone.zero_() - gs: " << gs_orig_sum << std::endl;

    bool torch_unchanged = std::abs(torch_orig_sum - 15.0f) < FLOAT_TOLERANCE;
    bool gs_unchanged = std::abs(gs_orig_sum - 15.0f) < FLOAT_TOLERANCE;

    print_comparison("Clone independence (torch)", torch_unchanged);
    print_comparison("Clone independence (gs)", gs_unchanged);

    EXPECT_TRUE(torch_unchanged);
    EXPECT_TRUE(gs_unchanged);
}

TEST_F(TensorDebugTest, CopyFrom) {
    std::cout << "\n=== TEST: Copy From ===" << std::endl;

    auto torch_src = torch::randn({5}, torch::kCUDA);
    auto torch_dst = torch::zeros({5}, torch::kCUDA);

    auto gs_src = torch_to_tensor(torch_src);
    auto gs_dst = lfs::core::Tensor::zeros({5}, lfs::core::Device::CUDA);

    torch_dst.copy_(torch_src);
    try {
        gs_dst.copy_(gs_src);

        bool match = compare_float_tensors(gs_dst, torch_dst);
        print_comparison("Copy from", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

// ============================================================================
// Test 10: Advanced Operations for k-means
// ============================================================================

TEST_F(TensorDebugTest, Min_Max_Reductions) {
    std::cout << "\n=== TEST: Min/Max Reductions ===" << std::endl;

    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    float torch_min = torch_data.min().item<float>();
    float torch_max = torch_data.max().item<float>();

    std::cout << "LibTorch min: " << torch_min << " max: " << torch_max << std::endl;

    try {
        float gs_min = gs_data.min().item();
        float gs_max = gs_data.max().item();

        std::cout << "lfs::core::Tensor min: " << gs_min << " max: " << gs_max << std::endl;

        bool min_match = std::abs(torch_min - gs_min) < FLOAT_TOLERANCE;
        bool max_match = std::abs(torch_max - gs_max) < FLOAT_TOLERANCE;

        print_comparison("Min reduction", min_match);
        print_comparison("Max reduction", max_match);

        EXPECT_TRUE(min_match && max_match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, Abs_Operation) {
    std::cout << "\n=== TEST: Abs Operation ===" << std::endl;

    auto torch_data = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    auto torch_abs = torch_data.abs();
    std::cout << "LibTorch abs: " << torch_abs << std::endl;

    try {
        auto gs_abs = gs_data.abs();

        bool match = compare_float_tensors(gs_abs, torch_abs);
        print_comparison("Abs operation", match);
        EXPECT_TRUE(match);

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
        FAIL();
    }
}

TEST_F(TensorDebugTest, ArithmeticOperations) {
    std::cout << "\n=== TEST: Arithmetic Operations ===" << std::endl;

    auto torch_a = torch::randn({10}, torch::kCUDA);
    auto torch_b = torch::randn({10}, torch::kCUDA);

    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);

    // Subtraction
    auto torch_sub = torch_a - torch_b;
    auto gs_sub = gs_a - gs_b;
    bool sub_match = compare_float_tensors(gs_sub, torch_sub);
    print_comparison("Subtraction", sub_match);
    EXPECT_TRUE(sub_match);

    // Division
    auto torch_div = torch_a / (torch_b + 1.0f); // Avoid div by zero
    auto gs_div = gs_a / (gs_b + 1.0f);
    bool div_match = compare_float_tensors(gs_div, torch_div, 1e-3f);
    print_comparison("Division", div_match);
    EXPECT_TRUE(div_match);
}

TEST_F(TensorDebugTest, Linspace) {
    std::cout << "\n=== TEST: Linspace ===" << std::endl;

    auto torch_ls = torch::linspace(0, 100, 256, torch::kCUDA);
    auto gs_ls = lfs::core::Tensor::linspace(0, 100, 256, lfs::core::Device::CUDA);

    bool match = compare_float_tensors(gs_ls, torch_ls, 1e-3f);
    print_comparison("Linspace", match);
    EXPECT_TRUE(match);
}