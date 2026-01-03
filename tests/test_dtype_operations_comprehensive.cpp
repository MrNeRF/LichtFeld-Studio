/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

using namespace lfs::core;

/**
 * Comprehensive Data Type Operations Tests
 *
 * This test suite verifies:
 * 1. Original comparison bug fixes (Bool mask operations)
 * 2. Type promotion across all dtype combinations
 * 3. Bool operations with all numeric types
 * 4. Float16 operations
 * 5. Int64 operations
 * 6. Mixed-dtype arithmetic and comparisons
 */

// =============================================================================
// SECTION 1: Original Comparison Bug Tests
// =============================================================================

TEST(ComparisonBugTests, BasicGreaterThan) {
    std::cout << "\n=== Test 1: Basic > comparison ===" << std::endl;

    auto t = Tensor::from_vector({0.1f, 0.5f, 0.9f}, {3}, Device::CUDA);
    std::cout << "Input: [0.1, 0.5, 0.9], threshold=0.5" << std::endl;

    auto mask = t > 0.5f;
    std::cout << "Mask dtype: " << static_cast<int>(mask.dtype()) << std::endl;
    std::cout << "Mask shape: [" << mask.shape()[0] << "]" << std::endl;

    EXPECT_EQ(mask.dtype(), DataType::Bool);
    EXPECT_EQ(mask.shape()[0], 3);

    auto mask_vec = mask.cpu().to_vector();
    std::cout << "Mask values: [" << mask_vec[0] << ", " << mask_vec[1] << ", " << mask_vec[2] << "]" << std::endl;

    EXPECT_FLOAT_EQ(mask_vec[0], 0.0f);
    EXPECT_FLOAT_EQ(mask_vec[1], 0.0f);
    EXPECT_FLOAT_EQ(mask_vec[2], 1.0f);
}

TEST(ComparisonBugTests, BooleanMaskMultiplication) {
    std::cout << "\n=== Test 3: Boolean mask multiplication ===" << std::endl;

    auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);
    std::cout << "Input: [1.0, 2.0, 3.0], threshold=1.5" << std::endl;

    auto mask = t > 1.5f;
    std::cout << "Mask dtype: " << static_cast<int>(mask.dtype()) << std::endl;

    auto mask_vec = mask.cpu().to_vector();
    std::cout << "Mask: [" << mask_vec[0] << ", " << mask_vec[1] << ", " << mask_vec[2] << "]" << std::endl;

    auto result = t * mask;
    auto result_vec = result.cpu().to_vector();
    std::cout << "Result: [" << result_vec[0] << ", " << result_vec[1] << ", " << result_vec[2] << "]" << std::endl;

    EXPECT_FLOAT_EQ(result_vec[0], 0.0f);
    EXPECT_FLOAT_EQ(result_vec[1], 2.0f);
    EXPECT_FLOAT_EQ(result_vec[2], 3.0f);
}

TEST(ComparisonBugTests, LargeTensorComparison) {
    std::cout << "\n=== Test 6: Large tensor comparison (1000 elements) ===" << std::endl;

    std::vector<float> vals(1000);
    for (int i = 0; i < 1000; i++) {
        vals[i] = i / 1000.0f;
    }

    std::cout << "Created tensor with 1000 values from 0 to 1" << std::endl;
    auto t = Tensor::from_vector(vals, {1000}, Device::CUDA);

    float threshold = 0.5f;
    std::cout << "Threshold: " << threshold << std::endl;

    auto mask = t > threshold;

    auto mask_sum = mask.cpu().to(DataType::Float32).sum();
    auto sum_vec = mask_sum.to_vector();
    int num_true = static_cast<int>(sum_vec[0]);

    std::cout << "Number of true values in mask: " << num_true << std::endl;
    std::cout << "Expected: ~500" << std::endl;

    EXPECT_NEAR(num_true, 500, 5);

    auto result = t * mask;
    auto result_sum = result.cpu().sum();
    auto result_sum_vec = result_sum.to_vector();
    int num_nonzero = 0;
    auto result_vec = result.cpu().to_vector();
    for (const auto& v : result_vec) {
        if (v > 0.0f)
            num_nonzero++;
    }

    std::cout << "Number of non-zero values in result: " << num_nonzero << std::endl;
    EXPECT_EQ(num_nonzero, num_true);
}

TEST(ComparisonBugTests, PruneZScenario) {
    std::cout << "\n=== Test 7: Exact prune_z scenario ===" << std::endl;

    size_t n = 100;
    float prune_ratio = 0.6f;
    int prune_count = static_cast<int>(n * prune_ratio);

    std::cout << "Simulating prune_z with n=" << n << ", prune_ratio=" << prune_ratio
              << ", index=" << prune_count << std::endl;

    std::vector<float> z_sorted(n);
    for (int i = 0; i < n; i++) {
        z_sorted[i] = i / 100.0f;
    }

    auto z_tensor = Tensor::from_vector(z_sorted, {n}, Device::CUDA);
    float threshold = z_sorted[prune_count - 1];

    std::cout << "Threshold (z_sorted[" << (prune_count - 1) << "]): " << threshold << std::endl;
    std::cout << "z_sorted[" << (prune_count - 2) << "]=" << z_sorted[prune_count - 2] << std::endl;
    std::cout << "z_sorted[" << (prune_count - 1) << "]=" << z_sorted[prune_count - 1] << std::endl;
    std::cout << "z_sorted[" << prune_count << "]=" << z_sorted[prune_count] << std::endl;

    auto prune_mask = z_tensor > threshold;

    auto mask_vec = prune_mask.cpu().to_vector();
    int true_count = 0;
    for (const auto& v : mask_vec) {
        if (v > 0.5f)
            true_count++;
    }

    std::cout << "Mask has " << true_count << " true values (expected " << (n - prune_count) << ")" << std::endl;
    EXPECT_EQ(true_count, n - prune_count);

    auto result = z_tensor * prune_mask;
    auto result_vec = result.cpu().to_vector();
    int nonzero_count = 0;
    for (const auto& v : result_vec) {
        if (v > 0.0f)
            nonzero_count++;
    }

    std::cout << "Result has " << nonzero_count << " non-zero values" << std::endl;
    EXPECT_EQ(nonzero_count, n - prune_count);
}

// =============================================================================
// SECTION 2: Bool Operations with All Numeric Types
// =============================================================================

TEST(BoolOperationsTests, BoolTimesAllTypes) {
    std::cout << "\n=== Bool × All Numeric Types ===" << std::endl;

    auto bool_tensor = Tensor::from_vector({0.0f, 1.0f}, {2}, Device::CUDA).to(DataType::Bool);

    // Bool × Float32
    {
        auto float32_tensor = Tensor::from_vector({10.0f, 20.0f}, {2}, Device::CUDA);
        auto result = bool_tensor * float32_tensor;
        auto vec = result.cpu().to_vector();

        std::cout << "Bool × Float32: [" << vec[0] << ", " << vec[1] << "]" << std::endl;
        EXPECT_FLOAT_EQ(vec[0], 0.0f);
        EXPECT_FLOAT_EQ(vec[1], 20.0f);
    }

    // Bool × Float16
    {
        auto float16_tensor = Tensor::from_vector({10.0f, 20.0f}, {2}, Device::CUDA).to(DataType::Float16);
        auto result = bool_tensor * float16_tensor;
        auto vec = result.cpu().to(DataType::Float32).to_vector();

        std::cout << "Bool × Float16: [" << vec[0] << ", " << vec[1] << "]" << std::endl;
        EXPECT_NEAR(vec[0], 0.0f, 0.01f);
        EXPECT_NEAR(vec[1], 20.0f, 0.01f);
    }

    // Bool × Int32
    {
        auto int32_tensor = Tensor::from_vector({10.0f, 20.0f}, {2}, Device::CUDA).to(DataType::Int32);
        auto result = bool_tensor * int32_tensor;
        auto vec = result.cpu().to(DataType::Float32).to_vector();

        std::cout << "Bool × Int32: [" << vec[0] << ", " << vec[1] << "]" << std::endl;
        EXPECT_FLOAT_EQ(vec[0], 0.0f);
        EXPECT_FLOAT_EQ(vec[1], 20.0f);
    }

    // Bool × Int64
    {
        auto int64_tensor = Tensor::from_vector({10.0f, 20.0f}, {2}, Device::CUDA).to(DataType::Int64);
        auto result = bool_tensor * int64_tensor;
        auto vec = result.cpu().to(DataType::Float32).to_vector();

        std::cout << "Bool × Int64: [" << vec[0] << ", " << vec[1] << "]" << std::endl;
        EXPECT_FLOAT_EQ(vec[0], 0.0f);
        EXPECT_FLOAT_EQ(vec[1], 20.0f);
    }

    std::cout << "✓ All Bool × Type operations work correctly" << std::endl;
    std::cout << "Note: UInt8 tested separately in SystematicAllCombinations" << std::endl;
}

TEST(BoolOperationsTests, BoolAllArithmeticOps) {
    std::cout << "\n=== Bool All Arithmetic Operations with Float32 ===" << std::endl;

    auto bool_tensor = Tensor::from_vector({0.0f, 1.0f, 1.0f}, {3}, Device::CUDA).to(DataType::Bool);
    auto float_tensor = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);

    // Addition
    auto add_result = bool_tensor + float_tensor;
    auto add_vec = add_result.cpu().to_vector();
    std::cout << "Bool + Float32: [" << add_vec[0] << ", " << add_vec[1] << ", " << add_vec[2] << "]" << std::endl;
    EXPECT_FLOAT_EQ(add_vec[0], 1.0f);
    EXPECT_FLOAT_EQ(add_vec[1], 3.0f);
    EXPECT_FLOAT_EQ(add_vec[2], 4.0f);

    // Subtraction
    auto sub_result = float_tensor - bool_tensor;
    auto sub_vec = sub_result.cpu().to_vector();
    std::cout << "Float32 - Bool: [" << sub_vec[0] << ", " << sub_vec[1] << ", " << sub_vec[2] << "]" << std::endl;
    EXPECT_FLOAT_EQ(sub_vec[0], 1.0f);
    EXPECT_FLOAT_EQ(sub_vec[1], 1.0f);
    EXPECT_FLOAT_EQ(sub_vec[2], 2.0f);

    // Multiplication
    auto mul_result = bool_tensor * float_tensor;
    auto mul_vec = mul_result.cpu().to_vector();
    std::cout << "Bool × Float32: [" << mul_vec[0] << ", " << mul_vec[1] << ", " << mul_vec[2] << "]" << std::endl;
    EXPECT_FLOAT_EQ(mul_vec[0], 0.0f);
    EXPECT_FLOAT_EQ(mul_vec[1], 2.0f);
    EXPECT_FLOAT_EQ(mul_vec[2], 3.0f);
}

// =============================================================================
// SECTION 3: Float16 Operations
// =============================================================================

TEST(Float16Tests, Float16BasicOperations) {
    std::cout << "\n=== Float16 Basic Operations ===" << std::endl;

    // Create Float16 tensors
    auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA).to(DataType::Float16);
    auto b = Tensor::from_vector({4.0f, 5.0f, 6.0f}, {3}, Device::CUDA).to(DataType::Float16);

    // Multiplication
    auto mul_result = a * b;
    EXPECT_EQ(mul_result.dtype(), DataType::Float16);
    auto mul_vec = mul_result.cpu().to(DataType::Float32).to_vector();
    std::cout << "Float16 × Float16: [" << mul_vec[0] << ", " << mul_vec[1] << ", " << mul_vec[2] << "]" << std::endl;
    EXPECT_NEAR(mul_vec[0], 4.0f, 0.01f);
    EXPECT_NEAR(mul_vec[1], 10.0f, 0.01f);
    EXPECT_NEAR(mul_vec[2], 18.0f, 0.01f);

    // Addition
    auto add_result = a + b;
    auto add_vec = add_result.cpu().to(DataType::Float32).to_vector();
    std::cout << "Float16 + Float16: [" << add_vec[0] << ", " << add_vec[1] << ", " << add_vec[2] << "]" << std::endl;
    EXPECT_NEAR(add_vec[0], 5.0f, 0.01f);
    EXPECT_NEAR(add_vec[1], 7.0f, 0.01f);
    EXPECT_NEAR(add_vec[2], 9.0f, 0.01f);
}

TEST(Float16Tests, Float16Conversions) {
    std::cout << "\n=== Float16 Type Conversions ===" << std::endl;

    auto float32_tensor = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);

    // Float32 → Float16
    auto float16_tensor = float32_tensor.to(DataType::Float16);
    EXPECT_EQ(float16_tensor.dtype(), DataType::Float16);

    // Float16 → Float32
    auto back_to_float32 = float16_tensor.to(DataType::Float32);
    auto vec = back_to_float32.cpu().to_vector();

    std::cout << "Round-trip Float32→Float16→Float32: [" << vec[0] << ", " << vec[1] << ", " << vec[2] << "]" << std::endl;
    EXPECT_NEAR(vec[0], 1.0f, 0.01f);
    EXPECT_NEAR(vec[1], 2.0f, 0.01f);
    EXPECT_NEAR(vec[2], 3.0f, 0.01f);
}

// =============================================================================
// SECTION 4: Int64 Operations
// =============================================================================

TEST(Int64Tests, Int64BasicOperations) {
    std::cout << "\n=== Int64 Basic Operations ===" << std::endl;

    auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA).to(DataType::Int64);
    auto b = Tensor::from_vector({4.0f, 5.0f, 6.0f}, {3}, Device::CUDA).to(DataType::Int64);

    // Multiplication
    auto mul_result = a * b;
    EXPECT_EQ(mul_result.dtype(), DataType::Int64);
    auto mul_vec = mul_result.cpu().to(DataType::Float32).to_vector();
    std::cout << "Int64 × Int64: [" << mul_vec[0] << ", " << mul_vec[1] << ", " << mul_vec[2] << "]" << std::endl;
    EXPECT_FLOAT_EQ(mul_vec[0], 4.0f);
    EXPECT_FLOAT_EQ(mul_vec[1], 10.0f);
    EXPECT_FLOAT_EQ(mul_vec[2], 18.0f);

    // Addition
    auto add_result = a + b;
    auto add_vec = add_result.cpu().to(DataType::Float32).to_vector();
    std::cout << "Int64 + Int64: [" << add_vec[0] << ", " << add_vec[1] << ", " << add_vec[2] << "]" << std::endl;
    EXPECT_FLOAT_EQ(add_vec[0], 5.0f);
    EXPECT_FLOAT_EQ(add_vec[1], 7.0f);
    EXPECT_FLOAT_EQ(add_vec[2], 9.0f);
}

TEST(Int64Tests, Int64Conversions) {
    std::cout << "\n=== Int64 Type Conversions ===" << std::endl;

    // Bool → Int64
    auto bool_tensor = Tensor::from_vector({0.0f, 1.0f, 1.0f}, {3}, Device::CUDA).to(DataType::Bool);
    auto int64_from_bool = bool_tensor.to(DataType::Int64);
    auto vec1 = int64_from_bool.cpu().to(DataType::Float32).to_vector();
    std::cout << "Bool→Int64: [" << vec1[0] << ", " << vec1[1] << ", " << vec1[2] << "]" << std::endl;
    EXPECT_FLOAT_EQ(vec1[0], 0.0f);
    EXPECT_FLOAT_EQ(vec1[1], 1.0f);
    EXPECT_FLOAT_EQ(vec1[2], 1.0f);

    // Int64 → Bool
    auto int64_tensor = Tensor::from_vector({0.0f, 1.0f, 2.0f}, {3}, Device::CUDA).to(DataType::Int64);
    auto bool_from_int64 = int64_tensor.to(DataType::Bool);
    auto vec2 = bool_from_int64.cpu().to_vector();
    std::cout << "Int64→Bool: [" << vec2[0] << ", " << vec2[1] << ", " << vec2[2] << "]" << std::endl;
    EXPECT_FLOAT_EQ(vec2[0], 0.0f);
    EXPECT_FLOAT_EQ(vec2[1], 1.0f);
    EXPECT_FLOAT_EQ(vec2[2], 1.0f); // Non-zero becomes 1
}

// =============================================================================
// SECTION 5: Type Promotion Tests
// =============================================================================

class TypePromotionTest : public ::testing::Test {
protected:
    Tensor create_tensor(DataType dtype) {
        std::vector<float> vals = {1.0f, 2.0f, 3.0f};
        auto t = Tensor::from_vector(vals, {3}, Device::CUDA);
        return t.to(dtype);
    }

    void verify_no_garbage(const Tensor& t, const std::string& op_name,
                           DataType lhs_type, DataType rhs_type) {
        auto cpu = t.cpu().to(DataType::Float32);
        auto vec = cpu.to_vector();

        for (size_t i = 0; i < vec.size(); i++) {
            EXPECT_FALSE(std::isnan(vec[i]))
                << op_name << " produced NaN for "
                << dtype_name(lhs_type) << " op " << dtype_name(rhs_type)
                << " at index " << i;

            if (vec[i] != 0.0f) {
                EXPECT_GT(std::abs(vec[i]), 1e-10f)
                    << op_name << " produced garbage value " << vec[i];
            }
        }
    }
};

TEST_F(TypePromotionTest, IntegerTypePromotions) {
    std::cout << "\n=== Integer Type Promotions ===" << std::endl;

    auto int32_t = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA).to(DataType::Int32);
    auto int64_t = Tensor::from_vector({4.0f, 5.0f, 6.0f}, {3}, Device::CUDA).to(DataType::Int64);
    auto uint8_t = Tensor::from_vector({1.0f, 1.0f, 1.0f}, {3}, Device::CUDA).to(DataType::UInt8);

    // Int32 + Int64 → Int64
    auto result1 = int32_t + int64_t;
    EXPECT_EQ(result1.dtype(), DataType::Int64);
    verify_no_garbage(result1, "Int32 + Int64", DataType::Int32, DataType::Int64);
    std::cout << "✓ Int32 + Int64 promotes to Int64" << std::endl;

    // UInt8 + Int32 → Int32
    auto result2 = uint8_t + int32_t;
    EXPECT_EQ(result2.dtype(), DataType::Int32);
    verify_no_garbage(result2, "UInt8 + Int32", DataType::UInt8, DataType::Int32);
    std::cout << "✓ UInt8 + Int32 promotes to Int32" << std::endl;

    // UInt8 + Int64 → Int64
    auto result3 = uint8_t + int64_t;
    EXPECT_EQ(result3.dtype(), DataType::Int64);
    verify_no_garbage(result3, "UInt8 + Int64", DataType::UInt8, DataType::Int64);
    std::cout << "✓ UInt8 + Int64 promotes to Int64" << std::endl;
}

TEST_F(TypePromotionTest, IntegerFloatPromotions) {
    std::cout << "\n=== Integer + Float Promotions ===" << std::endl;

    auto int32_t = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA).to(DataType::Int32);
    auto float32_t = Tensor::from_vector({1.5f, 2.5f, 3.5f}, {3}, Device::CUDA);
    auto float16_t = float32_t.to(DataType::Float16);

    // Int32 + Float32 → Float32
    auto result1 = int32_t + float32_t;
    EXPECT_EQ(result1.dtype(), DataType::Float32);
    verify_no_garbage(result1, "Int32 + Float32", DataType::Int32, DataType::Float32);
    std::cout << "✓ Int32 + Float32 promotes to Float32" << std::endl;

    // Int32 + Float16 → Float16
    auto result2 = int32_t + float16_t;
    EXPECT_EQ(result2.dtype(), DataType::Float16);
    verify_no_garbage(result2, "Int32 + Float16", DataType::Int32, DataType::Float16);
    std::cout << "✓ Int32 + Float16 promotes to Float16" << std::endl;

    // Float16 + Float32 → Float32
    auto result3 = float16_t + float32_t;
    EXPECT_EQ(result3.dtype(), DataType::Float32);
    verify_no_garbage(result3, "Float16 + Float32", DataType::Float16, DataType::Float32);
    std::cout << "✓ Float16 + Float32 promotes to Float32" << std::endl;
}

TEST_F(TypePromotionTest, ComparisonsMixedTypes) {
    std::cout << "\n=== Comparisons with Mixed Types ===" << std::endl;

    auto bool_tensor = Tensor::from_vector({0.0f, 1.0f, 1.0f}, {3}, Device::CUDA).to(DataType::Bool);
    auto int32_t = Tensor::from_vector({0.0f, 1.0f, 2.0f}, {3}, Device::CUDA).to(DataType::Int32);
    auto float32_t = Tensor::from_vector({0.5f, 1.0f, 1.5f}, {3}, Device::CUDA);

    // Bool == Int32
    auto cmp1 = bool_tensor == int32_t;
    EXPECT_EQ(cmp1.dtype(), DataType::Bool);
    auto cmp1_vec = cmp1.cpu().to_vector();
    EXPECT_FLOAT_EQ(cmp1_vec[0], 1.0f); // 0 == 0
    EXPECT_FLOAT_EQ(cmp1_vec[1], 1.0f); // 1 == 1
    EXPECT_FLOAT_EQ(cmp1_vec[2], 0.0f); // 1 != 2
    std::cout << "✓ Bool == Int32 works correctly" << std::endl;

    // Int32 > Float32
    auto cmp2 = int32_t > float32_t;
    EXPECT_EQ(cmp2.dtype(), DataType::Bool);
    verify_no_garbage(cmp2, "Int32 > Float32", DataType::Int32, DataType::Float32);
    std::cout << "✓ Int32 > Float32 works correctly" << std::endl;
}

TEST_F(TypePromotionTest, SystematicAllCombinations) {
    std::cout << "\n=== Testing All 36 Dtype Combinations ===" << std::endl;

    std::vector<DataType> all_types = {
        DataType::Bool, DataType::UInt8, DataType::Int32,
        DataType::Int64, DataType::Float16, DataType::Float32};

    int test_count = 0;
    int pass_count = 0;

    for (auto lhs_type : all_types) {
        for (auto rhs_type : all_types) {
            auto lhs = Tensor::from_vector({1.0f, 2.0f}, {2}, Device::CUDA).to(lhs_type);
            auto rhs = Tensor::from_vector({1.0f, 1.0f}, {2}, Device::CUDA).to(rhs_type);

            // Test multiplication
            try {
                auto result = lhs * rhs;
                verify_no_garbage(result,
                                  std::string(dtype_name(lhs_type)) + " * " + std::string(dtype_name(rhs_type)),
                                  lhs_type, rhs_type);
                pass_count++;
            } catch (...) {
                FAIL() << "Failed: " << dtype_name(lhs_type) << " * " << dtype_name(rhs_type);
            }
            test_count++;

            // Test addition
            try {
                auto result = lhs + rhs;
                verify_no_garbage(result,
                                  std::string(dtype_name(lhs_type)) + " + " + std::string(dtype_name(rhs_type)),
                                  lhs_type, rhs_type);
                pass_count++;
            } catch (...) {
                FAIL() << "Failed: " << dtype_name(lhs_type) << " + " << dtype_name(rhs_type);
            }
            test_count++;
        }
    }

    std::cout << "Tested " << test_count << " dtype combinations, "
              << pass_count << " passed" << std::endl;
    EXPECT_EQ(pass_count, test_count);
}

TEST_F(TypePromotionTest, BroadcastingMixedTypes) {
    std::cout << "\n=== Broadcasting with Mixed Types ===" << std::endl;

    auto bool_scalar = Tensor::from_vector({1.0f}, {1}, Device::CUDA).to(DataType::Bool);
    auto float_vector = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);

    auto result = bool_scalar * float_vector;
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.dtype(), DataType::Float32);
    verify_no_garbage(result, "Bool[1] * Float32[3]", DataType::Bool, DataType::Float32);

    auto result_vec = result.cpu().to_vector();
    EXPECT_FLOAT_EQ(result_vec[0], 1.0f);
    EXPECT_FLOAT_EQ(result_vec[1], 2.0f);
    EXPECT_FLOAT_EQ(result_vec[2], 3.0f);
    std::cout << "✓ Broadcasting with type promotion works" << std::endl;
}

TEST_F(TypePromotionTest, TwoDimensionalMixedTypes) {
    std::cout << "\n=== 2D Tensors with Mixed Types ===" << std::endl;

    auto bool_2d = Tensor::from_vector({0.0f, 1.0f, 1.0f, 0.0f}, {2, 2}, Device::CUDA).to(DataType::Bool);
    auto float_2d = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, Device::CUDA);

    auto result = bool_2d * float_2d;
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
    EXPECT_EQ(result.dtype(), DataType::Float32);
    verify_no_garbage(result, "Bool[2,2] * Float32[2,2]", DataType::Bool, DataType::Float32);

    auto result_vec = result.cpu().to_vector();
    EXPECT_FLOAT_EQ(result_vec[0], 0.0f);
    EXPECT_FLOAT_EQ(result_vec[1], 2.0f);
    EXPECT_FLOAT_EQ(result_vec[2], 3.0f);
    EXPECT_FLOAT_EQ(result_vec[3], 0.0f);
    std::cout << "✓ 2D mixed-dtype operations work" << std::endl;
}
