/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper to create PyTorch tensor from vector data
    torch::Tensor create_torch_bool_tensor(const std::vector<bool>& data,
                                           const std::vector<int64_t>& shape,
                                           torch::Device device = torch::kCUDA) {
        // Convert bool to uint8 for PyTorch
        std::vector<uint8_t> uint8_data(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            uint8_data[i] = data[i] ? 1 : 0;
        }

        // Create on CPU first
        auto cpu_tensor = torch::from_blob(
                              uint8_data.data(),
                              shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(data.size())} : shape,
                              torch::TensorOptions().dtype(torch::kBool))
                              .clone(); // Clone to own the memory

        // Now move to target device
        return cpu_tensor.to(device);
    }

    void compare_bool_tensors(const Tensor& custom, const torch::Tensor& reference,
                              const std::string& msg = "") {
        // Flatten and move to CPU for comparison
        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector_bool();

        // Use flattened accessor for bool tensors
        auto ref_accessor = ref_cpu.accessor<bool, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            bool ref_val = ref_accessor[i];
            bool custom_val = custom_vec[i];

            EXPECT_EQ(custom_val, ref_val)
                << msg << ": Mismatch at index " << i
                << " (custom=" << custom_val << ", ref=" << ref_val << ")";
        }
    }

} // anonymous namespace

class TensorBitwiseTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// ============= Bitwise NOT (~) Tests =============

TEST_F(TensorBitwiseTest, BitwiseNotBasic) {
    std::vector<bool> data = {true, false, true, false};

    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {4});

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.dtype(), DataType::Bool);

    compare_bool_tensors(result_custom, result_torch, "BitwiseNotBasic");
}

TEST_F(TensorBitwiseTest, BitwiseNotCPU) {
    std::vector<bool> data = {true, false, true, false};

    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CPU);
    auto tensor_torch = create_torch_bool_tensor(data, {4}, torch::kCPU);

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.device(), Device::CPU);

    compare_bool_tensors(result_custom, result_torch, "BitwiseNotCPU");
}

TEST_F(TensorBitwiseTest, BitwiseNotCUDA) {
    std::vector<bool> data = {true, false, true, false};

    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {4}, torch::kCUDA);

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.device(), Device::CUDA);

    compare_bool_tensors(result_custom, result_torch, "BitwiseNotCUDA");
}

TEST_F(TensorBitwiseTest, BitwiseNotMultiDimensional) {
    std::vector<bool> data = {true, false, false, true, true, false};

    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {2, 3});

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.shape(), TensorShape({2, 3}));

    compare_bool_tensors(result_custom, result_torch, "BitwiseNotMultiDim");
}

TEST_F(TensorBitwiseTest, BitwiseNotAllTrue) {
    std::vector<bool> data = {true, true, true, true, true};

    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {5});

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    compare_bool_tensors(result_custom, result_torch, "BitwiseNotAllTrue");
}

TEST_F(TensorBitwiseTest, BitwiseNotAllFalse) {
    std::vector<bool> data = {false, false, false, false};

    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {4});

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    compare_bool_tensors(result_custom, result_torch, "BitwiseNotAllFalse");
}

TEST_F(TensorBitwiseTest, BitwiseNotOnNonBoolFails) {
    auto t = Tensor::ones({4}, Device::CPU, DataType::Float32);
    auto result = ~t;

    // Should fail gracefully
    EXPECT_FALSE(result.is_valid());
}

// ============= Bitwise OR (|) Tests =============

TEST_F(TensorBitwiseTest, BitwiseOrBasic) {
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {4});
    auto b_torch = create_torch_bool_tensor(data_b, {4});

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.dtype(), DataType::Bool);

    compare_bool_tensors(result_custom, result_torch, "BitwiseOrBasic");
}

TEST_F(TensorBitwiseTest, BitwiseOrCPU) {
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CPU);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CPU);
    auto a_torch = create_torch_bool_tensor(data_a, {4}, torch::kCPU);
    auto b_torch = create_torch_bool_tensor(data_b, {4}, torch::kCPU);

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    ASSERT_TRUE(result_custom.is_valid());
    compare_bool_tensors(result_custom, result_torch, "BitwiseOrCPU");
}

TEST_F(TensorBitwiseTest, BitwiseOrCUDA) {
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {4});
    auto b_torch = create_torch_bool_tensor(data_b, {4});

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    ASSERT_TRUE(result_custom.is_valid());
    compare_bool_tensors(result_custom, result_torch, "BitwiseOrCUDA");
}

TEST_F(TensorBitwiseTest, BitwiseOrMultiDimensional) {
    std::vector<bool> data_a = {true, false, false, true, true, false};
    std::vector<bool> data_b = {false, false, true, true, false, true};

    auto a_custom = Tensor::from_vector(data_a, {2, 3}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {2, 3}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {2, 3});
    auto b_torch = create_torch_bool_tensor(data_b, {2, 3});

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    ASSERT_TRUE(result_custom.is_valid());
    compare_bool_tensors(result_custom, result_torch, "BitwiseOrMultiDim");
}

TEST_F(TensorBitwiseTest, BitwiseOrBroadcast) {
    std::vector<bool> data_a = {true, false};
    std::vector<bool> data_b = {true, false, true};

    auto a_custom = Tensor::from_vector(data_a, {2, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {3}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {2, 1});
    auto b_torch = create_torch_bool_tensor(data_b, {3});

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.shape(), TensorShape({2, 3}));

    compare_bool_tensors(result_custom, result_torch, "BitwiseOrBroadcast");
}

TEST_F(TensorBitwiseTest, BitwiseOrAllCombinations) {
    // Test all 4 combinations of (true/false, true/false)
    std::vector<bool> data_a = {true, true, false, false};
    std::vector<bool> data_b = {true, false, true, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {4});
    auto b_torch = create_torch_bool_tensor(data_b, {4});

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    compare_bool_tensors(result_custom, result_torch, "BitwiseOrAllCombos");

    // Verify expected truth table
    auto values = result_custom.to_vector_bool();
    EXPECT_TRUE(values[0]);  // true | true = true
    EXPECT_TRUE(values[1]);  // true | false = true
    EXPECT_TRUE(values[2]);  // false | true = true
    EXPECT_FALSE(values[3]); // false | false = false
}

TEST_F(TensorBitwiseTest, BitwiseOrOnNonBoolFails) {
    auto a = Tensor::ones({4}, Device::CPU);
    auto b = Tensor::zeros({4}, Device::CPU);
    auto result = a | b;

    EXPECT_FALSE(result.is_valid());
}

// ============= Logical vs Bitwise Operations =============

TEST_F(TensorBitwiseTest, LogicalVsBitwiseOr) {
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {4});
    auto b_torch = create_torch_bool_tensor(data_b, {4});

    // Bitwise OR
    auto bitwise_custom = a_custom | b_custom;
    auto bitwise_torch = a_torch | b_torch;
    compare_bool_tensors(bitwise_custom, bitwise_torch, "BitwiseOr");

    // Logical OR (should be identical for bool tensors)
    auto logical_custom = a_custom || b_custom;
    auto logical_torch = a_torch.logical_or(b_torch);
    compare_bool_tensors(logical_custom, logical_torch, "LogicalOr");

    // They should match for bool tensors
    auto bitwise_vals = bitwise_custom.to_vector_bool();
    auto logical_vals = logical_custom.to_vector_bool();
    EXPECT_EQ(bitwise_vals, logical_vals);
}

TEST_F(TensorBitwiseTest, LogicalVsBitwiseAnd) {
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {4});
    auto b_torch = create_torch_bool_tensor(data_b, {4});

    // Logical AND
    auto logical_custom = a_custom && b_custom;
    auto logical_torch = a_torch.logical_and(b_torch);
    compare_bool_tensors(logical_custom, logical_torch, "LogicalAnd");

    // Verify expected truth table
    auto values = logical_custom.to_vector_bool();
    EXPECT_TRUE(values[0]);  // true && true = true
    EXPECT_FALSE(values[1]); // false && true = false
    EXPECT_FALSE(values[2]); // true && false = false
    EXPECT_FALSE(values[3]); // false && false = false
}

TEST_F(TensorBitwiseTest, LogicalNot) {
    std::vector<bool> data = {true, false, true, false};

    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {4});

    // Using ! operator (logical not)
    auto result_custom = !tensor_custom;
    auto result_torch = tensor_torch.logical_not();

    compare_bool_tensors(result_custom, result_torch, "LogicalNot");

    // Should match bitwise NOT for bool tensors
    auto bitwise_result = ~tensor_custom;
    auto logical_vals = result_custom.to_vector_bool();
    auto bitwise_vals = bitwise_result.to_vector_bool();
    EXPECT_EQ(logical_vals, bitwise_vals);
}

// ============= Combined Operations =============

TEST_F(TensorBitwiseTest, DeMorgansLaw) {
    // Test: ~(a | b) == (~a) && (~b)  (De Morgan's law)
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {4});
    auto b_torch = create_torch_bool_tensor(data_b, {4});

    // Left side: ~(a | b)
    auto left_custom = ~(a_custom | b_custom);
    auto left_torch = ~(a_torch | b_torch);
    compare_bool_tensors(left_custom, left_torch, "DeMorgan_Left");

    // Right side: (~a) && (~b)
    auto right_custom = (~a_custom) && (~b_custom);
    auto right_torch = (~a_torch).logical_and(~b_torch);
    compare_bool_tensors(right_custom, right_torch, "DeMorgan_Right");

    // Both sides should be equal
    auto left_vals = left_custom.to_vector_bool();
    auto right_vals = right_custom.to_vector_bool();
    EXPECT_EQ(left_vals, right_vals);
}

TEST_F(TensorBitwiseTest, DoubleNegation) {
    std::vector<bool> data = {true, false, true, false, false};

    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {5});

    // ~~a should equal a
    auto double_neg_custom = ~~tensor_custom;
    auto double_neg_torch = ~~tensor_torch;

    compare_bool_tensors(double_neg_custom, double_neg_torch, "DoubleNegation");
    compare_bool_tensors(double_neg_custom, tensor_torch, "DoubleNegationOriginal");
}

TEST_F(TensorBitwiseTest, ComplexExpression) {
    // Test: (a | b) & ~c
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {false, true, true, false};
    std::vector<bool> data_c = {true, true, false, false};

    auto a_custom = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto c_custom = Tensor::from_vector(data_c, {4}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {4});
    auto b_torch = create_torch_bool_tensor(data_b, {4});
    auto c_torch = create_torch_bool_tensor(data_c, {4});

    auto result_custom = (a_custom | b_custom) && (~c_custom);
    auto result_torch = (a_torch | b_torch).logical_and(~c_torch);

    compare_bool_tensors(result_custom, result_torch, "ComplexExpression");
}

// ============= Shape and Device Tests =============

TEST_F(TensorBitwiseTest, BitwiseNotPreservesShape) {
    std::vector<bool> data = {true, false, true, false, false, true, true, false};

    auto tensor_custom = Tensor::from_vector(data, {2, 2, 2}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {2, 2, 2});

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    EXPECT_EQ(result_custom.shape(), tensor_custom.shape());
    EXPECT_EQ(result_custom.numel(), tensor_custom.numel());

    compare_bool_tensors(result_custom, result_torch, "PreservesShape");
}

TEST_F(TensorBitwiseTest, BitwiseOrPreservesDevice) {
    std::vector<bool> data_a = {true, false};
    std::vector<bool> data_b = {false, true};

    // CPU test
    auto cpu_a = Tensor::from_vector(data_a, {2}, Device::CPU);
    auto cpu_b = Tensor::from_vector(data_b, {2}, Device::CPU);
    auto result_cpu = cpu_a | cpu_b;
    EXPECT_EQ(result_cpu.device(), Device::CPU);

    auto torch_a_cpu = create_torch_bool_tensor(data_a, {2}, torch::kCPU);
    auto torch_b_cpu = create_torch_bool_tensor(data_b, {2}, torch::kCPU);
    auto result_torch_cpu = torch_a_cpu | torch_b_cpu;
    compare_bool_tensors(result_cpu, result_torch_cpu, "PreservesDevice_CPU");

    // CUDA test
    auto cuda_a = Tensor::from_vector(data_a, {2}, Device::CUDA);
    auto cuda_b = Tensor::from_vector(data_b, {2}, Device::CUDA);
    auto result_cuda = cuda_a | cuda_b;
    EXPECT_EQ(result_cuda.device(), Device::CUDA);

    auto torch_a_cuda = create_torch_bool_tensor(data_a, {2});
    auto torch_b_cuda = create_torch_bool_tensor(data_b, {2});
    auto result_torch_cuda = torch_a_cuda | torch_b_cuda;
    compare_bool_tensors(result_cuda, result_torch_cuda, "PreservesDevice_CUDA");
}

// ============= Edge Cases =============

TEST_F(TensorBitwiseTest, BitwiseNotEmptyTensor) {
    std::vector<bool> empty_data;

    auto tensor_custom = Tensor::from_vector(empty_data, {0}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(empty_data, {0});

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    EXPECT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.numel(), 0);

    compare_bool_tensors(result_custom, result_torch, "BitwiseNotEmpty");
}

TEST_F(TensorBitwiseTest, BitwiseOrEmptyTensor) {
    std::vector<bool> empty_data;

    auto a_custom = Tensor::from_vector(empty_data, {0}, Device::CUDA);
    auto b_custom = Tensor::from_vector(empty_data, {0}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(empty_data, {0});
    auto b_torch = create_torch_bool_tensor(empty_data, {0});

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    EXPECT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.numel(), 0);

    compare_bool_tensors(result_custom, result_torch, "BitwiseOrEmpty");
}

TEST_F(TensorBitwiseTest, BitwiseNotLargeTensor) {
    std::vector<bool> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (i % 2 == 0);
    }

    auto tensor_custom = Tensor::from_vector(data, {10000}, Device::CUDA);
    auto tensor_torch = create_torch_bool_tensor(data, {10000});

    auto result_custom = ~tensor_custom;
    auto result_torch = ~tensor_torch;

    ASSERT_TRUE(result_custom.is_valid());
    compare_bool_tensors(result_custom, result_torch, "BitwiseNotLarge");
}

TEST_F(TensorBitwiseTest, BitwiseOrLargeTensor) {
    std::vector<bool> data_a(5000);
    std::vector<bool> data_b(5000);
    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = (i % 3 == 0);
        data_b[i] = (i % 5 == 0);
    }

    auto a_custom = Tensor::from_vector(data_a, {5000}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {5000}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {5000});
    auto b_torch = create_torch_bool_tensor(data_b, {5000});

    auto result_custom = a_custom | b_custom;
    auto result_torch = a_torch | b_torch;

    ASSERT_TRUE(result_custom.is_valid());
    compare_bool_tensors(result_custom, result_torch, "BitwiseOrLarge");
}

// ============= Random Data Tests =============

TEST_F(TensorBitwiseTest, RandomBitwiseOperations) {
    // Generate random bool data
    std::vector<bool> data_a(100);
    std::vector<bool> data_b(100);

    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = (i * 7 + 13) % 2 == 0;
        data_b[i] = (i * 11 + 5) % 2 == 0;
    }

    auto a_custom = Tensor::from_vector(data_a, {10, 10}, Device::CUDA);
    auto b_custom = Tensor::from_vector(data_b, {10, 10}, Device::CUDA);
    auto a_torch = create_torch_bool_tensor(data_a, {10, 10});
    auto b_torch = create_torch_bool_tensor(data_b, {10, 10});

    // Test NOT
    auto not_custom = ~a_custom;
    auto not_torch = ~a_torch;
    compare_bool_tensors(not_custom, not_torch, "RandomNot");

    // Test OR
    auto or_custom = a_custom | b_custom;
    auto or_torch = a_torch | b_torch;
    compare_bool_tensors(or_custom, or_torch, "RandomOr");

    // Test AND
    auto and_custom = a_custom && b_custom;
    auto and_torch = a_torch.logical_and(b_torch);
    compare_bool_tensors(and_custom, and_torch, "RandomAnd");

    // Test complex expression
    auto complex_custom = (~a_custom) | (a_custom && b_custom);
    auto complex_torch = (~a_torch) | (a_torch.logical_and(b_torch));
    compare_bool_tensors(complex_custom, complex_torch, "RandomComplex");
}