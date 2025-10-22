/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper for comparing boolean tensors
    void compare_bool_tensors(const Tensor& custom, const torch::Tensor& reference,
                              const std::string& msg = "") {
        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        // Get boolean values
        auto custom_vec = custom_cpu.to_vector_bool();
        auto ref_accessor = ref_cpu.accessor<bool, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            EXPECT_EQ(custom_vec[i], ref_accessor[i])
                << msg << ": Mismatch at index " << i
                << " (custom=" << custom_vec[i] << ", ref=" << ref_accessor[i] << ")";
        }
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
        // Handle boolean tensors specially
        if (reference.dtype() == torch::kBool) {
            compare_bool_tensors(custom, reference, msg);
            return;
        }

        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector();
        auto ref_accessor = ref_cpu.accessor<float, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            float ref_val = ref_accessor[i];
            float custom_val = custom_vec[i];

            if (std::isnan(ref_val)) {
                EXPECT_TRUE(std::isnan(custom_val)) << msg << ": Expected NaN at index " << i;
            } else if (std::isinf(ref_val)) {
                EXPECT_TRUE(std::isinf(custom_val)) << msg << ": Expected Inf at index " << i;
            } else {
                float diff = std::abs(custom_val - ref_val);
                float threshold = atol + rtol * std::abs(ref_val);
                EXPECT_LE(diff, threshold)
                    << msg << ": Mismatch at index " << i
                    << " (custom=" << custom_val << ", ref=" << ref_val << ")";
            }
        }
    }

} // anonymous namespace

class TensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
        gen.seed(42);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// ============= Scalar Operations =============

TEST_F(TensorOpsTest, ScalarAdd) {
    std::vector<float> data(12);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({3, 4});

    float scalar = 2.5f;

    auto custom_result = custom_tensor.add(scalar);
    auto torch_result = torch_tensor + scalar;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ScalarAdd");

    // Test operator overload
    auto custom_result2 = custom_tensor + scalar;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "ScalarAdd_Operator");
}

TEST_F(TensorOpsTest, ScalarSubtract) {
    std::vector<float> data(10);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {2, 5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 5});

    float scalar = 1.5f;

    auto custom_result = custom_tensor.sub(scalar);
    auto torch_result = torch_tensor - scalar;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ScalarSub");

    // Test operator overload
    auto custom_result2 = custom_tensor - scalar;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "ScalarSub_Operator");
}

TEST_F(TensorOpsTest, ScalarMultiply) {
    std::vector<float> data(12);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {4, 3}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({4, 3});

    float scalar = 3.0f;

    auto custom_result = custom_tensor.mul(scalar);
    auto torch_result = torch_tensor * scalar;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ScalarMul");

    // Test operator overload
    auto custom_result2 = custom_tensor * scalar;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "ScalarMul_Operator");
}

TEST_F(TensorOpsTest, ScalarDivide) {
    std::vector<float> data(9);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {3, 3}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({3, 3});

    float scalar = 2.0f;

    auto custom_result = custom_tensor.div(scalar);
    auto torch_result = torch_tensor / scalar;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ScalarDiv");

    // Test operator overload
    auto custom_result2 = custom_tensor / scalar;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "ScalarDiv_Operator");
}

TEST_F(TensorOpsTest, ScalarPower) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_tensor = Tensor::from_vector(data, {2, 2}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 2});

    float exponent = 2.0f;

    auto custom_result = custom_tensor.pow(exponent);
    auto torch_result = torch::pow(torch_tensor, exponent);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ScalarPow");
}

TEST_F(TensorOpsTest, ScalarModulo) {
    std::vector<float> data = {10.5f, 7.3f, 15.8f, 22.1f};

    auto custom_tensor = Tensor::from_vector(data, {2, 2}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 2});

    float modulus = 3.0f;

    auto custom_result = custom_tensor.mod(modulus);
    auto torch_result = torch::fmod(torch_tensor, modulus);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ScalarMod");
}

// ============= Unary Operations =============

TEST_F(TensorOpsTest, Negation) {
    std::vector<float> data(8);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {2, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 4});

    auto custom_result = custom_tensor.neg();
    auto torch_result = -torch_tensor;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Neg");

    // Test operator overload
    auto custom_result2 = -custom_tensor;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "Neg_Operator");
}

TEST_F(TensorOpsTest, Abs) {
    std::vector<float> data = {-5.0f, 3.0f, -2.0f, 7.0f, -1.0f, 4.0f};

    auto custom_tensor = Tensor::from_vector(data, {6}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.abs();
    auto torch_result = torch::abs(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Abs");
}

TEST_F(TensorOpsTest, Sqrt) {
    std::vector<float> data = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.sqrt();
    auto torch_result = torch::sqrt(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Sqrt");
}

TEST_F(TensorOpsTest, Square) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.square();
    auto torch_result = torch::square(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Square");
}

TEST_F(TensorOpsTest, Exp) {
    std::vector<float> data = {0.0f, 1.0f, 2.0f, -1.0f, -2.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.exp();
    auto torch_result = torch::exp(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "Exp");
}

TEST_F(TensorOpsTest, Log) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 10.0f, 100.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.log();
    auto torch_result = torch::log(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Log");
}

TEST_F(TensorOpsTest, Sigmoid) {
    std::vector<float> data(10);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {10}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.sigmoid();
    auto torch_result = torch::sigmoid(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Sigmoid");
}

TEST_F(TensorOpsTest, Relu) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 1.0f, 3.0f, 10.0f};

    auto custom_tensor = Tensor::from_vector(data, {6}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.relu();
    auto torch_result = torch::relu(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Relu");
}

TEST_F(TensorOpsTest, Tanh) {
    std::vector<float> data(8);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {8}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.tanh();
    auto torch_result = torch::tanh(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Tanh");
}

TEST_F(TensorOpsTest, Sin) {
    std::vector<float> data = {0.0f, M_PI / 6, M_PI / 4, M_PI / 3, M_PI / 2};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.sin();
    auto torch_result = torch::sin(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Sin");
}

TEST_F(TensorOpsTest, Cos) {
    std::vector<float> data = {0.0f, M_PI / 6, M_PI / 4, M_PI / 3, M_PI / 2};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.cos();
    auto torch_result = torch::cos(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Cos");
}

TEST_F(TensorOpsTest, Floor) {
    std::vector<float> data = {1.2f, 2.7f, -1.3f, -2.8f, 0.5f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.floor();
    auto torch_result = torch::floor(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Floor");
}

TEST_F(TensorOpsTest, Ceil) {
    std::vector<float> data = {1.2f, 2.7f, -1.3f, -2.8f, 0.5f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.ceil();
    auto torch_result = torch::ceil(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Ceil");
}

TEST_F(TensorOpsTest, Round) {
    // Avoid half-way values (x.5) as different implementations use different tie-breaking rules
    // PyTorch uses "round half to even", while many implementations use "round half away from zero"
    std::vector<float> data = {1.2f, 1.7f, 2.3f, 2.8f, -1.3f, -1.7f, -2.2f, -2.9f};

    auto custom_tensor = Tensor::from_vector(data, {8}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.round();
    auto torch_result = torch::round(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Round");
}

TEST_F(TensorOpsTest, Sign) {
    std::vector<float> data = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_tensor.sign();
    auto torch_result = torch::sign(torch_tensor);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Sign");
}

// ============= Element-wise Binary Operations =============

TEST_F(TensorOpsTest, ElementWiseAdd) {
    std::vector<float> data_a(12), data_b(12);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});

    auto custom_result = custom_a.add(custom_b);
    auto torch_result = torch_a + torch_b;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ElementAdd");

    // Test operator overload
    auto custom_result2 = custom_a + custom_b;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "ElementAdd_Operator");
}

TEST_F(TensorOpsTest, ElementWiseSubtract) {
    std::vector<float> data_a(10), data_b(10);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {2, 5}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {2, 5}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 5});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 5});

    auto custom_result = custom_a.sub(custom_b);
    auto torch_result = torch_a - torch_b;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ElementSub");

    // Test operator overload
    auto custom_result2 = custom_a - custom_b;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "ElementSub_Operator");
}

TEST_F(TensorOpsTest, ElementWiseMultiply) {
    std::vector<float> data_a(12), data_b(12);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {4, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4, 3}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 3});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 3});

    auto custom_result = custom_a.mul(custom_b);
    auto torch_result = torch_a * torch_b;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ElementMul");

    // Test operator overload
    auto custom_result2 = custom_a * custom_b;
    compare_tensors(custom_result2, torch_result, 1e-5f, 1e-6f, "ElementMul_Operator");
}

TEST_F(TensorOpsTest, ElementWiseDivide) {
    std::vector<float> data_a(9), data_b(9);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen) + 2.0f; // Avoid division by zero

    auto custom_a = Tensor::from_vector(data_a, {3, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 3}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 3});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 3});

    auto custom_result = custom_a.div(custom_b);
    auto torch_result = torch_a / torch_b;

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ElementDiv");

    // Test operator overload
    auto custom_result2 = custom_a / custom_b;
    compare_tensors(custom_result2, torch_result, 1e-4f, 1e-5f, "ElementDiv_Operator");
}

TEST_F(TensorOpsTest, ElementWisePower) {
    std::vector<float> data_base = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> data_exp = {2.0f, 3.0f, 2.0f, 1.0f};

    auto custom_base = Tensor::from_vector(data_base, {2, 2}, Device::CUDA);
    auto custom_exp = Tensor::from_vector(data_exp, {2, 2}, Device::CUDA);

    auto torch_base = torch::tensor(data_base, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});
    auto torch_exp = torch::tensor(data_exp, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});

    auto custom_result = custom_base.pow(custom_exp);
    auto torch_result = torch::pow(torch_base, torch_exp);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ElementPow");
}

TEST_F(TensorOpsTest, Maximum) {
    std::vector<float> data_a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<float> data_b = {2.0f, 4.0f, 6.0f, 7.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.maximum(custom_b);
    auto torch_result = torch::maximum(torch_a, torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Maximum");
}

TEST_F(TensorOpsTest, Minimum) {
    std::vector<float> data_a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<float> data_b = {2.0f, 4.0f, 6.0f, 7.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.minimum(custom_b);
    auto torch_result = torch::minimum(torch_a, torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Minimum");
}

// ============= In-place Operations =============

TEST_F(TensorOpsTest, InPlaceScalarAdd) {
    std::vector<float> data(12);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({3, 4});

    float scalar = 2.5f;

    custom_tensor.add_(scalar);
    torch_tensor.add_(scalar);

    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlaceScalarAdd");
}

TEST_F(TensorOpsTest, InPlaceScalarMultiply) {
    std::vector<float> data(6);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3});

    float scalar = 3.0f;

    custom_tensor.mul_(scalar);
    torch_tensor.mul_(scalar);

    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlaceScalarMul");
}

TEST_F(TensorOpsTest, InPlaceElementWiseAdd) {
    std::vector<float> data_a(9), data_b(9);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {3, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 3}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 3});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 3});

    custom_a.add_(custom_b);
    torch_a.add_(torch_b);

    compare_tensors(custom_a, torch_a, 1e-5f, 1e-6f, "InPlaceElementAdd");
}

TEST_F(TensorOpsTest, InPlaceElementWiseSubtract) {
    std::vector<float> data_a(8), data_b(8);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {2, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {2, 4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 4});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 4});

    custom_a.sub_(custom_b);
    torch_a.sub_(torch_b);

    compare_tensors(custom_a, torch_a, 1e-5f, 1e-6f, "InPlaceElementSub");
}

TEST_F(TensorOpsTest, InPlaceElementWiseMultiply) {
    std::vector<float> data_a(6), data_b(6);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {3, 2}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 2}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 2});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 2});

    custom_a.mul_(custom_b);
    torch_a.mul_(torch_b);

    compare_tensors(custom_a, torch_a, 1e-5f, 1e-6f, "InPlaceElementMul");
}

TEST_F(TensorOpsTest, InPlaceElementWiseDivide) {
    std::vector<float> data_a(4), data_b(4);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen) + 2.0f; // Avoid division by zero

    auto custom_a = Tensor::from_vector(data_a, {2, 2}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {2, 2}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});

    custom_a.div_(custom_b);
    torch_a.div_(torch_b);

    compare_tensors(custom_a, torch_a, 1e-4f, 1e-5f, "InPlaceElementDiv");
}

// ============= Comparison Operations =============

TEST_F(TensorOpsTest, Equal) {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {1.0f, 3.0f, 3.0f, 5.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.eq(custom_b);
    auto torch_result = torch_a.eq(torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Equal");

    // Test operator overload
    auto custom_result2 = custom_a == custom_b;
    compare_tensors(custom_result2, torch_result, 1e-6f, 1e-7f, "Equal_Operator");
}

TEST_F(TensorOpsTest, NotEqual) {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {1.0f, 3.0f, 3.0f, 5.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.ne(custom_b);
    auto torch_result = torch_a.ne(torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "NotEqual");

    // Test operator overload
    auto custom_result2 = custom_a != custom_b;
    compare_tensors(custom_result2, torch_result, 1e-6f, 1e-7f, "NotEqual_Operator");
}

TEST_F(TensorOpsTest, LessThan) {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 2.0f, 2.0f, 3.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.lt(custom_b);
    auto torch_result = torch_a.lt(torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "LessThan");

    // Test operator overload
    auto custom_result2 = custom_a < custom_b;
    compare_tensors(custom_result2, torch_result, 1e-6f, 1e-7f, "LessThan_Operator");
}

TEST_F(TensorOpsTest, LessEqual) {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 2.0f, 2.0f, 3.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.le(custom_b);
    auto torch_result = torch_a.le(torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "LessEqual");

    // Test operator overload
    auto custom_result2 = custom_a <= custom_b;
    compare_tensors(custom_result2, torch_result, 1e-6f, 1e-7f, "LessEqual_Operator");
}

TEST_F(TensorOpsTest, GreaterThan) {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 2.0f, 2.0f, 3.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.gt(custom_b);
    auto torch_result = torch_a.gt(torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "GreaterThan");

    // Test operator overload
    auto custom_result2 = custom_a > custom_b;
    compare_tensors(custom_result2, torch_result, 1e-6f, 1e-7f, "GreaterThan_Operator");
}

TEST_F(TensorOpsTest, GreaterEqual) {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 2.0f, 2.0f, 3.0f};

    auto custom_a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.ge(custom_b);
    auto torch_result = torch_a.ge(torch_b);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "GreaterEqual");

    // Test operator overload
    auto custom_result2 = custom_a >= custom_b;
    compare_tensors(custom_result2, torch_result, 1e-6f, 1e-7f, "GreaterEqual_Operator");
}

TEST_F(TensorOpsTest, ComparisonWithScalar) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float threshold = 3.0f;

    auto custom_gt = custom_tensor > threshold;
    auto torch_gt = torch_tensor > threshold;

    compare_tensors(custom_gt, torch_gt, 1e-6f, 1e-7f, "ScalarComparison_GT");

    auto custom_le = custom_tensor <= threshold;
    auto torch_le = torch_tensor <= threshold;

    compare_tensors(custom_le, torch_le, 1e-6f, 1e-7f, "ScalarComparison_LE");
}

// ============= Clamp Operations =============

TEST_F(TensorOpsTest, Clamp) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 3.0f, 8.0f, 12.0f};

    auto custom_tensor = Tensor::from_vector(data, {6}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float min_val = -1.0f;
    float max_val = 5.0f;

    auto custom_result = custom_tensor.clamp(min_val, max_val);
    auto torch_result = torch::clamp(torch_tensor, min_val, max_val);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Clamp");
}

TEST_F(TensorOpsTest, ClampMin) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 3.0f, 8.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float min_val = 0.0f;

    auto custom_result = custom_tensor.clamp_min(min_val);
    auto torch_result = torch::clamp_min(torch_tensor, min_val);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ClampMin");
}

TEST_F(TensorOpsTest, ClampMax) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 3.0f, 8.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float max_val = 5.0f;

    auto custom_result = custom_tensor.clamp_max(max_val);
    auto torch_result = torch::clamp_max(torch_tensor, max_val);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ClampMax");
}

TEST_F(TensorOpsTest, InPlaceClamp) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 3.0f, 8.0f, 12.0f};

    auto custom_tensor = Tensor::from_vector(data, {6}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float min_val = -1.0f;
    float max_val = 5.0f;

    custom_tensor.clamp_(min_val, max_val);
    torch_tensor.clamp_(min_val, max_val);

    compare_tensors(custom_tensor, torch_tensor, 1e-6f, 1e-7f, "InPlaceClamp");
}

// ============= Chained Operations =============

TEST_F(TensorOpsTest, ChainedOperations) {
    std::vector<float> data(9);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {3, 3}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({3, 3});

    // Chain operations: ((t + 2) * 3) - 1
    auto custom_result = ((custom_tensor + 2.0f) * 3.0f) - 1.0f;
    auto torch_result = ((torch_tensor + 2.0f) * 3.0f) - 1.0f;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ChainedOps");
}

TEST_F(TensorOpsTest, ComplexExpression) {
    std::vector<float> data_a(16), data_b(16);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {4, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4, 4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 4});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 4});

    // Complex expression: (a * 2 + b) / (a - b + 1)
    auto custom_result = (custom_a * 2.0f + custom_b) / (custom_a - custom_b + 1.0f);
    auto torch_result = (torch_a * 2.0f + torch_b) / (torch_a - torch_b + 1.0f);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ComplexExpression");
}

TEST_F(TensorOpsTest, MathematicalIdentity) {
    std::vector<float> data(10);
    for (auto& val : data)
        val = std::abs(dist(gen));

    auto custom_tensor = Tensor::from_vector(data, {10}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    // Test: log(exp(x)) ≈ x
    auto custom_result = custom_tensor.exp().log();
    auto torch_result = torch::log(torch::exp(torch_tensor));

    compare_tensors(custom_result, torch_tensor, 1e-4f, 1e-5f, "MathIdentity_LogExp");
}

// ============= Edge Cases =============

TEST_F(TensorOpsTest, ZeroTensor) {
    auto custom_zero = Tensor::zeros({4, 4}, Device::CUDA);
    auto torch_zero = torch::zeros({4, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_zero + 5.0f;
    auto torch_result = torch_zero + 5.0f;

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ZeroTensor");
}

TEST_F(TensorOpsTest, OneTensor) {
    auto custom_one = Tensor::ones({3, 3}, Device::CUDA);
    auto torch_one = torch::ones({3, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_one * 10.0f - 5.0f;
    auto torch_result = torch_one * 10.0f - 5.0f;

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "OneTensor");
}

TEST_F(TensorOpsTest, NegativeValues) {
    std::vector<float> data = {-1.0f, -2.0f, -3.0f, -4.0f};

    auto custom_tensor = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_abs = custom_tensor.abs();
    auto torch_abs = torch::abs(torch_tensor);

    compare_tensors(custom_abs, torch_abs, 1e-6f, 1e-7f, "NegativeAbs");

    auto custom_square = custom_tensor.square();
    auto torch_square = torch::square(torch_tensor);

    compare_tensors(custom_square, torch_square, 1e-5f, 1e-6f, "NegativeSquare");
}

TEST_F(TensorOpsTest, MixedSignValues) {
    std::vector<float> data = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    // Test operations on mixed sign values
    auto custom_add = custom_tensor + 2.0f;
    auto torch_add = torch_tensor + 2.0f;

    compare_tensors(custom_add, torch_add, 1e-6f, 1e-7f, "MixedSign_Add");

    auto custom_mul = custom_tensor * -1.0f;
    auto torch_mul = torch_tensor * -1.0f;

    compare_tensors(custom_mul, torch_mul, 1e-6f, 1e-7f, "MixedSign_Mul");
}

TEST_F(TensorOpsTest, LargeValues) {
    std::vector<float> data = {1e6f, 1e7f, 1e8f};

    auto custom_tensor = Tensor::from_vector(data, {3}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_div = custom_tensor / 1e6f;
    auto torch_div = torch_tensor / 1e6f;

    compare_tensors(custom_div, torch_div, 1e-3f, 1e-4f, "LargeValues");
}

TEST_F(TensorOpsTest, SmallValues) {
    std::vector<float> data = {1e-6f, 1e-7f, 1e-8f};

    auto custom_tensor = Tensor::from_vector(data, {3}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_mul = custom_tensor * 1e6f;
    auto torch_mul = torch_tensor * 1e6f;

    compare_tensors(custom_mul, torch_mul, 1e-9f, 1e-10f, "SmallValues");
}

// ============= CPU Operations =============

TEST_F(TensorOpsTest, CPUOperations) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_tensor = Tensor::from_vector(data, {4}, Device::CPU);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    auto custom_result = (custom_tensor + 1.0f) * 2.0f;
    auto torch_result = (torch_tensor + 1.0f) * 2.0f;

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CPU_Ops");
}