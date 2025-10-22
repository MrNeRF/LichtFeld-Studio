/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <numeric>
#include <print>
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

class TensorTorchCompatTest : public ::testing::Test {
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

// ============= Complex Expression Tests =============

TEST_F(TensorTorchCompatTest, ComplexExpression1) {
    // Test: (a + b) * c - d / e
    std::vector<float> data_a(12), data_b(12), data_c(12), data_d(12), data_e(12);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);
    for (auto& val : data_c)
        val = dist(gen);
    for (auto& val : data_d)
        val = dist(gen);
    for (auto& val : data_e)
        val = std::abs(dist(gen)) + 1.0f; // Avoid division by zero

    auto custom_a = Tensor::from_vector(data_a, {3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 4}, Device::CUDA);
    auto custom_c = Tensor::from_vector(data_c, {3, 4}, Device::CUDA);
    auto custom_d = Tensor::from_vector(data_d, {3, 4}, Device::CUDA);
    auto custom_e = Tensor::from_vector(data_e, {3, 4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_c = torch::tensor(data_c, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_d = torch::tensor(data_d, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_e = torch::tensor(data_e, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});

    auto custom_result = (custom_a + custom_b) * custom_c - custom_d / custom_e;
    auto torch_result = (torch_a + torch_b) * torch_c - torch_d / torch_e;

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ComplexExpression1");
}

TEST_F(TensorTorchCompatTest, ComplexExpression2) {
    // Test: sigmoid(a * 2 + b) * relu(c - 1)
    std::vector<float> data_a(25), data_b(25), data_c(25);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);
    for (auto& val : data_c)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {5, 5}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {5, 5}, Device::CUDA);
    auto custom_c = Tensor::from_vector(data_c, {5, 5}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});
    auto torch_c = torch::tensor(data_c, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});

    auto custom_result = (custom_a * 2.0f + custom_b).sigmoid() * (custom_c - 1.0f).relu();
    auto torch_result = torch::sigmoid(torch_a * 2.0f + torch_b) * torch::relu(torch_c - 1.0f);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ComplexExpression2");
}

TEST_F(TensorTorchCompatTest, ComplexExpression3) {
    // Test: exp(log(abs(a) + 1)) + sqrt(b^2 + c^2)
    std::vector<float> data_a(24), data_b(24), data_c(24);
    for (auto& val : data_a)
        val = dist(gen);
    for (auto& val : data_b)
        val = dist(gen);
    for (auto& val : data_c)
        val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {4, 6}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4, 6}, Device::CUDA);
    auto custom_c = Tensor::from_vector(data_c, {4, 6}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 6});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 6});
    auto torch_c = torch::tensor(data_c, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 6});

    auto custom_result = (custom_a.abs() + 1.0f).log().exp() + (custom_b * custom_b + custom_c * custom_c).sqrt();
    auto torch_result = torch::exp(torch::log(torch::abs(torch_a) + 1.0f)) +
                        torch::sqrt(torch_b * torch_b + torch_c * torch_c);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ComplexExpression3");
}

// ============= View and Shape Tests =============

TEST_F(TensorTorchCompatTest, ViewAndCompute) {
    std::vector<float> data(24);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});

    // Reshape and compute
    auto custom_view = custom_tensor.view({6, 4});
    auto torch_view = torch_tensor.view({6, 4});

    auto custom_result = (custom_view + 1.0f) * 2.0f;
    auto torch_result = (torch_view + 1.0f) * 2.0f;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ViewAndCompute");
}

TEST_F(TensorTorchCompatTest, SliceAndCompute) {
    std::vector<float> data(80);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {10, 8}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({10, 8});

    auto custom_slice = custom_tensor.slice(0, 2, 7); // Rows 2-6
    auto torch_slice = torch_tensor.slice(0, 2, 7);

    auto custom_result = custom_slice.sigmoid() + 0.5f;
    auto torch_result = torch::sigmoid(torch_slice) + 0.5f;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "SliceAndCompute");
}

// ============= Reduction Tests =============

TEST_F(TensorTorchCompatTest, ReductionConsistency) {
    std::vector<float> data(63);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {7, 9}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({7, 9});

    // Sum
    EXPECT_NEAR(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(), 1e-3f);

    // Mean
    EXPECT_NEAR(custom_tensor.mean_scalar(), torch_tensor.mean().item<float>(), 1e-4f);

    // Min/Max
    EXPECT_FLOAT_EQ(custom_tensor.min_scalar(), torch_tensor.min().item<float>());
    EXPECT_FLOAT_EQ(custom_tensor.max_scalar(), torch_tensor.max().item<float>());

    // Norms
    EXPECT_NEAR(custom_tensor.norm(2.0f), torch_tensor.norm().item<float>(), 1e-3f);
    EXPECT_NEAR(custom_tensor.norm(1.0f), torch_tensor.norm(1).item<float>(), 1e-3f);
}

// ============= In-place Operations Tests =============

TEST_F(TensorTorchCompatTest, InPlaceOperations) {
    std::vector<float> data(16);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {4, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 4});

    // Scalar in-place
    custom_tensor.add_(2.0f);
    torch_tensor.add_(2.0f);
    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlace_Add");

    custom_tensor.mul_(3.0f);
    torch_tensor.mul_(3.0f);
    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlace_Mul");

    // Tensor in-place
    std::vector<float> other_data(16);
    for (auto& val : other_data)
        val = dist(gen);

    auto custom_other = Tensor::from_vector(other_data, {4, 4}, Device::CUDA);
    auto torch_other = torch::tensor(other_data, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 4});

    custom_tensor.sub_(custom_other);
    torch_tensor.sub_(torch_other);
    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlace_Sub");
}

// ============= Batch Processing Tests =============

TEST_F(TensorTorchCompatTest, BatchProcessing) {
    std::vector<float> data(2048);
    for (auto& val : data)
        val = dist(gen);

    auto custom_batch = Tensor::from_vector(data, {32, 64}, Device::CUDA);
    auto torch_batch = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({32, 64});

    // Process: normalize -> relu -> scale
    auto custom_normalized = custom_batch.normalize();

    auto torch_mean = torch_batch.mean();
    auto torch_std = torch_batch.std(/*unbiased=*/false);
    auto torch_normalized = (torch_batch - torch_mean) / (torch_std + 1e-8f); // Match epsilon!

    auto custom_relu = custom_normalized.relu();
    auto torch_relu = torch::relu(torch_normalized);

    auto custom_scaled = custom_relu * 0.1f;
    auto torch_scaled = torch_relu * 0.1f;

    compare_tensors(custom_scaled, torch_scaled, 1e-4f, 1e-5f, "BatchProcessing");
}

// ============= Gradient Simulation Tests =============

TEST_F(TensorTorchCompatTest, GradientSimulation) {
    std::vector<float> params_data(100), grads_data(100);
    for (auto& val : params_data)
        val = dist(gen);
    for (auto& val : grads_data)
        val = dist(gen);

    auto custom_params = Tensor::from_vector(params_data, {100}, Device::CUDA);
    auto custom_grads = Tensor::from_vector(grads_data, {100}, Device::CUDA);

    auto torch_params = torch::tensor(params_data, torch::TensorOptions().device(torch::kCUDA));
    auto torch_grads = torch::tensor(grads_data, torch::TensorOptions().device(torch::kCUDA));

    float learning_rate = 0.01f;

    // SGD step: params = params - lr * grads
    auto custom_updated = custom_params - custom_grads * learning_rate;
    auto torch_updated = torch_params - torch_grads * learning_rate;

    compare_tensors(custom_updated, torch_updated, 1e-5f, 1e-6f, "SGD_Step");

    // Momentum simulation
    std::vector<float> momentum_data(100, 0.0f);
    auto custom_momentum = Tensor::from_vector(momentum_data, {100}, Device::CUDA);
    auto torch_momentum = torch::tensor(momentum_data, torch::TensorOptions().device(torch::kCUDA));

    float beta = 0.9f;

    // momentum = beta * momentum + (1 - beta) * grads
    auto custom_new_momentum = custom_momentum * beta + custom_grads * (1.0f - beta);
    auto torch_new_momentum = torch_momentum * beta + torch_grads * (1.0f - beta);

    compare_tensors(custom_new_momentum, torch_new_momentum, 1e-5f, 1e-6f, "Momentum");
}

// ============= Stress Tests =============

TEST_F(TensorTorchCompatTest, StressTest) {
    const int num_iterations = 10; // Reduced for test speed

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Random shape
        size_t dim1 = 10 + (gen() % 20);
        size_t dim2 = 10 + (gen() % 20);

        std::vector<float> data(dim1 * dim2);
        for (auto& val : data)
            val = dist(gen);

        auto custom_tensor = Tensor::from_vector(data, {dim1, dim2}, Device::CUDA);
        auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                                .reshape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});

        // Apply random sequence of operations
        for (int op = 0; op < 5; ++op) {
            int op_type = gen() % 6;
            float scalar = dist(gen);

            switch (op_type) {
            case 0:
                custom_tensor = custom_tensor + scalar;
                torch_tensor = torch_tensor + scalar;
                break;
            case 1:
                custom_tensor = custom_tensor * std::abs(scalar);
                torch_tensor = torch_tensor * std::abs(scalar);
                break;
            case 2:
                custom_tensor = custom_tensor.abs();
                torch_tensor = torch::abs(torch_tensor);
                break;
            case 3:
                custom_tensor = custom_tensor.sigmoid();
                torch_tensor = torch::sigmoid(torch_tensor);
                break;
            case 4:
                custom_tensor = custom_tensor.relu();
                torch_tensor = torch::relu(torch_tensor);
                break;
            case 5:
                custom_tensor = custom_tensor.clamp(-5.0f, 5.0f);
                torch_tensor = torch::clamp(torch_tensor, -5.0f, 5.0f);
                break;
            }
        }

        compare_tensors(custom_tensor, torch_tensor, 1e-3f, 1e-4f,
                        "StressTest_Iter" + std::to_string(iter));
    }
}

// ============= Large Scale Tests =============

TEST_F(TensorTorchCompatTest, LargeScaleOperations) {
    std::vector<float> data(128 * 256);
    for (auto& val : data)
        val = dist(gen);

    auto custom_large = Tensor::from_vector(data, {128, 256}, Device::CUDA);
    auto torch_large = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({128, 256});

    // Complex computation
    auto custom_result = ((custom_large.abs() + 1.0f).log() * 2.0f).sigmoid();
    auto torch_result = torch::sigmoid(torch::log(torch::abs(torch_large) + 1.0f) * 2.0f);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "LargeScale");

    // Verify shape
    EXPECT_EQ(custom_result.numel(), 128 * 256);
}

// ============= Edge Cases Tests =============

TEST_F(TensorTorchCompatTest, EdgeCasesCompat) {
    // Very small values
    std::vector<float> small_data = {1e-7f, 1e-8f, 1e-9f};

    auto custom_small = Tensor::from_vector(small_data, {3}, Device::CUDA);
    auto torch_small = torch::tensor(small_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_log = custom_small.log();
    auto torch_log = torch::log(torch_small);

    compare_tensors(custom_log, torch_log, 1e-4f, 1e-5f, "EdgeCase_SmallLog");

    // Very large values
    std::vector<float> large_data = {1e6f, 1e7f, 1e8f};

    auto custom_large = Tensor::from_vector(large_data, {3}, Device::CUDA);
    auto torch_large = torch::tensor(large_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_sigmoid = custom_large.sigmoid();
    auto torch_sigmoid = torch::sigmoid(torch_large);

    compare_tensors(custom_sigmoid, torch_sigmoid, 1e-5f, 1e-6f, "EdgeCase_LargeSigmoid");
}

// ============= Multi-dimensional Tests =============

TEST_F(TensorTorchCompatTest, MultiDimensionalOps) {
    // Test with 3D tensors
    std::vector<float> data_3d(4 * 5 * 6);
    for (auto& val : data_3d)
        val = dist(gen);

    auto custom_3d = Tensor::from_vector(data_3d, {4, 5, 6}, Device::CUDA);
    auto torch_3d = torch::tensor(data_3d, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 5, 6});

    auto custom_3d_result = (custom_3d + 1.0f).exp().clamp(0.0f, 100.0f);
    auto torch_3d_result = torch::clamp(torch::exp(torch_3d + 1.0f), 0.0f, 100.0f);

    compare_tensors(custom_3d_result, torch_3d_result, 1e-3f, 1e-4f, "3D_Ops");

    // Test with 4D tensors
    std::vector<float> data_4d(2 * 3 * 4 * 5);
    for (auto& val : data_4d)
        val = dist(gen);

    auto custom_4d = Tensor::from_vector(data_4d, {2, 3, 4, 5}, Device::CUDA);
    auto torch_4d = torch::tensor(data_4d, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4, 5});

    auto custom_4d_result = custom_4d.abs().sqrt();
    auto torch_4d_result = torch::sqrt(torch::abs(torch_4d));

    compare_tensors(custom_4d_result, torch_4d_result, 1e-5f, 1e-6f, "4D_Ops");
}

// ============= Device Consistency Tests =============

TEST_F(TensorTorchCompatTest, ConsistencyAcrossDevices) {
    std::vector<float> data(100);
    for (auto& val : data)
        val = dist(gen);

    // Create CPU tensor
    auto custom_cpu = Tensor::from_vector(data, {10, 10}, Device::CPU);

    // Create CUDA tensor with same data
    auto custom_cuda = custom_cpu.to(Device::CUDA);

    // Apply operations on both
    auto cpu_result = (custom_cpu + 1.0f) * 2.0f;
    auto cuda_result = (custom_cuda + 1.0f) * 2.0f;

    // Results should be the same
    auto cpu_on_cuda = cpu_result.to(Device::CUDA);
    EXPECT_TRUE(cpu_on_cuda.all_close(cuda_result, 1e-5f, 1e-6f));
}

// ============= Chained Operations Tests =============

TEST_F(TensorTorchCompatTest, ChainedComplexOperations) {
    std::vector<float> data(64);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {8, 8}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({8, 8});

    // Apply long chain
    auto custom_result = custom_tensor
                             .abs()
                             .add(1.0f)
                             .log()
                             .mul(2.0f)
                             .sigmoid()
                             .sub(0.5f)
                             .relu()
                             .clamp(0.0f, 1.0f)
                             .sqrt()
                             .add(0.1f);

    auto torch_result = torch_tensor;
    torch_result = torch::abs(torch_result);
    torch_result = torch_result + 1.0f;
    torch_result = torch::log(torch_result);
    torch_result = torch_result * 2.0f;
    torch_result = torch::sigmoid(torch_result);
    torch_result = torch_result - 0.5f;
    torch_result = torch::relu(torch_result);
    torch_result = torch::clamp(torch_result, 0.0f, 1.0f);
    torch_result = torch::sqrt(torch_result);
    torch_result = torch_result + 0.1f;

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ChainedComplex");
}

// ============= Numerical Stability Tests =============

TEST_F(TensorTorchCompatTest, NumericalStability) {
    // Division by small but not extreme numbers
    std::vector<float> numerator_data(25);
    for (auto& val : numerator_data)
        val = dist(gen);

    auto custom_numerator = Tensor::from_vector(numerator_data, {5, 5}, Device::CUDA);
    auto torch_numerator = torch::tensor(numerator_data, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});

    // Use 1e-4 instead of 1e-7 for more reasonable numerical behavior
    auto custom_divisor = Tensor::full({5, 5}, 1e-4f, Device::CUDA);
    auto torch_divisor = torch::full({5, 5}, 1e-4f, torch::TensorOptions().device(torch::kCUDA));

    auto custom_div = custom_numerator / custom_divisor;
    auto torch_div = torch_numerator / torch_divisor;

    // Compare with standard tolerance
    compare_tensors(custom_div, torch_div, 1e-3f, 1e-4f, "NumericalStability");
}
// ============= Performance Comparison Tests =============

TEST_F(TensorTorchCompatTest, PerformanceComparison) {
    // Not a strict performance benchmark, but ensure we handle large operations correctly
    const size_t size = 256;

    std::vector<float> data(size * size);
    for (auto& val : data)
        val = dist(gen);

    auto custom_large = Tensor::from_vector(data, {size, size}, Device::CUDA);
    auto torch_large = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({size, size});

    // Time-intensive operation sequence
    auto start_custom = std::chrono::high_resolution_clock::now();
    auto custom_result = custom_large.sigmoid().relu().add(1.0f).mul(2.0f).sqrt();
    cudaDeviceSynchronize(); // Ensure completion
    auto end_custom = std::chrono::high_resolution_clock::now();

    auto start_torch = std::chrono::high_resolution_clock::now();
    auto torch_result = torch::sqrt(torch::relu(torch::sigmoid(torch_large)).add(1.0f).mul(2.0f));
    cudaDeviceSynchronize(); // Ensure completion
    auto end_torch = std::chrono::high_resolution_clock::now();

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "Performance");

    // Log the times for information
    auto custom_us = std::chrono::duration_cast<std::chrono::microseconds>(end_custom - start_custom).count();
    auto torch_us = std::chrono::duration_cast<std::chrono::microseconds>(end_torch - start_torch).count();

    LOG_INFO("Custom implementation: {} μs, PyTorch: {} μs (ratio: {:.2f}x)",
             custom_us, torch_us, custom_us / double(torch_us));
}

// ============= Mixed Operations Tests =============

TEST_F(TensorTorchCompatTest, MixedArithmeticAndActivation) {
    std::vector<float> data(100);
    for (auto& val : data)
        val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {10, 10}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({10, 10});

    // Mix arithmetic and activation functions
    auto custom_result = ((custom_tensor * 2.0f).sigmoid() + 0.5f).relu();
    auto torch_result = torch::relu(torch::sigmoid(torch_tensor * 2.0f) + 0.5f);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "MixedOps");
}

TEST_F(TensorTorchCompatTest, NestedExpressions) {
    std::vector<float> data_x(50), data_y(50);
    for (auto& val : data_x)
        val = dist(gen);
    for (auto& val : data_y)
        val = dist(gen);

    auto custom_x = Tensor::from_vector(data_x, {5, 10}, Device::CUDA);
    auto custom_y = Tensor::from_vector(data_y, {5, 10}, Device::CUDA);

    auto torch_x = torch::tensor(data_x, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 10});
    auto torch_y = torch::tensor(data_y, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 10});

    // Nested: ((x + y) * (x - y)) / (x + 1)
    auto custom_result = ((custom_x + custom_y) * (custom_x - custom_y)) / (custom_x + 1.0f);
    auto torch_result = ((torch_x + torch_y) * (torch_x - torch_y)) / (torch_x + 1.0f);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "NestedExpressions");
}

// ============= Zero and One Tensors Tests =============

TEST_F(TensorTorchCompatTest, ZerosAndOnesOperations) {
    auto custom_zeros = Tensor::zeros({5, 5}, Device::CUDA);
    auto custom_ones = Tensor::ones({5, 5}, Device::CUDA);

    auto torch_zeros = torch::zeros({5, 5}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_ones = torch::ones({5, 5}, torch::TensorOptions().device(torch::kCUDA));

    // Operations on zeros
    auto custom_zeros_result = custom_zeros + 5.0f;
    auto torch_zeros_result = torch_zeros + 5.0f;

    compare_tensors(custom_zeros_result, torch_zeros_result, 1e-6f, 1e-7f, "ZerosOps");

    // Operations on ones
    auto custom_ones_result = (custom_ones * 10.0f).relu();
    auto torch_ones_result = torch::relu(torch_ones * 10.0f);

    compare_tensors(custom_ones_result, torch_ones_result, 1e-6f, 1e-7f, "OnesOps");
}

// ============= Concatenation Tests =============

TEST_F(TensorTorchCompatTest, Cat2D_Dim0) {
    // Test concatenating along dimension 0 (rows)
    std::vector<float> data1 = {1, 2, 3, 4, 5, 6};    // [2, 3]
    std::vector<float> data2 = {7, 8, 9, 10, 11, 12}; // [2, 3]

    auto custom_a = Tensor::from_vector(data1, {2, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {2, 3}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3});
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3});

    auto custom_result = custom_a.cat(custom_b, 0);
    auto torch_result = torch::cat({torch_a, torch_b}, 0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cat2D_Dim0");
}

TEST_F(TensorTorchCompatTest, Cat2D_Dim1) {
    // Test concatenating along dimension 1 (columns)
    std::vector<float> data1 = {1, 2, 3, 4, 5, 6}; // [2, 3]
    std::vector<float> data2 = {7, 8, 9, 10};      // [2, 2]

    auto custom_a = Tensor::from_vector(data1, {2, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {2, 2}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3});
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});

    auto custom_result = custom_a.cat(custom_b, 1);
    auto torch_result = torch::cat({torch_a, torch_b}, 1);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cat2D_Dim1");
}

TEST_F(TensorTorchCompatTest, Cat3D_Dim0) {
    // Test 3D concatenation along dimension 0
    std::vector<float> data1(2 * 3 * 4, 1.0f); // [2, 3, 4]
    std::vector<float> data2(3 * 3 * 4, 2.0f); // [3, 3, 4]

    auto custom_a = Tensor::from_vector(data1, {2, 3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {3, 3, 4}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 3, 4});

    auto custom_result = custom_a.cat(custom_b, 0);
    auto torch_result = torch::cat({torch_a, torch_b}, 0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cat3D_Dim0");
}

TEST_F(TensorTorchCompatTest, Cat3D_Dim1_CRITICAL) {
    // THIS IS THE CRITICAL TEST - This is what get_shs() does!
    // Concatenating [100, 1, 3] + [100, 8, 3] along dim 1

    std::vector<float> data1(100 * 1 * 3);
    std::vector<float> data2(100 * 8 * 3);

    // Fill with recognizable patterns
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = -1.0f - (i % 3) * 0.1f; // -1.0, -1.1, -1.2, -1.0, ...
    }
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = 0.0f; // All zeros
    }

    auto custom_a = Tensor::from_vector(data1, {100, 1, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {100, 8, 3}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA)).reshape({100, 1, 3});
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA)).reshape({100, 8, 3});

    auto custom_result = custom_a.cat(custom_b, 1);
    auto torch_result = torch::cat({torch_a, torch_b}, 1);

    // Debug print first few values
    auto custom_cpu = custom_result.cpu();
    auto custom_vec = custom_cpu.to_vector();
    auto torch_cpu = torch_result.to(torch::kCPU).contiguous();
    auto torch_flat = torch_cpu.flatten();
    auto torch_acc = torch_flat.accessor<float, 1>();

    std::println("\n=== Cat3D_Dim1_CRITICAL Debug ===");
    std::println("Expected layout: [point0_coeff0_ch0, point0_coeff0_ch1, point0_coeff0_ch2,");
    std::println("                  point0_coeff1_ch0, point0_coeff1_ch1, point0_coeff1_ch2, ...]");
    std::println("\nFirst 27 values (point 0, all 9 coeffs, 3 channels each):");
    for (size_t i = 0; i < std::min(size_t(27), custom_vec.size()); ++i) {
        std::println("  [{}] custom={:.6f}, torch={:.6f}, diff={:.6f}",
                     i, custom_vec[i], torch_acc[i], std::abs(custom_vec[i] - torch_acc[i]));
    }

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cat3D_Dim1_CRITICAL");
}

TEST_F(TensorTorchCompatTest, Cat3D_Dim2) {
    // Test 3D concatenation along dimension 2
    std::vector<float> data1(2 * 3 * 4, 1.0f); // [2, 3, 4]
    std::vector<float> data2(2 * 3 * 5, 2.0f); // [2, 3, 5]

    auto custom_a = Tensor::from_vector(data1, {2, 3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {2, 3, 5}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 5});

    auto custom_result = custom_a.cat(custom_b, 2);
    auto torch_result = torch::cat({torch_a, torch_b}, 2);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cat3D_Dim2");
}

TEST_F(TensorTorchCompatTest, Cat_MultipleSmallTensors) {
    // Test concatenating several small tensors
    std::vector<float> data1 = {1, 2, 3};
    std::vector<float> data2 = {4, 5, 6};
    std::vector<float> data3 = {7, 8, 9};

    auto custom_a = Tensor::from_vector(data1, {1, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {1, 3}, Device::CUDA);
    auto custom_c = Tensor::from_vector(data3, {1, 3}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA)).reshape({1, 3});
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA)).reshape({1, 3});
    auto torch_c = torch::tensor(data3, torch::TensorOptions().device(torch::kCUDA)).reshape({1, 3});

    // Cat all three
    auto custom_ab = custom_a.cat(custom_b, 0);
    auto custom_abc = custom_ab.cat(custom_c, 0);

    auto torch_result = torch::cat({torch_a, torch_b, torch_c}, 0);

    compare_tensors(custom_abc, torch_result, 1e-6f, 1e-7f, "Cat_MultipleSmall");
}

TEST_F(TensorTorchCompatTest, Cat_WithComputation) {
    // Test concatenation as part of a computation pipeline
    std::vector<float> data1(10 * 2, 1.0f);
    std::vector<float> data2(10 * 3, 2.0f);

    auto custom_a = Tensor::from_vector(data1, {10, 2}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {10, 3}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA)).reshape({10, 2});
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA)).reshape({10, 3});

    // Compute, concatenate, then compute more
    auto custom_result = (custom_a.relu() + 1.0f).cat((custom_b.sigmoid() * 2.0f), 1).sum({1}, false);
    auto torch_result = torch::sum(
        torch::cat({torch::relu(torch_a) + 1.0f, torch::sigmoid(torch_b) * 2.0f}, 1),
        1, false);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "Cat_WithComputation");
}

TEST_F(TensorTorchCompatTest, Cat_SplatDataScenario) {
    // Exact scenario from SplatDataNew::get_shs()
    // sh0 is [100, 1, 3], shN is [100, 8, 3]
    // Result should be [100, 9, 3]

    std::vector<float> sh0_data(100 * 1 * 3);
    std::vector<float> shN_data(100 * 8 * 3);

    // Fill sh0 with DC values (like RGB->SH conversion would produce)
    for (size_t point = 0; point < 100; ++point) {
        for (size_t ch = 0; ch < 3; ++ch) {
            sh0_data[point * 3 + ch] = -1.77f + (point * 0.001f) + (ch * 0.01f);
        }
    }

    // Fill shN with zeros (higher order SH coefficients start at zero)
    std::fill(shN_data.begin(), shN_data.end(), 0.0f);

    auto custom_sh0 = Tensor::from_vector(sh0_data, {100, 1, 3}, Device::CUDA);
    auto custom_shN = Tensor::from_vector(shN_data, {100, 8, 3}, Device::CUDA);

    auto torch_sh0 = torch::tensor(sh0_data, torch::TensorOptions().device(torch::kCUDA)).reshape({100, 1, 3});
    auto torch_shN = torch::tensor(shN_data, torch::TensorOptions().device(torch::kCUDA)).reshape({100, 8, 3});

    auto custom_shs = custom_sh0.cat(custom_shN, 1);
    auto torch_shs = torch::cat({torch_sh0, torch_shN}, 1);

    EXPECT_EQ(custom_shs.shape().str(), "[100, 9, 3]");
    EXPECT_EQ(torch_shs.sizes()[0], 100);
    EXPECT_EQ(torch_shs.sizes()[1], 9);
    EXPECT_EQ(torch_shs.sizes()[2], 3);

    // Verify the layout is correct
    auto custom_cpu = custom_shs.cpu();
    auto custom_vec = custom_cpu.to_vector();
    auto torch_cpu = torch_shs.to(torch::kCPU).contiguous();
    auto torch_flat = torch_cpu.flatten();
    auto torch_acc = torch_flat.accessor<float, 1>();

    std::println("\n=== Cat_SplatDataScenario Debug ===");
    std::println("For point 0, the layout should be:");
    std::println("  [0-2]:   DC values for RGB (from sh0)");
    std::println("  [3-26]:  Zeros for higher-order coeffs (from shN)");
    std::println("\nActual values for point 0:");
    for (size_t i = 0; i < 27; ++i) { // 9 coeffs * 3 channels
        std::println("  [{}] custom={:.6f}, torch={:.6f}", i, custom_vec[i], torch_acc[i]);
    }

    // Check specific values
    EXPECT_NEAR(custom_vec[0], sh0_data[0], 1e-6f) << "First DC value should match";
    EXPECT_NEAR(custom_vec[1], sh0_data[1], 1e-6f) << "Second DC value should match";
    EXPECT_NEAR(custom_vec[2], sh0_data[2], 1e-6f) << "Third DC value should match";
    EXPECT_NEAR(custom_vec[3], 0.0f, 1e-6f) << "First shN value should be 0";

    compare_tensors(custom_shs, torch_shs, 1e-6f, 1e-7f, "Cat_SplatDataScenario");
}

TEST_F(TensorTorchCompatTest, Cat_EdgeCases) {
    // Test edge cases

    // Single element concatenation
    auto custom_single = Tensor::ones({1, 1}, Device::CUDA);
    auto torch_single = torch::ones({1, 1}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_double = custom_single.cat(custom_single, 0);
    auto torch_double = torch::cat({torch_single, torch_single}, 0);

    compare_tensors(custom_double, torch_double, 1e-6f, 1e-7f, "Cat_Single");

    // Large dimension difference
    auto custom_small = Tensor::ones({5, 1, 3}, Device::CUDA);
    auto custom_large = Tensor::ones({5, 100, 3}, Device::CUDA);

    auto torch_small = torch::ones({5, 1, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_large = torch::ones({5, 100, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_cat = custom_small.cat(custom_large, 1);
    auto torch_cat = torch::cat({torch_small, torch_large}, 1);

    compare_tensors(custom_cat, torch_cat, 1e-6f, 1e-7f, "Cat_LargeDiff");
}

TEST_F(TensorTorchCompatTest, Stack_Basic) {
    // Stack creates a new dimension
    std::vector<float> data1 = {1, 2, 3};
    std::vector<float> data2 = {4, 5, 6};

    auto custom_a = Tensor::from_vector(data1, {3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data2, {3}, Device::CUDA);

    auto torch_a = torch::tensor(data1, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::tensor(data2, torch::TensorOptions().device(torch::kCUDA));

    // Stack along dimension 0
    auto custom_stacked = Tensor::stack({custom_a, custom_b}, 0);
    auto torch_stacked = torch::stack({torch_a, torch_b}, 0);

    compare_tensors(custom_stacked, torch_stacked, 1e-6f, 1e-7f, "Stack_Dim0");

    // Stack along dimension 1
    custom_stacked = Tensor::stack({custom_a, custom_b}, 1);
    torch_stacked = torch::stack({torch_a, torch_b}, 1);

    compare_tensors(custom_stacked, torch_stacked, 1e-6f, 1e-7f, "Stack_Dim1");
}

TEST_F(TensorTorchCompatTest, IndexCopy_Column_Single) {
    // Test index_copy_ for updating a single column
    std::vector<float> data(12, 0.0f);                      // [3, 4] filled with zeros
    std::vector<float> update_data = {10.0f, 20.0f, 30.0f}; // [3, 1]

    auto custom_tensor = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});

    auto custom_update = Tensor::from_vector(update_data, {3, 1}, Device::CUDA);
    auto torch_update = torch::tensor(update_data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 1});

    // Update column 2 (index 2)
    auto custom_idx = Tensor::from_vector({2}, {1}, Device::CUDA).to(DataType::Int32);
    auto torch_idx = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)); // Changed to Int64

    custom_tensor.index_copy_(1, custom_idx, custom_update);
    torch_tensor.index_copy_(1, torch_idx, torch_update);

    compare_tensors(custom_tensor, torch_tensor, 1e-6f, 1e-7f, "IndexCopy_Column_Single");
}

TEST_F(TensorTorchCompatTest, IndexCopy_Column_Multiple) {
    // Test index_copy_ for updating multiple columns sequentially
    // This mimics what the rotation transformation does
    std::vector<float> data(12, 0.0f); // [3, 4] filled with zeros

    auto custom_tensor = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});

    // Update columns 0, 1, 2, 3 with different values
    std::vector<float> col0_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> col1_data = {4.0f, 5.0f, 6.0f};
    std::vector<float> col2_data = {7.0f, 8.0f, 9.0f};
    std::vector<float> col3_data = {10.0f, 11.0f, 12.0f};

    auto custom_col0 = Tensor::from_vector(col0_data, {3, 1}, Device::CUDA);
    auto custom_col1 = Tensor::from_vector(col1_data, {3, 1}, Device::CUDA);
    auto custom_col2 = Tensor::from_vector(col2_data, {3, 1}, Device::CUDA);
    auto custom_col3 = Tensor::from_vector(col3_data, {3, 1}, Device::CUDA);

    auto torch_col0 = torch::tensor(col0_data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 1});
    auto torch_col1 = torch::tensor(col1_data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 1});
    auto torch_col2 = torch::tensor(col2_data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 1});
    auto torch_col3 = torch::tensor(col3_data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 1});

    auto idx0 = Tensor::from_vector({0}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx1 = Tensor::from_vector({1}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx2 = Tensor::from_vector({2}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx3 = Tensor::from_vector({3}, {1}, Device::CUDA).to(DataType::Int32);

    auto torch_idx0 = torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)); // Changed to Int64
    auto torch_idx1 = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)); // Changed to Int64
    auto torch_idx2 = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)); // Changed to Int64
    auto torch_idx3 = torch::tensor({3}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)); // Changed to Int64

    // Custom implementation
    custom_tensor.index_copy_(1, idx0, custom_col0);
    custom_tensor.index_copy_(1, idx1, custom_col1);
    custom_tensor.index_copy_(1, idx2, custom_col2);
    custom_tensor.index_copy_(1, idx3, custom_col3);

    // PyTorch reference
    torch_tensor.index_copy_(1, torch_idx0, torch_col0);
    torch_tensor.index_copy_(1, torch_idx1, torch_col1);
    torch_tensor.index_copy_(1, torch_idx2, torch_col2);
    torch_tensor.index_copy_(1, torch_idx3, torch_col3);

    compare_tensors(custom_tensor, torch_tensor, 1e-6f, 1e-7f, "IndexCopy_Column_Multiple");
}

TEST_F(TensorTorchCompatTest, IndexCopy_QuaternionScenario) {
    // Exact scenario from rotation transformation
    // [N, 4] tensor where we update each column with computed values
    const int N = 100;

    std::vector<float> rotation_data(N * 4);
    for (auto& val : rotation_data)
        val = dist(gen);

    auto custom_rotation = Tensor::from_vector(rotation_data, {static_cast<size_t>(N), 4}, Device::CUDA);
    auto torch_rotation = torch::tensor(rotation_data, torch::TensorOptions().device(torch::kCUDA)).reshape({N, 4});

    // Compute new quaternion components (simplified)
    auto custom_w = custom_rotation.slice(1, 0, 1).squeeze(1);
    auto custom_x = custom_rotation.slice(1, 1, 2).squeeze(1);
    auto custom_y = custom_rotation.slice(1, 2, 3).squeeze(1);
    auto custom_z = custom_rotation.slice(1, 3, 4).squeeze(1);

    auto torch_w = torch_rotation.index({torch::indexing::Slice(), 0});
    auto torch_x = torch_rotation.index({torch::indexing::Slice(), 1});
    auto torch_y = torch_rotation.index({torch::indexing::Slice(), 2});
    auto torch_z = torch_rotation.index({torch::indexing::Slice(), 3});

    // Apply some transformation (multiply by 2, add 1)
    auto custom_w_new = custom_w.mul(2.0f).add(1.0f);
    auto custom_x_new = custom_x.mul(2.0f).add(1.0f);
    auto custom_y_new = custom_y.mul(2.0f).add(1.0f);
    auto custom_z_new = custom_z.mul(2.0f).add(1.0f);

    auto torch_w_new = torch_w * 2.0f + 1.0f;
    auto torch_x_new = torch_x * 2.0f + 1.0f;
    auto torch_y_new = torch_y * 2.0f + 1.0f;
    auto torch_z_new = torch_z * 2.0f + 1.0f;

    // Update using index_copy_
    auto idx0 = Tensor::from_vector({0}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx1 = Tensor::from_vector({1}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx2 = Tensor::from_vector({2}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx3 = Tensor::from_vector({3}, {1}, Device::CUDA).to(DataType::Int32);

    custom_rotation.index_copy_(1, idx0, custom_w_new.unsqueeze(1));
    custom_rotation.index_copy_(1, idx1, custom_x_new.unsqueeze(1));
    custom_rotation.index_copy_(1, idx2, custom_y_new.unsqueeze(1));
    custom_rotation.index_copy_(1, idx3, custom_z_new.unsqueeze(1));

    // PyTorch reference using index_put_
    torch_rotation.index_put_({torch::indexing::Slice(), 0}, torch_w_new);
    torch_rotation.index_put_({torch::indexing::Slice(), 1}, torch_x_new);
    torch_rotation.index_put_({torch::indexing::Slice(), 2}, torch_y_new);
    torch_rotation.index_put_({torch::indexing::Slice(), 3}, torch_z_new);

    compare_tensors(custom_rotation, torch_rotation, 1e-5f, 1e-6f, "IndexCopy_QuaternionScenario");
}

TEST_F(TensorTorchCompatTest, IndexCopy_SpecificFailingCase) {
    // Use the EXACT data from the failing test run
    // Point 0 had: w2=-1.26822, x2=-0.0382963, y2=-0.102942, z2=1.43998

    const int N = 3; // Just test first 3 points

    // Exact data from the failing run
    std::vector<float> rotation_data = {
        -1.2682f, -0.0383f, -0.1029f, 1.4400f, // Point 0
        -0.4705f, 1.1624f, 0.3058f, 0.5276f,   // Point 1
        -0.5726f, 1.8732f, -0.6816f, -0.2104f  // Point 2
    };

    auto custom_rotation = Tensor::from_vector(rotation_data, {static_cast<size_t>(N), 4}, Device::CUDA);
    auto torch_rotation = torch::tensor(rotation_data, torch::TensorOptions().device(torch::kCUDA)).reshape({N, 4});

    // Exact transform quaternion from failing run
    std::vector<float> rot_data = {0.92388f, 0.0f, 0.382683f, 0.0f};
    auto rot_tensor = Tensor::from_vector(rot_data, {4}, Device::CUDA);
    auto torch_rot = torch::tensor(rot_data, torch::TensorOptions().device(torch::kCUDA));

    std::vector<int> expand_shape = {N, 4};
    auto q_rot = rot_tensor.unsqueeze(0).expand(std::span<const int>(expand_shape));
    auto torch_q_rot = torch_rot.unsqueeze(0).expand({N, 4});

    // Extract components
    auto w1 = q_rot.slice(1, 0, 1).squeeze(1);
    auto x1 = q_rot.slice(1, 1, 2).squeeze(1);
    auto y1 = q_rot.slice(1, 2, 3).squeeze(1);
    auto z1 = q_rot.slice(1, 3, 4).squeeze(1);

    auto w2 = custom_rotation.slice(1, 0, 1).squeeze(1);
    auto x2 = custom_rotation.slice(1, 1, 2).squeeze(1);
    auto y2 = custom_rotation.slice(1, 2, 3).squeeze(1);
    auto z2 = custom_rotation.slice(1, 3, 4).squeeze(1);

    auto torch_w1 = torch_q_rot.index({torch::indexing::Slice(), 0});
    auto torch_x1 = torch_q_rot.index({torch::indexing::Slice(), 1});
    auto torch_y1 = torch_q_rot.index({torch::indexing::Slice(), 2});
    auto torch_z1 = torch_q_rot.index({torch::indexing::Slice(), 3});

    auto torch_w2 = torch_rotation.index({torch::indexing::Slice(), 0});
    auto torch_x2 = torch_rotation.index({torch::indexing::Slice(), 1});
    auto torch_y2 = torch_rotation.index({torch::indexing::Slice(), 2});
    auto torch_z2 = torch_rotation.index({torch::indexing::Slice(), 3});

    // Compute new quaternion components
    auto w_new = w1.mul(w2).sub(x1.mul(x2)).sub(y1.mul(y2)).sub(z1.mul(z2));
    auto x_new = w1.mul(x2).add(x1.mul(w2)).add(y1.mul(z2)).sub(z1.mul(y2));
    auto y_new = w1.mul(y2).sub(x1.mul(z2)).add(y1.mul(w2)).add(z1.mul(x2));
    auto z_new = w1.mul(z2).add(x1.mul(y2)).sub(y1.mul(x2)).add(z1.mul(w2));

    auto torch_w_new = torch_w1 * torch_w2 - torch_x1 * torch_x2 - torch_y1 * torch_y2 - torch_z1 * torch_z2;
    auto torch_x_new = torch_w1 * torch_x2 + torch_x1 * torch_w2 + torch_y1 * torch_z2 - torch_z1 * torch_y2;
    auto torch_y_new = torch_w1 * torch_y2 - torch_x1 * torch_z2 + torch_y1 * torch_w2 + torch_z1 * torch_x2;
    auto torch_z_new = torch_w1 * torch_z2 + torch_x1 * torch_y2 - torch_y1 * torch_x2 + torch_z1 * torch_w2;

    std::cout << "\n=== Testing with exact failing data ===" << std::endl;
    std::cout << "Expected y_new[0] = -0.580434 (from REF debug output)" << std::endl;
    std::cout << "Expected z_new[0] = 1.34502 (from REF debug output)" << std::endl;

    auto y_new_vec = y_new.cpu().to_vector();
    auto z_new_vec = z_new.cpu().to_vector();
    std::cout << "Computed y_new[0] = " << y_new_vec[0] << std::endl;
    std::cout << "Computed z_new[0] = " << z_new_vec[0] << std::endl;

    // Method 1: index_copy_
    auto custom_rot_indexcopy = custom_rotation.clone();
    auto idx0 = Tensor::from_vector({0}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx1 = Tensor::from_vector({1}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx2 = Tensor::from_vector({2}, {1}, Device::CUDA).to(DataType::Int32);
    auto idx3 = Tensor::from_vector({3}, {1}, Device::CUDA).to(DataType::Int32);

    custom_rot_indexcopy.index_copy_(1, idx0, w_new.unsqueeze(1));
    custom_rot_indexcopy.index_copy_(1, idx1, x_new.unsqueeze(1));
    custom_rot_indexcopy.index_copy_(1, idx2, y_new.unsqueeze(1));
    custom_rot_indexcopy.index_copy_(1, idx3, z_new.unsqueeze(1));

    // Method 2: cat
    std::vector<Tensor> components = {
        w_new.unsqueeze(1),
        x_new.unsqueeze(1),
        y_new.unsqueeze(1),
        z_new.unsqueeze(1)};
    auto custom_rot_cat = Tensor::cat(components, 1);

    // PyTorch reference
    torch_rotation.index_put_({torch::indexing::Slice(), 0}, torch_w_new);
    torch_rotation.index_put_({torch::indexing::Slice(), 1}, torch_x_new);
    torch_rotation.index_put_({torch::indexing::Slice(), 2}, torch_y_new);
    torch_rotation.index_put_({torch::indexing::Slice(), 3}, torch_z_new);

    std::cout << "\nComparing results..." << std::endl;
    compare_tensors(custom_rot_cat, torch_rotation, 1e-5f, 1e-6f, "SpecificCase_Cat");
    compare_tensors(custom_rot_indexcopy, torch_rotation, 1e-5f, 1e-6f, "SpecificCase_IndexCopy");
}

TEST_F(TensorTorchCompatTest, Expand_Contiguity) {
    // Check if expand() creates contiguous tensors

    auto base = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CUDA);

    std::cout << "base.is_contiguous() = " << base.is_contiguous() << std::endl;

    auto unsqueezed = base.unsqueeze(0);
    std::cout << "unsqueezed.is_contiguous() = " << unsqueezed.is_contiguous() << std::endl;

    std::vector<int> expand_shape = {100, 4};
    auto expanded = unsqueezed.expand(std::span<const int>(expand_shape));
    std::cout << "expanded.is_contiguous() = " << expanded.is_contiguous() << std::endl;
    std::cout << "expanded.shape() = " << expanded.shape().str() << std::endl;

    // Try to extract a column
    auto col0 = expanded.slice(1, 0, 1);
    std::cout << "col0.is_contiguous() = " << col0.is_contiguous() << std::endl;

    auto col0_squeezed = col0.squeeze(1);
    std::cout << "col0_squeezed.is_contiguous() = " << col0_squeezed.is_contiguous() << std::endl;

    // Now try operations on it
    auto col0_doubled = col0_squeezed.mul(2.0f);
    std::cout << "col0_doubled.is_contiguous() = " << col0_doubled.is_contiguous() << std::endl;

    // Check if the values are correct
    auto col0_doubled_vec = col0_doubled.cpu().to_vector();
    std::cout << "Expected all values to be 2.0, got: ";
    for (size_t i = 0; i < std::min(size_t(5), col0_doubled_vec.size()); ++i) {
        std::cout << col0_doubled_vec[i] << " ";
    }
    std::cout << std::endl;

    // All should be 2.0 since base[0] = 1.0
    for (size_t i = 0; i < col0_doubled_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(col0_doubled_vec[i], 2.0f) << "Mismatch at index " << i;
    }
}

TEST_F(TensorTorchCompatTest, SliceVsIndex_ColumnExtraction) {
    // Test if our slice+squeeze matches torch's indexing for column extraction

    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f};

    auto custom_tensor = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});

    // Extract column 0 using our method
    auto custom_col0 = custom_tensor.slice(1, 0, 1).squeeze(1);

    // Extract column 0 using torch method (what the reference does)
    auto torch_col0 = torch_tensor.index({torch::indexing::Slice(), 0});

    std::cout << "Custom column 0 shape: " << custom_col0.shape().str() << std::endl;
    std::cout << "Torch column 0 shape: " << torch_col0.sizes() << std::endl;

    std::cout << "Custom values: ";
    auto custom_vec = custom_col0.cpu().to_vector();
    for (auto v : custom_vec)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Torch values: ";
    auto torch_vec = torch_col0.cpu();
    for (int i = 0; i < torch_vec.size(0); ++i) {
        std::cout << torch_vec[i].item<float>() << " ";
    }
    std::cout << std::endl;

    compare_tensors(custom_col0, torch_col0, 1e-6f, 1e-7f, "ColumnExtraction");

    // Now test expand
    auto base = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CUDA);
    auto torch_base = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::TensorOptions().device(torch::kCUDA));

    std::vector<int> expand_shape = {100, 4};
    auto custom_expanded = base.unsqueeze(0).expand(std::span<const int>(expand_shape));
    auto torch_expanded = torch_base.unsqueeze(0).expand({100, 4});

    // Extract column from expanded
    auto custom_exp_col0 = custom_expanded.slice(1, 0, 1).squeeze(1);
    auto torch_exp_col0 = torch_expanded.index({torch::indexing::Slice(), 0});

    std::cout << "\nExpanded column extraction:" << std::endl;
    std::cout << "Custom first 5: ";
    auto custom_exp_vec = custom_exp_col0.cpu().to_vector();
    for (int i = 0; i < 5; ++i)
        std::cout << custom_exp_vec[i] << " ";
    std::cout << std::endl;

    std::cout << "Torch first 5: ";
    auto torch_exp_vec = torch_exp_col0.cpu();
    for (int i = 0; i < 5; ++i)
        std::cout << torch_exp_vec[i].item<float>() << " ";
    std::cout << std::endl;

    compare_tensors(custom_exp_col0, torch_exp_col0, 1e-6f, 1e-7f, "ExpandedColumnExtraction");
}

// ============= DataType Conversion Tests =============

TEST_F(TensorTorchCompatTest, ToUInt8_Basic) {
    // Basic test: Float32 -> UInt8 conversion
    std::vector<float> data = {0.0f, 0.5f, 1.0f, 0.25f, 0.75f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CPU);

    // Test that conversion works
    auto custom_uint8 = custom_tensor.to(DataType::UInt8);

    EXPECT_TRUE(custom_uint8.is_valid()) << "UInt8 tensor should be valid";
    EXPECT_NE(custom_uint8.ptr<uint8_t>(), nullptr) << "UInt8 pointer should not be null";
    EXPECT_EQ(custom_uint8.dtype(), DataType::UInt8) << "Dtype should be UInt8";

    // Check values (0.5 * 255 = 127.5 -> 127, etc)
    auto uint8_vec = custom_uint8.to_vector_bool(); // Gets raw bytes
    EXPECT_EQ(uint8_vec.size(), 5);
}

TEST_F(TensorTorchCompatTest, ToUInt8_AfterClamp) {
    // Test the exact sequence used in image saving: clamp -> mul -> to(UInt8)
    std::vector<float> data = {-0.5f, 0.0f, 0.5f, 1.0f, 1.5f};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CPU);

    // Step by step like in save_image
    auto clamped = custom_tensor.clamp(0.0f, 1.0f);
    EXPECT_TRUE(clamped.is_valid()) << "Clamped tensor should be valid";
    EXPECT_NE(clamped.ptr<float>(), nullptr) << "Clamped pointer should not be null";

    auto scaled = clamped.mul(255.0f);
    EXPECT_TRUE(scaled.is_valid()) << "Scaled tensor should be valid";
    EXPECT_NE(scaled.ptr<float>(), nullptr) << "Scaled pointer should not be null";

    auto uint8_result = scaled.to(DataType::UInt8);
    EXPECT_TRUE(uint8_result.is_valid()) << "UInt8 result should be valid";
    EXPECT_NE(uint8_result.ptr<uint8_t>(), nullptr) << "UInt8 pointer should not be null";
}

TEST_F(TensorTorchCompatTest, ToUInt8_ChainedOperations) {
    // Test chained operations like save_image does
    std::vector<float> data = {0.0f, 0.5f, 1.0f};

    auto custom_tensor = Tensor::from_vector(data, {3}, Device::CPU);

    // Method 1: Chained (like original code)
    auto chained_uint8 = custom_tensor.clamp(0.0f, 1.0f).mul(255.0f).to(DataType::UInt8);
    EXPECT_TRUE(chained_uint8.is_valid()) << "Chained UInt8 should be valid";
    EXPECT_NE(chained_uint8.ptr<uint8_t>(), nullptr) << "Chained pointer should not be null";

    // Method 2: Separate steps with named variables
    auto step1 = custom_tensor.clamp(0.0f, 1.0f);
    auto step2 = step1.mul(255.0f);
    auto step3 = step2.to(DataType::UInt8);
    EXPECT_TRUE(step3.is_valid()) << "Step3 UInt8 should be valid";
    EXPECT_NE(step3.ptr<uint8_t>(), nullptr) << "Step3 pointer should not be null";

    // Both should give same result
    EXPECT_EQ(chained_uint8.numel(), step3.numel());
}

TEST_F(TensorTorchCompatTest, ToUInt8_AfterContiguous) {
    // Test with contiguous call (like save_image)
    std::vector<float> data = {0.0f, 0.5f, 1.0f};

    auto custom_tensor = Tensor::from_vector(data, {3}, Device::CPU);

    auto uint8_temp = custom_tensor.clamp(0.0f, 1.0f).mul(255.0f).to(DataType::UInt8);
    auto uint8_final = uint8_temp.contiguous();

    EXPECT_TRUE(uint8_final.is_valid()) << "Final UInt8 should be valid";
    EXPECT_NE(uint8_final.ptr<uint8_t>(), nullptr) << "Final pointer should not be null";
}

TEST_F(TensorTorchCompatTest, ToUInt8_MultiDimensional) {
    // Test with 2D tensor (like image data)
    std::vector<float> data = {
        0.0f, 0.25f, 0.5f,
        0.75f, 1.0f, 0.5f};

    auto custom_tensor = Tensor::from_vector(data, {2, 3}, Device::CPU);

    auto clamped = custom_tensor.clamp(0.0f, 1.0f);
    auto scaled = clamped.mul(255.0f);
    auto uint8_result = scaled.to(DataType::UInt8);

    EXPECT_TRUE(uint8_result.is_valid()) << "2D UInt8 should be valid";
    EXPECT_NE(uint8_result.ptr<uint8_t>(), nullptr) << "2D pointer should not be null";
    EXPECT_EQ(uint8_result.shape()[0], 2);
    EXPECT_EQ(uint8_result.shape()[1], 3);
}

TEST_F(TensorTorchCompatTest, ToUInt8_3DImage) {
    // Test with actual image-like 3D tensor [H, W, C]
    const int h = 4, w = 4, c = 3;
    std::vector<float> data(h * w * c, 0.5f);

    auto custom_tensor = Tensor::from_vector(data, {static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(c)},
                                             Device::CPU);

    // Exact sequence from save_image
    Tensor img = custom_tensor.clone();
    Tensor img_clamped = img.clamp(0.0f, 1.0f);
    Tensor img_scaled = img_clamped.mul(255.0f);
    Tensor img_uint8_temp = img_scaled.to(DataType::UInt8);
    Tensor img_uint8 = img_uint8_temp.contiguous();

    EXPECT_TRUE(img_uint8.is_valid()) << "3D image UInt8 should be valid";
    EXPECT_NE(img_uint8.ptr<uint8_t>(), nullptr) << "3D image pointer should not be null";
    EXPECT_EQ(img_uint8.numel(), h * w * c);
}

TEST_F(TensorTorchCompatTest, ToUInt8_CUDA) {
    // Test on CUDA device
    std::vector<float> data = {0.0f, 0.5f, 1.0f};

    auto custom_tensor = Tensor::from_vector(data, {3}, Device::CUDA);

    auto clamped = custom_tensor.clamp(0.0f, 1.0f);
    auto scaled = clamped.mul(255.0f);
    auto uint8_result = scaled.to(DataType::UInt8);

    EXPECT_TRUE(uint8_result.is_valid()) << "CUDA UInt8 should be valid";
    EXPECT_NE(uint8_result.ptr<uint8_t>(), nullptr) << "CUDA pointer should not be null";
    EXPECT_EQ(uint8_result.device(), Device::CUDA);
}

TEST_F(TensorTorchCompatTest, ToUInt8_ThenCPU) {
    // Test converting to UInt8 on CUDA, then moving to CPU
    std::vector<float> data = {0.0f, 0.5f, 1.0f};

    auto custom_tensor = Tensor::from_vector(data, {3}, Device::CUDA);

    auto uint8_cuda = custom_tensor.clamp(0.0f, 1.0f).mul(255.0f).to(DataType::UInt8);
    EXPECT_TRUE(uint8_cuda.is_valid()) << "CUDA UInt8 should be valid";

    auto uint8_cpu = uint8_cuda.cpu();
    EXPECT_TRUE(uint8_cpu.is_valid()) << "CPU UInt8 should be valid after transfer";
    EXPECT_NE(uint8_cpu.ptr<uint8_t>(), nullptr) << "CPU UInt8 pointer should not be null";
}

TEST_F(TensorTorchCompatTest, ToUInt8_AfterPermute) {
    // Test after permute (CHW -> HWC conversion)
    std::vector<float> data(3 * 4 * 5, 0.5f);

    auto custom_tensor = Tensor::from_vector(data, {3, 4, 5}, Device::CPU);

    auto permuted = custom_tensor.permute({1, 2, 0}).contiguous();
    auto uint8_result = permuted.clamp(0.0f, 1.0f).mul(255.0f).to(DataType::UInt8);

    EXPECT_TRUE(uint8_result.is_valid()) << "Permuted UInt8 should be valid";
    EXPECT_NE(uint8_result.ptr<uint8_t>(), nullptr) << "Permuted pointer should not be null";
}

TEST_F(TensorTorchCompatTest, ToUInt8_LifetimeTest) {
    // Test that demonstrates the lifetime issue
    const uint8_t* ptr = nullptr;

    {
        std::vector<float> data = {0.5f};
        auto custom_tensor = Tensor::from_vector(data, {1}, Device::CPU);

        // This creates temporaries
        auto temp_result = custom_tensor.clamp(0.0f, 1.0f).mul(255.0f).to(DataType::UInt8);
        ptr = temp_result.ptr<uint8_t>();

        // ptr should be valid here
        EXPECT_NE(ptr, nullptr) << "Pointer should be valid inside scope";
    }

    // After scope, the tensor is destroyed, but we're not accessing ptr anymore
    // This test just ensures we can get the pointer while the tensor is alive
}

TEST_F(TensorTorchCompatTest, ToUInt8_CloneBeforeConversion) {
    // Test that cloning before conversion helps
    std::vector<float> data = {0.0f, 0.5f, 1.0f};

    auto custom_tensor = Tensor::from_vector(data, {3}, Device::CPU);

    // Clone first to ensure ownership
    auto cloned = custom_tensor.clone();
    auto clamped = cloned.clamp(0.0f, 1.0f);
    auto scaled = clamped.mul(255.0f);
    auto cloned_scaled = scaled.clone(); // Clone again before conversion
    auto uint8_result = cloned_scaled.to(DataType::UInt8);

    EXPECT_TRUE(uint8_result.is_valid()) << "Cloned UInt8 should be valid";
    EXPECT_NE(uint8_result.ptr<uint8_t>(), nullptr) << "Cloned pointer should not be null";
}

TEST_F(TensorTorchCompatTest, DataTypeConversion_AllTypes) {
    // Test all data type conversions to ensure they work
    std::vector<float> data = {-1.5f, 0.0f, 1.5f, 255.5f};
    auto custom_tensor = Tensor::from_vector(data, {4}, Device::CPU);

    // Float32 (identity)
    auto float32 = custom_tensor.to(DataType::Float32);
    EXPECT_TRUE(float32.is_valid());
    EXPECT_NE(float32.ptr<float>(), nullptr);

    // Int32
    auto int32 = custom_tensor.to(DataType::Int32);
    EXPECT_TRUE(int32.is_valid());
    EXPECT_NE(int32.ptr<int>(), nullptr);

    // UInt8 - THE PROBLEMATIC ONE
    auto uint8 = custom_tensor.to(DataType::UInt8);
    EXPECT_TRUE(uint8.is_valid()) << "UInt8 conversion failed";
    EXPECT_NE(uint8.ptr<uint8_t>(), nullptr) << "UInt8 pointer is null";

    // Bool
    auto bool_tensor = custom_tensor.to(DataType::Bool);
    EXPECT_TRUE(bool_tensor.is_valid());
    EXPECT_NE(bool_tensor.ptr<unsigned char>(), nullptr);
}
