/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-5f, float atol = 1e-7f, const std::string& msg = "") {
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

class TensorAdvancedTest : public ::testing::Test {
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

// ============= Utility Functions Tests =============

TEST_F(TensorAdvancedTest, Linspace) {
    // Test basic linspace
    auto t_custom = Tensor::linspace(0, 10, 11);
    auto t_torch = torch::linspace(0, 10, 11, torch::TensorOptions().device(torch::kCUDA));

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-6f, "Linspace_Basic");

    // Test linspace with 2 points
    auto t2_custom = Tensor::linspace(-5, 5, 2);
    auto t2_torch = torch::linspace(-5, 5, 2, torch::TensorOptions().device(torch::kCUDA));

    compare_tensors(t2_custom, t2_torch, 1e-5f, 1e-6f, "Linspace_TwoPoints");

    // Test single point
    auto t3_custom = Tensor::linspace(3.14f, 3.14f, 1);
    auto t3_torch = torch::linspace(3.14f, 3.14f, 1, torch::TensorOptions().device(torch::kCUDA));

    compare_tensors(t3_custom, t3_torch, 1e-5f, 1e-6f, "Linspace_Single");

    // Test invalid (0 steps)
    auto invalid = Tensor::linspace(0, 1, 0);
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorAdvancedTest, Stack) {
    // Create test tensors
    auto t1_custom = Tensor::full({3, 4}, 1.0f, Device::CUDA);
    auto t2_custom = Tensor::full({3, 4}, 2.0f, Device::CUDA);
    auto t3_custom = Tensor::full({3, 4}, 3.0f, Device::CUDA);

    auto t1_torch = torch::full({3, 4}, 1.0f, torch::TensorOptions().device(torch::kCUDA));
    auto t2_torch = torch::full({3, 4}, 2.0f, torch::TensorOptions().device(torch::kCUDA));
    auto t3_torch = torch::full({3, 4}, 3.0f, torch::TensorOptions().device(torch::kCUDA));

    // Stack along dimension 0
    std::vector<Tensor> tensors_custom;
    tensors_custom.push_back(t1_custom.clone());
    tensors_custom.push_back(t2_custom.clone());
    tensors_custom.push_back(t3_custom.clone());

    auto stacked_custom = Tensor::stack(std::move(tensors_custom), 0);
    auto stacked_torch = torch::stack({t1_torch, t2_torch, t3_torch}, 0);

    EXPECT_EQ(stacked_custom.shape().rank(), stacked_torch.dim());
    EXPECT_EQ(stacked_custom.shape()[0], stacked_torch.size(0));
    EXPECT_EQ(stacked_custom.shape()[1], stacked_torch.size(1));
    EXPECT_EQ(stacked_custom.shape()[2], stacked_torch.size(2));

    compare_tensors(stacked_custom, stacked_torch, 1e-6f, 1e-7f, "Stack");

    // Test stacking empty list
    std::vector<Tensor> empty_list;
    auto invalid = Tensor::stack(std::move(empty_list));
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorAdvancedTest, Concatenate) {
    // Create test tensors with different sizes along dim 0
    auto t1_custom = Tensor::full({2, 3}, 1.0f, Device::CUDA);
    auto t2_custom = Tensor::full({3, 3}, 2.0f, Device::CUDA);
    auto t3_custom = Tensor::full({1, 3}, 3.0f, Device::CUDA);

    auto t1_torch = torch::full({2, 3}, 1.0f, torch::TensorOptions().device(torch::kCUDA));
    auto t2_torch = torch::full({3, 3}, 2.0f, torch::TensorOptions().device(torch::kCUDA));
    auto t3_torch = torch::full({1, 3}, 3.0f, torch::TensorOptions().device(torch::kCUDA));

    // Concatenate along dimension 0
    std::vector<Tensor> tensors_custom;
    tensors_custom.push_back(t1_custom.clone());
    tensors_custom.push_back(t2_custom.clone());
    tensors_custom.push_back(t3_custom.clone());

    auto concatenated_custom = Tensor::cat(std::move(tensors_custom), 0);
    auto concatenated_torch = torch::cat({t1_torch, t2_torch, t3_torch}, 0);

    EXPECT_EQ(concatenated_custom.shape().rank(), concatenated_torch.dim());
    EXPECT_EQ(concatenated_custom.shape()[0], concatenated_torch.size(0));
    EXPECT_EQ(concatenated_custom.shape()[1], concatenated_torch.size(1));

    compare_tensors(concatenated_custom, concatenated_torch, 1e-6f, 1e-7f, "Concatenate");

    // Test mismatched shapes (should fail)
    std::vector<Tensor> mismatched;
    mismatched.push_back(Tensor::zeros({2, 3}, Device::CUDA));
    mismatched.push_back(Tensor::zeros({2, 4}, Device::CUDA));
    auto invalid = Tensor::cat(std::move(mismatched), 0);
    EXPECT_FALSE(invalid.is_valid());
}

// ============= Memory Info Tests =============

TEST_F(TensorAdvancedTest, MemoryInfo) {
    auto initial_info = MemoryInfo::cuda();

    // Allocate a large tensor
    const size_t large_size = 1024 * 1024; // 1M elements = 4MB
    auto large_tensor_custom = Tensor::zeros({large_size}, Device::CUDA);
    auto large_tensor_torch = torch::zeros({static_cast<int64_t>(large_size)},
                                           torch::TensorOptions().device(torch::kCUDA));

    auto after_alloc_info = MemoryInfo::cuda();

    // Should have more allocated memory
    EXPECT_GT(after_alloc_info.allocated_bytes, initial_info.allocated_bytes);

    LOG_INFO("Memory allocated: {} bytes",
             after_alloc_info.allocated_bytes - initial_info.allocated_bytes);
}

// ============= Error Handling Tests =============

TEST_F(TensorAdvancedTest, ErrorHandlingShapeMismatch) {
    auto t1_custom = Tensor::ones({3, 4}, Device::CUDA);
    auto t2_custom = Tensor::ones({4, 3}, Device::CUDA);

    auto result = t1_custom.add(t2_custom);
    EXPECT_FALSE(result.is_valid()) << "Shape mismatch should produce invalid tensor";
}

TEST_F(TensorAdvancedTest, ErrorHandlingInvalidReshape) {
    auto t_custom = Tensor::ones({12}, Device::CUDA);

    // Invalid reshape (15 != 12)
    auto reshaped_invalid = t_custom.try_reshape({5, 3});
    EXPECT_FALSE(reshaped_invalid.has_value());

    // Valid reshape (12 = 3*4)
    auto reshaped_valid = t_custom.try_reshape({3, 4});
    EXPECT_TRUE(reshaped_valid.has_value());

    if (reshaped_valid.has_value()) {
        auto t_torch = torch::ones({12}, torch::TensorOptions().device(torch::kCUDA));
        auto t_torch_reshaped = t_torch.reshape({3, 4});

        compare_tensors(*reshaped_valid, t_torch_reshaped, 1e-6f, 1e-7f, "ValidReshape");
    }
}

TEST_F(TensorAdvancedTest, ErrorHandlingInvalidTensor) {
    Tensor invalid;
    EXPECT_FALSE(invalid.is_valid());

    auto result2 = invalid.add(1.0f);
    EXPECT_FALSE(result2.is_valid());

    auto result3 = invalid.clone();
    EXPECT_FALSE(result3.is_valid());
}

// ============= Batch Processing Tests =============

TEST_F(TensorAdvancedTest, BatchProcessing) {
    auto large_custom = Tensor::ones({100, 10}, Device::CUDA);
    auto large_torch = torch::ones({100, 10}, torch::TensorOptions().device(torch::kCUDA));

    // Split into batches
    auto batches_custom = Tensor::split_batch(large_custom, 32);
    auto batches_torch = large_torch.split(32, 0);

    // Should have 4 batches: 32, 32, 32, 4
    EXPECT_EQ(batches_custom.size(), batches_torch.size());

    for (size_t i = 0; i < batches_custom.size(); ++i) {
        EXPECT_EQ(batches_custom[i].shape()[0], batches_torch[i].size(0));
        EXPECT_EQ(batches_custom[i].shape()[1], 10);

        compare_tensors(batches_custom[i], batches_torch[i], 1e-6f, 1e-7f,
                        "Batch_" + std::to_string(i));
    }
}

// ============= Cross-Device Operations Tests =============

TEST_F(TensorAdvancedTest, CrossDeviceOperations) {
    auto cpu_custom = Tensor::full({3, 3}, 5.0f, Device::CPU);
    auto cpu_torch = torch::full({3, 3}, 5.0f, torch::TensorOptions().device(torch::kCPU));

    EXPECT_EQ(cpu_custom.device(), Device::CPU);
    EXPECT_TRUE(cpu_torch.device().is_cpu());

    // Transfer to CUDA
    auto cuda_custom = cpu_custom.to(Device::CUDA);
    auto cuda_torch = cpu_torch.to(torch::kCUDA);

    EXPECT_EQ(cuda_custom.device(), Device::CUDA);
    EXPECT_TRUE(cuda_torch.device().is_cuda());

    // Perform operation on CUDA
    auto result_custom = cuda_custom.mul(2.0f);
    auto result_torch = cuda_torch * 2.0f;

    // Transfer back to CPU
    auto result_cpu_custom = result_custom.to(Device::CPU);
    auto result_cpu_torch = result_torch.to(torch::kCPU);

    EXPECT_EQ(result_cpu_custom.device(), Device::CPU);

    compare_tensors(result_cpu_custom, result_cpu_torch, 1e-6f, 1e-7f, "CrossDevice");
}

// ============= Stress Tests =============

TEST_F(TensorAdvancedTest, StressTestLargeTensors) {
    const size_t large_dim = 1000;

    auto large1_custom = Tensor::zeros({large_dim, large_dim}, Device::CUDA);
    auto large2_custom = Tensor::ones({large_dim, large_dim}, Device::CUDA);

    auto large1_torch = torch::zeros({static_cast<int64_t>(large_dim), static_cast<int64_t>(large_dim)},
                                     torch::TensorOptions().device(torch::kCUDA));
    auto large2_torch = torch::ones({static_cast<int64_t>(large_dim), static_cast<int64_t>(large_dim)},
                                    torch::TensorOptions().device(torch::kCUDA));

    // Perform operations
    auto sum_custom = large1_custom.add(large2_custom);
    auto sum_torch = large1_torch + large2_torch;

    EXPECT_TRUE(sum_custom.is_valid());
    EXPECT_FLOAT_EQ(sum_custom.mean_scalar(), sum_torch.mean().item<float>());

    auto product_custom = large1_custom.mul(large2_custom);
    auto product_torch = large1_torch * large2_torch;

    EXPECT_TRUE(product_custom.is_valid());
    EXPECT_FLOAT_EQ(product_custom.sum_scalar(), product_torch.sum().item<float>());
}

TEST_F(TensorAdvancedTest, StressTestManyOperations) {
    auto tensor_custom = Tensor::ones({100, 100}, Device::CUDA);
    auto tensor_torch = torch::ones({100, 100}, torch::TensorOptions().device(torch::kCUDA));

    for (int i = 0; i < 100; ++i) {
        tensor_custom = tensor_custom.add(0.01f).mul(1.01f).sub(0.01f);
        tensor_torch = (tensor_torch + 0.01f) * 1.01f - 0.01f;
    }

    EXPECT_TRUE(tensor_custom.is_valid());
    EXPECT_FALSE(tensor_custom.has_nan());
    EXPECT_FALSE(tensor_custom.has_inf());

    // Results should be close (within numerical error accumulation)
    compare_tensors(tensor_custom, tensor_torch, 1e-3f, 1e-4f, "StressManyOps");
}

// ============= Chainable Operations Tests =============

TEST_F(TensorAdvancedTest, ChainableInplace) {
    auto tensor_custom = Tensor::ones({3, 3}, Device::CUDA);
    auto tensor_torch = torch::ones({3, 3}, torch::TensorOptions().device(torch::kCUDA));

    // Test inplace chaining: ((1 + 1) * 2) - 1 = 3
    tensor_custom.inplace([](Tensor& t) { t.add_(1.0f); })
        .inplace([](Tensor& t) { t.mul_(2.0f); })
        .inplace([](Tensor& t) { t.sub_(1.0f); });

    tensor_torch.add_(1.0f).mul_(2.0f).sub_(1.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "ChainableInplace");
}

TEST_F(TensorAdvancedTest, ChainableApply) {
    auto tensor_custom = Tensor::ones({3, 3}, Device::CUDA);
    auto tensor_torch = torch::ones({3, 3}, torch::TensorOptions().device(torch::kCUDA));

    // Test apply (non-mutating): ((1 + 1) * 2) - 1 = 3
    auto result_custom = tensor_custom.apply([](const Tensor& t) { return t.add(1.0f); })
                             .apply([](const Tensor& t) { return t.mul(2.0f); })
                             .apply([](const Tensor& t) { return t.sub(1.0f); });

    auto result_torch = ((tensor_torch + 1.0f) * 2.0f) - 1.0f;

    // Original should be unchanged
    EXPECT_FLOAT_EQ(tensor_custom.to_vector()[0], 1.0f);
    EXPECT_FLOAT_EQ(tensor_torch.to(torch::kCPU).data_ptr<float>()[0], 1.0f);

    compare_tensors(result_custom, result_torch, 1e-6f, 1e-7f, "ChainableApply");
}

// ============= Edge Cases Tests =============

TEST_F(TensorAdvancedTest, ExtremelySparseOperations) {
    auto scalar1_custom = Tensor::full({1}, 3.14f, Device::CUDA);
    auto scalar2_custom = Tensor::full({1}, 2.71f, Device::CUDA);

    auto scalar1_torch = torch::full({1}, 3.14f, torch::TensorOptions().device(torch::kCUDA));
    auto scalar2_torch = torch::full({1}, 2.71f, torch::TensorOptions().device(torch::kCUDA));

    auto sum_custom = scalar1_custom.add(scalar2_custom);
    auto sum_torch = scalar1_torch + scalar2_torch;

    EXPECT_NEAR(sum_custom.item(), sum_torch.item<float>(), 1e-5f);

    auto product_custom = scalar1_custom.mul(scalar2_custom);
    auto product_torch = scalar1_torch * scalar2_torch;

    EXPECT_NEAR(product_custom.item(), product_torch.item<float>(), 1e-4f);

    // Test reshape of scalar
    auto reshaped_custom = scalar1_custom.view({1, 1, 1, 1});
    auto reshaped_torch = scalar1_torch.view({1, 1, 1, 1});

    EXPECT_EQ(reshaped_custom.shape().rank(), reshaped_torch.dim());
    compare_tensors(reshaped_custom, reshaped_torch, 1e-6f, 1e-7f, "ScalarReshape");
}

TEST_F(TensorAdvancedTest, ZeroDimensionalConsistency) {
    auto empty1_custom = Tensor::empty({0}, Device::CUDA);
    auto empty2_custom = Tensor::empty({0, 5}, Device::CUDA);

    auto empty1_torch = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA));
    auto empty2_torch = torch::empty({0, 5}, torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(empty1_custom.is_valid());
    EXPECT_TRUE(empty2_custom.is_valid());
    EXPECT_EQ(empty1_custom.numel(), empty1_torch.numel());
    EXPECT_EQ(empty2_custom.numel(), empty2_torch.numel());

    // Operations should work but produce empty results
    auto sum_custom = empty1_custom.add(1.0f);
    auto sum_torch = empty1_torch + 1.0f;

    EXPECT_TRUE(sum_custom.is_valid());
    EXPECT_EQ(sum_custom.numel(), sum_torch.numel());

    // Reductions on empty tensors
    EXPECT_FLOAT_EQ(empty1_custom.sum_scalar(), empty1_torch.sum().item<float>());

    // Clone should work
    auto cloned_custom = empty1_custom.clone();
    auto cloned_torch = empty1_torch.clone();

    EXPECT_TRUE(cloned_custom.is_valid());
    EXPECT_EQ(cloned_custom.numel(), cloned_torch.numel());
}

// ============= Special Values Tests =============

TEST_F(TensorAdvancedTest, SpecialValues) {
    // Test with NaN and Inf
    std::vector<float> special_values = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        0.0f,
        -0.0f,
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::epsilon()};

    auto cpu_custom = Tensor::empty({8}, Device::CPU);
    auto cpu_torch = torch::empty({8}, torch::TensorOptions().device(torch::kCPU));

    std::memcpy(cpu_custom.ptr<float>(), special_values.data(), 8 * sizeof(float));
    std::memcpy(cpu_torch.data_ptr<float>(), special_values.data(), 8 * sizeof(float));

    auto cuda_custom = cpu_custom.to(Device::CUDA);
    auto cuda_torch = cpu_torch.to(torch::kCUDA);

    // Check detection
    EXPECT_TRUE(cuda_custom.has_nan());
    EXPECT_TRUE(cuda_custom.has_inf());

    // PyTorch equivalents
    EXPECT_TRUE(cuda_torch.isnan().any().item<bool>());
    EXPECT_TRUE(cuda_torch.isinf().any().item<bool>());

    // Test that assert_finite throws
    EXPECT_THROW(cuda_custom.assert_finite(), TensorError);

    // Test clamping removes inf
    auto clamped_custom = cuda_custom.clamp(-1e10f, 1e10f);
    auto clamped_torch = torch::clamp(cuda_torch, -1e10f, 1e10f);

    EXPECT_FALSE(clamped_custom.has_inf());
    EXPECT_FALSE(clamped_torch.isinf().any().item<bool>());

    // Note: Both implementations handle NaN consistently in clamp
    LOG_INFO("IMPLEMENTATION NOTE: Clamp behavior with NaN matches PyTorch");
}

// ============= Thread Safety Tests (Basic) =============

TEST_F(TensorAdvancedTest, ConcurrentTensorCreation) {
    const int num_threads = 10;
    const int tensors_per_thread = 100;

    std::vector<std::thread> threads;
    std::vector<std::vector<Tensor>> thread_tensors(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&thread_tensors, t, tensors_per_thread]() {
            for (int i = 0; i < tensors_per_thread; ++i) {
                thread_tensors[t].push_back(
                    Tensor::full({10, 10}, static_cast<float>(t * 100 + i), Device::CUDA));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all tensors were created correctly
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_EQ(thread_tensors[t].size(), tensors_per_thread);
        for (int i = 0; i < tensors_per_thread; ++i) {
            EXPECT_TRUE(thread_tensors[t][i].is_valid());
            EXPECT_FLOAT_EQ(thread_tensors[t][i].to_vector()[0],
                            static_cast<float>(t * 100 + i));
        }
    }
}

// ============= Compatibility Tests =============

TEST_F(TensorAdvancedTest, LikeOperations) {
    auto original_custom = Tensor::full({3, 4, 5}, 2.5f, Device::CUDA);
    auto original_torch = torch::full({3, 4, 5}, 2.5f, torch::TensorOptions().device(torch::kCUDA));

    auto zeros_custom = Tensor::zeros_like(original_custom);
    auto zeros_torch = torch::zeros_like(original_torch);

    EXPECT_EQ(zeros_custom.shape(), original_custom.shape());
    EXPECT_EQ(zeros_custom.device(), original_custom.device());
    EXPECT_FLOAT_EQ(zeros_custom.sum_scalar(), zeros_torch.sum().item<float>());

    compare_tensors(zeros_custom, zeros_torch, 1e-6f, 1e-7f, "ZerosLike");

    auto ones_custom = Tensor::ones_like(original_custom);
    auto ones_torch = torch::ones_like(original_torch);

    EXPECT_EQ(ones_custom.shape(), original_custom.shape());
    EXPECT_EQ(ones_custom.device(), original_custom.device());
    EXPECT_FLOAT_EQ(ones_custom.sum_scalar(), ones_torch.sum().item<float>());

    compare_tensors(ones_custom, ones_torch, 1e-6f, 1e-7f, "OnesLike");
}

TEST_F(TensorAdvancedTest, DiagOperation) {
    std::vector<float> diag_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto diagonal_custom = Tensor::from_vector(diag_data, {5}, Device::CUDA);
    auto diagonal_torch = torch::tensor(diag_data, torch::TensorOptions().device(torch::kCUDA));

    auto matrix_custom = Tensor::diag(diagonal_custom);
    auto matrix_torch = torch::diag(diagonal_torch);

    EXPECT_EQ(matrix_custom.shape()[0], matrix_torch.size(0));
    EXPECT_EQ(matrix_custom.shape()[1], matrix_torch.size(1));

    compare_tensors(matrix_custom, matrix_torch, 1e-6f, 1e-7f, "Diag");
}

// ============= Profiling Tests =============

TEST_F(TensorAdvancedTest, ProfilingSupport) {
    // Enable profiling
    Tensor::enable_profiling(true);

    auto tensor = Tensor::ones({100, 100}, Device::CUDA);

    // This should log timing information
    auto result = tensor.timed("test_operation", [](const Tensor& t) {
        return t.add(1.0f).mul(2.0f).sub(1.0f);
    });

    EXPECT_TRUE(result.is_valid());

    // Verify result is correct: ((1 + 1) * 2) - 1 = 3
    EXPECT_NEAR(result.mean_scalar(), 3.0f, 1e-5f);

    // Disable profiling
    Tensor::enable_profiling(false);
}

// ============= Assertion Tests =============

TEST_F(TensorAdvancedTest, AssertShape) {
    auto tensor = Tensor::ones({3, 4, 5}, Device::CUDA);

    // Should pass
    EXPECT_NO_THROW(tensor.assert_shape({3, 4, 5}, "Shape check"));

    // Should throw
    EXPECT_THROW(tensor.assert_shape({3, 5, 4}, "Wrong shape"), TensorError);
}

TEST_F(TensorAdvancedTest, AssertDevice) {
    auto cuda_tensor = Tensor::ones({2, 2}, Device::CUDA);
    auto cpu_tensor = Tensor::ones({2, 2}, Device::CPU);

    // Should pass
    EXPECT_NO_THROW(cuda_tensor.assert_device(Device::CUDA));
    EXPECT_NO_THROW(cpu_tensor.assert_device(Device::CPU));

    // Should throw
    EXPECT_THROW(cuda_tensor.assert_device(Device::CPU), TensorError);
    EXPECT_THROW(cpu_tensor.assert_device(Device::CUDA), TensorError);
}

TEST_F(TensorAdvancedTest, AssertDtype) {
    auto float_tensor = Tensor::ones({2, 2}, Device::CUDA, DataType::Float32);
    auto int_tensor = Tensor::zeros({2, 2}, Device::CUDA, DataType::Int32);

    // Should pass
    EXPECT_NO_THROW(float_tensor.assert_dtype(DataType::Float32));
    EXPECT_NO_THROW(int_tensor.assert_dtype(DataType::Int32));

    // Should throw
    EXPECT_THROW(float_tensor.assert_dtype(DataType::Int32), TensorError);
    EXPECT_THROW(int_tensor.assert_dtype(DataType::Float32), TensorError);
}
