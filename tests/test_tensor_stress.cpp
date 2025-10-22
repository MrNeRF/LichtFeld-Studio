/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Memory pool is now an internal implementation detail of tensor library
// #include "core/memory_pool.hpp"
#include "core_new/tensor.hpp"
#include <algorithm>
#include <c10/cuda/CUDACachingAllocator.h>
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
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

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

class TensorStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";

        // Record initial memory
        cudaMemGetInfo(&initial_free_mem_, &total_mem_);
        gen_.seed(42);

        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }

    void TearDown() override {
        // Force synchronization
        cudaDeviceSynchronize();

        // Clear PyTorch CUDA cache
        if (torch::cuda::is_available()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }

        // Note: Tensor library's memory pool is now an internal detail
        // Memory cleanup is handled automatically by the tensor library

        // Give the system a moment to actually free memory
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Check for memory leaks with reasonable tolerance
        size_t current_free_mem, current_total;
        cudaMemGetInfo(&current_free_mem, &current_total);

        // Allow 100MB tolerance for CUDA internal allocations and caching
        // Both our allocator and PyTorch's allocator cache freed memory
        size_t tolerance = 100 * 1024 * 1024;

        long long leaked = static_cast<long long>(initial_free_mem_) -
                           static_cast<long long>(current_free_mem);

        EXPECT_NEAR(static_cast<long long>(current_free_mem),
                    static_cast<long long>(initial_free_mem_),
                    static_cast<long long>(tolerance))
            << "Possible memory leak detected: "
            << leaked / (1024 * 1024) << " MB not freed";
    }

    size_t initial_free_mem_;
    size_t total_mem_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};
};

// ============= Memory Management Tests =============

TEST_F(TensorStressTest, MaxMemoryAllocation) {
    // Try to allocate tensors until we approach memory limit
    std::vector<Tensor> tensors;
    const size_t chunk_size = 10 * 1024 * 1024; // 10M floats = 40MB per tensor

    size_t total_allocated = 0;

    for (int i = 0; i < 50; ++i) { // Max 50 tensors (2GB) to avoid hanging
        // Check available memory before allocating
        size_t free_mem, total;
        cudaMemGetInfo(&free_mem, &total);
        if (free_mem < chunk_size * sizeof(float) * 3) {
            // Stop before we run too low
            break;
        }

        auto t = Tensor::empty({chunk_size}, Device::CUDA);
        if (!t.is_valid()) {
            break;
        }
        tensors.emplace_back(std::move(t));
        total_allocated += chunk_size * sizeof(float);
    }

    EXPECT_GT(tensors.size(), 0) << "Should be able to allocate at least one tensor";

    // Clear tensors to free memory
    tensors.clear();
    cudaDeviceSynchronize();

    // Clear PyTorch cache to actually release memory
    c10::cuda::CUDACachingAllocator::emptyCache();
    cudaDeviceSynchronize();

    // Memory should be freed
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    EXPECT_GT(free_after, initial_free_mem_ - 100 * 1024 * 1024)
        << "Memory not properly freed after clearing tensors";
}

TEST_F(TensorStressTest, RapidAllocationDeallocation) {
    // Rapidly allocate and deallocate tensors
    const int iterations = 1000;
    const size_t tensor_size = 1024 * 1024; // 1M elements

    for (int i = 0; i < iterations; ++i) {
        auto t = Tensor::zeros({tensor_size}, Device::CUDA);
        EXPECT_TRUE(t.is_valid());
        // Tensor destroyed here
    }

    cudaDeviceSynchronize();

    // Clear PyTorch cache to verify no leaks (caching is expected, leaks are not)
    c10::cuda::CUDACachingAllocator::emptyCache();
    cudaDeviceSynchronize();

    // Check memory is stable
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    EXPECT_NEAR(static_cast<long long>(free_after),
                static_cast<long long>(initial_free_mem_),
                50 * 1024 * 1024)
        << "Memory leak detected after rapid allocation/deallocation";
}

// ============= Numerical Correctness Tests =============

TEST_F(TensorStressTest, DeepOperationChain) {
    // Create a very deep chain of operations and compare with PyTorch
    auto custom_t = Tensor::ones({100, 100}, Device::CUDA);
    auto torch_t = torch::ones({100, 100}, torch::TensorOptions().device(torch::kCUDA));

    const int chain_length = 100; // Reduced for performance
    for (int i = 0; i < chain_length; ++i) {
        // Alternate operations to avoid overflow/underflow
        if (i % 4 == 0) {
            custom_t = custom_t.add(0.001f);
            torch_t = torch_t + 0.001f;
        } else if (i % 4 == 1) {
            custom_t = custom_t.mul(1.001f);
            torch_t = torch_t * 1.001f;
        } else if (i % 4 == 2) {
            custom_t = custom_t.sub(0.001f);
            torch_t = torch_t - 0.001f;
        } else {
            custom_t = custom_t.div(1.001f);
            torch_t = torch_t / 1.001f;
        }

        // Every 20 operations, check validity
        if (i % 20 == 19) {
            EXPECT_TRUE(custom_t.is_valid());
            EXPECT_FALSE(custom_t.has_nan());
            EXPECT_FALSE(custom_t.has_inf());
        }
    }

    // Compare final results
    compare_tensors(custom_t, torch_t, 1e-3f, 1e-4f, "DeepOperationChain");
}

TEST_F(TensorStressTest, LargeMatrixOperations) {
    const size_t dim = 1024; // 1024x1024 for reasonable test time

    std::vector<float> data_a(dim * dim);
    std::vector<float> data_b(dim * dim);

    for (auto& val : data_a)
        val = dist_(gen_);
    for (auto& val : data_b)
        val = dist_(gen_);

    auto custom_a = Tensor::from_vector(data_a, {dim, dim}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {dim, dim}, Device::CUDA);

    auto torch_a = torch::from_blob(data_a.data(), {static_cast<long>(dim), static_cast<long>(dim)},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .clone()
                       .to(torch::kCUDA);
    auto torch_b = torch::from_blob(data_b.data(), {static_cast<long>(dim), static_cast<long>(dim)},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .clone()
                       .to(torch::kCUDA);

    // Perform operations
    auto start = std::chrono::high_resolution_clock::now();

    auto custom_c = custom_a.add(custom_b);
    auto custom_d = custom_c.mul(2.0f);
    auto custom_e = custom_d.sub(custom_a);
    auto custom_result = custom_e.div(custom_b.add(1.0f));

    auto end = std::chrono::high_resolution_clock::now();
    auto custom_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // PyTorch operations
    start = std::chrono::high_resolution_clock::now();

    auto torch_c = torch_a + torch_b;
    auto torch_d = torch_c * 2.0f;
    auto torch_e = torch_d - torch_a;
    auto torch_result = torch_e / (torch_b + 1.0f);

    end = std::chrono::high_resolution_clock::now();
    auto torch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(custom_result.is_valid());
    EXPECT_FALSE(custom_result.has_nan());
    EXPECT_FALSE(custom_result.has_inf());

    // Compare results
    compare_tensors(custom_result, torch_result, 1e-3f, 1e-4f, "LargeMatrixOperations");

    std::cout << "Large matrix operations - Custom: " << custom_duration.count()
              << " ms, PyTorch: " << torch_duration.count() << " ms" << std::endl;
}

TEST_F(TensorStressTest, ManySmallTensors) {
    // Create many small tensors
    const int num_tensors = 1000; // Reduced for test speed
    std::vector<Tensor> custom_tensors;
    std::vector<torch::Tensor> torch_tensors;
    custom_tensors.reserve(num_tensors);
    torch_tensors.reserve(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
        custom_tensors.emplace_back(Tensor::full({10}, static_cast<float>(i), Device::CUDA));
        torch_tensors.push_back(torch::full({10}, static_cast<float>(i),
                                            torch::TensorOptions().device(torch::kCUDA)));
    }

    // Verify all tensors
    for (int i = 0; i < num_tensors; ++i) {
        EXPECT_TRUE(custom_tensors[i].is_valid());
        compare_tensors(custom_tensors[i], torch_tensors[i], 1e-6f, 1e-7f,
                        "SmallTensor" + std::to_string(i));
    }

    // Perform operation on all
    for (int i = 0; i < num_tensors; ++i) {
        custom_tensors[i] = custom_tensors[i].add(1.0f);
        torch_tensors[i] = torch_tensors[i] + 1.0f;
    }

    // Verify again
    for (int i = 0; i < num_tensors; ++i) {
        compare_tensors(custom_tensors[i], torch_tensors[i], 1e-6f, 1e-7f,
                        "SmallTensorAfterAdd" + std::to_string(i));
    }
}

// ============= View and Slice Tests =============

TEST_F(TensorStressTest, ViewStress) {
    // Create complex view hierarchies
    auto custom_original = Tensor::ones({24, 24}, Device::CUDA);
    auto torch_original = torch::ones({24, 24}, torch::TensorOptions().device(torch::kCUDA));

    std::vector<Tensor> custom_views;
    std::vector<torch::Tensor> torch_views;

    // Create many different views
    std::vector<std::vector<int64_t>> shapes = {
        {576},
        {4, 144},
        {8, 72},
        {16, 36},
        {2, 12, 24},
        {3, 8, 24},
        {4, 6, 24},
        {2, 2, 144}};

    for (const auto& shape : shapes) {
        custom_views.emplace_back(custom_original.view(std::vector<int>(shape.begin(), shape.end())));
        torch_views.push_back(torch_original.view(shape));
    }

    // Modify original
    custom_original.fill_(2.0f);
    torch_original.fill_(2.0f);

    // All views should see the change
    for (size_t i = 0; i < custom_views.size(); ++i) {
        compare_tensors(custom_views[i], torch_views[i], 1e-6f, 1e-7f,
                        "View" + std::to_string(i));
    }

    // Create views of views
    auto custom_view_of_view = custom_views[0].view({24, 24});
    auto torch_view_of_view = torch_views[0].view({24, 24});

    EXPECT_TRUE(custom_view_of_view.is_valid());
    compare_tensors(custom_view_of_view, torch_view_of_view, 1e-6f, 1e-7f, "ViewOfView");
}

TEST_F(TensorStressTest, SliceStress) {
    const size_t size = 200; // Reduced for test speed

    // Create tensor with row indices
    std::vector<float> data(size * size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            data[i * size + j] = static_cast<float>(i);
        }
    }

    auto custom_tensor = Tensor::from_vector(data, {size, size}, Device::CUDA);
    auto torch_tensor = torch::from_blob(data.data(),
                                         {static_cast<long>(size), static_cast<long>(size)},
                                         torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(torch::kCUDA);

    // Create many overlapping slices
    std::vector<Tensor> custom_slices;
    std::vector<torch::Tensor> torch_slices;

    for (size_t i = 0; i < size - 10; i += 10) {
        size_t end = std::min(i + 20, size);
        custom_slices.emplace_back(custom_tensor.slice(0, i, end));
        torch_slices.push_back(torch_tensor.slice(0, i, end));
    }

    // Verify slices
    for (size_t i = 0; i < custom_slices.size(); ++i) {
        EXPECT_EQ(custom_slices[i].shape()[0], static_cast<size_t>(torch_slices[i].size(0)));
        EXPECT_EQ(custom_slices[i].shape()[1], static_cast<size_t>(torch_slices[i].size(1)));

        compare_tensors(custom_slices[i], torch_slices[i], 1e-6f, 1e-7f,
                        "Slice" + std::to_string(i));
    }
}

// ============= Reduction Tests =============

TEST_F(TensorStressTest, ReductionStress) {
    // Test reductions on large tensors
    const size_t size = 100000; // 100K elements for faster test

    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i % 100); // Repeating pattern 0-99
    }

    auto custom_tensor = Tensor::from_vector(data, {size}, Device::CUDA);
    auto torch_tensor = torch::from_blob(data.data(), {static_cast<long>(size)},
                                         torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(torch::kCUDA);

    // Test reductions
    auto custom_sum = custom_tensor.sum();
    auto torch_sum = torch_tensor.sum();
    compare_tensors(custom_sum, torch_sum, 1e-3f, 1e-4f, "Sum");

    auto custom_mean = custom_tensor.mean();
    auto torch_mean = torch_tensor.mean();
    compare_tensors(custom_mean, torch_mean, 1e-4f, 1e-5f, "Mean");

    auto custom_min = custom_tensor.min();
    auto torch_min = torch_tensor.min();
    compare_tensors(custom_min, torch_min, 1e-6f, 1e-7f, "Min");

    auto custom_max = custom_tensor.max();
    auto torch_max = torch_tensor.max();
    compare_tensors(custom_max, torch_max, 1e-6f, 1e-7f, "Max");
}

TEST_F(TensorStressTest, MultiDimensionalReductions) {
    auto custom_t = Tensor::randn({10, 20, 30}, Device::CUDA);

    // Copy data to PyTorch
    auto data = custom_t.to_vector();
    auto torch_t = torch::from_blob(data.data(), {10, 20, 30},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .clone()
                       .to(torch::kCUDA);

    // Test reductions along different dimensions
    auto custom_sum0 = custom_t.sum(std::vector<int>{0});
    auto torch_sum0 = torch_t.sum(0);
    compare_tensors(custom_sum0, torch_sum0, 1e-3f, 1e-4f, "SumDim0");

    auto custom_mean1 = custom_t.mean(std::vector<int>{1});
    auto torch_mean1 = torch_t.mean(1);
    compare_tensors(custom_mean1, torch_mean1, 1e-4f, 1e-5f, "MeanDim1");

    auto custom_sum_all = custom_t.sum();
    auto torch_sum_all = torch_t.sum();
    compare_tensors(custom_sum_all, torch_sum_all, 1e-3f, 1e-4f, "SumAll");
}

// ============= Device Transfer Tests =============

TEST_F(TensorStressTest, MixedDeviceStress) {
    // Stress test device transfers
    const int iterations = 50; // Reduced for test speed
    const size_t size = 10000; // 10K elements

    for (int i = 0; i < iterations; ++i) {
        // Create on CPU
        auto custom_cpu = Tensor::full({size}, static_cast<float>(i), Device::CPU);
        auto torch_cpu = torch::full({static_cast<long>(size)}, static_cast<float>(i),
                                     torch::TensorOptions().device(torch::kCPU));

        // Transfer to CUDA
        auto custom_cuda = custom_cpu.to(Device::CUDA);
        auto torch_cuda = torch_cpu.to(torch::kCUDA);

        // Perform operation
        auto custom_result = custom_cuda.add(1.0f).mul(2.0f);
        auto torch_result = (torch_cuda + 1.0f) * 2.0f;

        // Transfer back
        auto custom_cpu_result = custom_result.to(Device::CPU);
        auto torch_cpu_result = torch_result.to(torch::kCPU);

        // Verify
        compare_tensors(custom_cpu_result, torch_cpu_result, 1e-6f, 1e-7f,
                        "MixedDevice" + std::to_string(i));
    }
}

// ============= Numerical Stability Tests =============

TEST_F(TensorStressTest, NumericalStability) {
    // Test numerical stability with extreme values
    std::vector<float> data(1000);
    for (size_t i = 0; i < 1000; ++i) {
        data[i] = std::pow(1.01f, static_cast<float>(i));
    }

    auto custom_tensor = Tensor::from_vector(data, {1000}, Device::CUDA);
    auto torch_tensor = torch::from_blob(data.data(), {1000},
                                         torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(torch::kCUDA);

    // Test log-exp roundtrip
    auto custom_log = custom_tensor.log();
    auto torch_log = torch_tensor.log();

    EXPECT_FALSE(custom_log.has_nan());
    EXPECT_FALSE(custom_log.has_inf());
    compare_tensors(custom_log, torch_log, 1e-4f, 1e-5f, "Log");

    auto custom_exp_log = custom_log.exp();
    auto torch_exp_log = torch_log.exp();

    compare_tensors(custom_exp_log, torch_exp_log, 1e-3f, 1e-3f, "ExpLog");

    // Test with very small values
    auto custom_small = Tensor::full({1000}, 1e-30f, Device::CUDA);
    auto torch_small = torch::full({1000}, 1e-30f,
                                   torch::TensorOptions().device(torch::kCUDA));

    auto custom_sqrt_small = custom_small.sqrt();
    auto torch_sqrt_small = torch_small.sqrt();

    EXPECT_FALSE(custom_sqrt_small.has_nan());
    compare_tensors(custom_sqrt_small, torch_sqrt_small, 1e-6f, 1e-7f, "SqrtSmall");
}

TEST_F(TensorStressTest, NormalizationStability) {
    // Test normalization with various distributions
    auto custom_t = Tensor::randn({1000}, Device::CUDA);

    auto data = custom_t.to_vector();
    auto torch_t = torch::from_blob(data.data(), {1000},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .clone()
                       .to(torch::kCUDA);

    auto custom_normalized = custom_t.normalize();

    auto torch_mean = torch_t.mean();
    auto torch_std = torch_t.std(/*unbiased=*/false);
    auto torch_normalized = (torch_t - torch_mean) / (torch_std + 1e-8f); // Add epsilon!

    EXPECT_FALSE(custom_normalized.has_nan());
    EXPECT_FALSE(custom_normalized.has_inf());

    compare_tensors(custom_normalized, torch_normalized, 1e-4f, 1e-5f, "Normalize");
}

// ============= Concurrent Operations Tests =============

TEST_F(TensorStressTest, ConcurrentOperations) {
    // Test concurrent operations on different tensors
    const int num_threads = 4; // Reduced for test reliability
    const int ops_per_thread = 50;

    std::vector<std::thread> threads;
    std::vector<Tensor> custom_results(num_threads);
    std::vector<torch::Tensor> torch_results(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&custom_results, &torch_results, t, ops_per_thread]() {
            auto custom_tensor = Tensor::full({50, 50}, static_cast<float>(t), Device::CUDA);
            auto torch_tensor = torch::full({50, 50}, static_cast<float>(t),
                                            torch::TensorOptions().device(torch::kCUDA));

            for (int i = 0; i < ops_per_thread; ++i) {
                custom_tensor = custom_tensor.add(0.1f).mul(1.01f).sub(0.05f);
                torch_tensor = (torch_tensor + 0.1f) * 1.01f - 0.05f;
            }

            custom_results[t] = std::move(custom_tensor);
            torch_results[t] = torch_tensor;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all results are valid and match PyTorch
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_TRUE(custom_results[t].is_valid());
        EXPECT_FALSE(custom_results[t].has_nan());
        EXPECT_FALSE(custom_results[t].has_inf());

        compare_tensors(custom_results[t], torch_results[t], 1e-3f, 1e-4f,
                        "Concurrent" + std::to_string(t));
    }
}

// ============= Edge Cases =============

TEST_F(TensorStressTest, EmptyTensorOperations) {
    auto custom_empty = Tensor::empty({0}, Device::CUDA);
    auto torch_empty = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(custom_empty.is_valid());
    EXPECT_EQ(custom_empty.numel(), 0);
    EXPECT_EQ(torch_empty.numel(), 0);

    // Operations on empty tensors should not crash
    auto custom_result = custom_empty.add(1.0f);
    auto torch_result = torch_empty + 1.0f;

    EXPECT_TRUE(custom_result.is_valid());
    EXPECT_EQ(custom_result.numel(), 0);
}

TEST_F(TensorStressTest, LargeReduction) {
    // Very large tensor reduction
    const size_t size = 10000000; // 10M elements

    auto custom_t = Tensor::ones({size}, Device::CUDA);
    auto torch_t = torch::ones({static_cast<long>(size)},
                               torch::TensorOptions().device(torch::kCUDA));

    auto custom_sum = custom_t.sum();
    auto torch_sum = torch_t.sum();

    compare_tensors(custom_sum, torch_sum, 1e-2f, 1e-3f, "LargeReduction");

    // Expected value
    EXPECT_NEAR(custom_sum.item(), static_cast<float>(size), size * 1e-5f);
}

TEST_F(TensorStressTest, ChainedReductions) {
    auto custom_t = Tensor::randn({100, 100, 10}, Device::CUDA);

    auto data = custom_t.to_vector();
    auto torch_t = torch::from_blob(data.data(), {100, 100, 10},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .clone()
                       .to(torch::kCUDA);

    // Chain multiple reductions
    auto custom_sum0 = custom_t.sum(std::vector<int>{0});
    auto custom_sum01 = custom_sum0.sum(std::vector<int>{0});

    auto torch_sum0 = torch_t.sum(0);
    auto torch_sum01 = torch_sum0.sum(0);

    compare_tensors(custom_sum01, torch_sum01, 1e-3f, 1e-4f, "ChainedReduction");
}