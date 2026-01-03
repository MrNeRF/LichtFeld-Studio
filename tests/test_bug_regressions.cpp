/* Binary search to find what corrupts CUDA state */
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "optimizer/render_output.hpp"
#include "training/strategies/default_strategy.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace lfs::core;

// Test 1: Just create SplatData
TEST(BinarySearchBug, Step1_CreateSplatData) {
    std::cout << "\n=== STEP 1: Create SplatData ===" << std::endl;

    cudaSetDevice(0);

    try {
        size_t n = 10000;
        auto rotation = Tensor::zeros({n, 4}, Device::CUDA);
        rotation.slice(1, 0, 1).fill_(1.0f);

        lfs::core::SplatData splat(3,
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 48}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA) - 2.0f,
                                   rotation,
                                   Tensor::randn({n, 1}, Device::CUDA),
                                   1.0f);

        std::cout << "[TEST] SplatData created, size=" << splat.size() << std::endl;

        // Now try tensor operation
        auto w = Tensor::zeros({1000}, Device::CUDA);
        auto two = Tensor::full_like(w, 2.0f);
        std::cout << "[TEST] Tensor operations after SplatData creation: SUCCESS" << std::endl;

        SUCCEED();
    } catch (const std::exception& e) {
        std::cout << "[TEST] FAILED: " << e.what() << std::endl;
        FAIL() << e.what();
    }
}

// Test 2: Create SplatData + DefaultStrategy (no initialize)
TEST(BinarySearchBug, Step2_CreateStrategy) {
    std::cout << "\n=== STEP 2: Create DefaultStrategy ===" << std::endl;

    cudaSetDevice(0);

    try {
        size_t n = 10000;
        auto rotation = Tensor::zeros({n, 4}, Device::CUDA);
        rotation.slice(1, 0, 1).fill_(1.0f);

        lfs::core::SplatData splat(3,
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 48}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA) - 2.0f,
                                   rotation,
                                   Tensor::randn({n, 1}, Device::CUDA),
                                   1.0f);

        std::cout << "[TEST] SplatData created" << std::endl;

        lfs::training::DefaultStrategy strat(splat);
        std::cout << "[TEST] DefaultStrategy created" << std::endl;

        // Now try tensor operation
        auto w = Tensor::zeros({1000}, Device::CUDA);
        auto two = Tensor::full_like(w, 2.0f);
        std::cout << "[TEST] Tensor operations after Strategy creation: SUCCESS" << std::endl;

        SUCCEED();
    } catch (const std::exception& e) {
        std::cout << "[TEST] FAILED: " << e.what() << std::endl;
        FAIL() << e.what();
    }
}

// Test 3: Create + Initialize Strategy
TEST(BinarySearchBug, Step3_InitializeStrategy) {
    std::cout << "\n=== STEP 3: Initialize Strategy ===" << std::endl;

    cudaSetDevice(0);

    try {
        size_t n = 10000;
        auto rotation = Tensor::zeros({n, 4}, Device::CUDA);
        rotation.slice(1, 0, 1).fill_(1.0f);

        lfs::core::SplatData splat(3,
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 48}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA) - 2.0f,
                                   rotation,
                                   Tensor::randn({n, 1}, Device::CUDA),
                                   1.0f);

        lfs::training::DefaultStrategy strat(splat);

        lfs::core::param::OptimizationParameters params;
        params.iterations = 30000;
        params.start_refine = 500;
        params.refine_every = 100;
        params.stop_refine = 15000;
        params.grad_threshold = 0.0002f;
        params.grow_scale3d = 0.01f;
        params.prune_scale3d = 0.15f;
        params.prune_opacity = 0.005f;
        params.reset_every = 3000;
        params.pause_refine_after_reset = 0;
        params.sh_degree_interval = 1000;

        strat.initialize(params);
        std::cout << "[TEST] Strategy initialized" << std::endl;

        // Now try tensor operation
        auto w = Tensor::zeros({1000}, Device::CUDA);
        auto two = Tensor::full_like(w, 2.0f);
        std::cout << "[TEST] Tensor operations after Strategy initialize: SUCCESS" << std::endl;

        SUCCEED();
    } catch (const std::exception& e) {
        std::cout << "[TEST] FAILED: " << e.what() << std::endl;
        FAIL() << e.what();
    }
}

// Test 4: Full workflow up to split()
TEST(BinarySearchBug, Step4_CallPostBackward) {
    std::cout << "\n=== STEP 4: Call post_backward ===" << std::endl;

    cudaSetDevice(0);

    try {
        size_t n = 10000;
        auto rotation = Tensor::zeros({n, 4}, Device::CUDA);
        rotation.slice(1, 0, 1).fill_(1.0f);

        lfs::core::SplatData splat(3,
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA),
                                   Tensor::randn({n, 48}, Device::CUDA),
                                   Tensor::randn({n, 3}, Device::CUDA) - 2.0f,
                                   rotation,
                                   Tensor::randn({n, 1}, Device::CUDA),
                                   1.0f);

        splat._densification_info = Tensor::ones({2, n}, Device::CUDA);
        auto numer = Tensor::ones({n}, Device::CUDA) * 10.0f;
        splat._densification_info[1] = numer;

        lfs::training::DefaultStrategy strat(splat);

        lfs::core::param::OptimizationParameters params;
        params.iterations = 30000;
        params.start_refine = 500;
        params.refine_every = 100;
        params.stop_refine = 15000;
        params.grad_threshold = 0.0002f;
        params.grow_scale3d = 0.01f;
        params.prune_scale3d = 0.15f;
        params.prune_opacity = 0.005f;
        params.reset_every = 3000;
        params.pause_refine_after_reset = 0;
        params.sh_degree_interval = 1000;

        strat.initialize(params);

        // Re-initialize densification info
        strat.get_model()._densification_info = Tensor::ones({2, n}, Device::CUDA);
        auto numer2 = Tensor::ones({n}, Device::CUDA) * 10.0f;
        strat.get_model()._densification_info[1] = numer2;

        lfs::training::RenderOutput render_output;
        int test_iter = 600;

        std::cout << "[TEST] Calling post_backward..." << std::endl;
        strat.post_backward(test_iter, render_output);

        std::cout << "[TEST] post_backward SUCCESS" << std::endl;
        SUCCEED();
    } catch (const std::exception& e) {
        std::cout << "[TEST] FAILED: " << e.what() << std::endl;
        FAIL() << e.what();
    }
}
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Dedicated tests to isolate the index_select bug that occurs at ~100K elements

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace lfs::core;

// Test index_select at different scales to find the breaking point
TEST(TensorIndexSelectBug, ScaleTest_10K) {
    std::cout << "Testing index_select with 10K elements..." << std::endl;

    // Create source tensor [10000, 4]
    auto data = Tensor::randn({10000, 4}, Device::CUDA);

    // Create indices for 9988 elements (similar to what fails at 100K)
    std::vector<int> idx_data(9988);
    for (int i = 0; i < 9988; ++i) {
        idx_data[i] = i;
    }
    auto indices = Tensor::from_vector(idx_data, TensorShape({9988}), Device::CUDA);

    std::cout << "  Source shape: [10000, 4], device=" << static_cast<int>(data.device()) << std::endl;
    std::cout << "  Indices shape: [9988], device=" << static_cast<int>(indices.device()) << std::endl;

    // This should work
    EXPECT_NO_THROW({
        auto result = data.index_select(0, indices);
        std::cout << "  Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]" << std::endl;
        EXPECT_EQ(result.shape()[0], 9988);
        EXPECT_EQ(result.shape()[1], 4);
    });
}

TEST(TensorIndexSelectBug, ScaleTest_50K) {
    std::cout << "Testing index_select with 50K elements..." << std::endl;

    // Create source tensor [50000, 4]
    auto data = Tensor::randn({50000, 4}, Device::CUDA);

    // Create indices for ~49940 elements
    std::vector<int> idx_data(49940);
    for (int i = 0; i < 49940; ++i) {
        idx_data[i] = i;
    }
    auto indices = Tensor::from_vector(idx_data, TensorShape({49940}), Device::CUDA);

    std::cout << "  Source shape: [50000, 4], device=" << static_cast<int>(data.device()) << std::endl;
    std::cout << "  Indices shape: [49940], device=" << static_cast<int>(indices.device()) << std::endl;

    EXPECT_NO_THROW({
        auto result = data.index_select(0, indices);
        std::cout << "  Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]" << std::endl;
        EXPECT_EQ(result.shape()[0], 49940);
        EXPECT_EQ(result.shape()[1], 4);
    });
}

TEST(TensorIndexSelectBug, ScaleTest_100K) {
    std::cout << "Testing index_select with 100K elements (THIS MAY FAIL)..." << std::endl;

    // Create source tensor [100000, 4]
    auto data = Tensor::randn({100000, 4}, Device::CUDA);

    // Create indices for 99880 elements (this is what fails in densification)
    std::vector<int> idx_data(99880);
    for (int i = 0; i < 99880; ++i) {
        idx_data[i] = i;
    }
    auto indices = Tensor::from_vector(idx_data, TensorShape({99880}), Device::CUDA);

    std::cout << "  Source shape: [100000, 4], device=" << static_cast<int>(data.device()) << std::endl;
    std::cout << "  Indices shape: [99880], device=" << static_cast<int>(indices.device()) << std::endl;

    try {
        auto result = data.index_select(0, indices);
        std::cout << "  SUCCESS! Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]" << std::endl;
        EXPECT_EQ(result.shape()[0], 99880);
        EXPECT_EQ(result.shape()[1], 4);
    } catch (const std::exception& e) {
        std::cout << "  FAILED with error: " << e.what() << std::endl;
        FAIL() << "index_select failed at 100K scale: " << e.what();
    }
}

// Test the specific pattern from get_rotation()
TEST(TensorIndexSelectBug, GetRotationPattern_10K) {
    std::cout << "Testing get_rotation() pattern with 10K elements..." << std::endl;

    // Simulate get_rotation(): [N, 4] -> normalize -> index_select
    auto rotation = Tensor::randn({10000, 4}, Device::CUDA);

    // Normalize (this is what get_rotation does)
    auto squared = rotation.square();
    auto sum_squared = squared.sum({1}, true); // [N, 1]
    auto norm = sum_squared.sqrt();
    auto normalized = rotation.div(norm.clamp_min(1e-12f));

    std::cout << "  Normalized rotation shape: [" << normalized.shape()[0] << ", " << normalized.shape()[1] << "]" << std::endl;

    // Now index_select
    std::vector<int> idx_data(9988);
    for (int i = 0; i < 9988; ++i) {
        idx_data[i] = i;
    }
    auto indices = Tensor::from_vector(idx_data, TensorShape({9988}), Device::CUDA);

    EXPECT_NO_THROW({
        auto result = normalized.index_select(0, indices);
        std::cout << "  Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]" << std::endl;
        EXPECT_EQ(result.shape()[0], 9988);
        EXPECT_EQ(result.shape()[1], 4);
    });
}

TEST(TensorIndexSelectBug, GetRotationPattern_100K) {
    std::cout << "Testing get_rotation() pattern with 100K elements (THIS MAY FAIL)..." << std::endl;

    // Simulate get_rotation(): [N, 4] -> normalize -> index_select
    auto rotation = Tensor::randn({100000, 4}, Device::CUDA);

    std::cout << "  Created rotation tensor" << std::endl;

    // Normalize (this is what get_rotation does)
    std::cout << "  Computing squared..." << std::endl;
    auto squared = rotation.square();

    std::cout << "  Computing sum_squared..." << std::endl;
    auto sum_squared = squared.sum({1}, true); // [N, 1] - THIS MIGHT BE WHERE IT FAILS

    std::cout << "  Computing norm..." << std::endl;
    auto norm = sum_squared.sqrt();

    std::cout << "  Computing normalized..." << std::endl;
    auto normalized = rotation.div(norm.clamp_min(1e-12f));

    std::cout << "  Normalized rotation shape: [" << normalized.shape()[0] << ", " << normalized.shape()[1] << "]" << std::endl;

    // Now index_select
    std::vector<int> idx_data(99880);
    for (int i = 0; i < 99880; ++i) {
        idx_data[i] = i;
    }
    auto indices = Tensor::from_vector(idx_data, TensorShape({99880}), Device::CUDA);

    try {
        std::cout << "  Calling index_select..." << std::endl;
        auto result = normalized.index_select(0, indices);
        std::cout << "  SUCCESS! Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]" << std::endl;
        EXPECT_EQ(result.shape()[0], 99880);
        EXPECT_EQ(result.shape()[1], 4);
    } catch (const std::exception& e) {
        std::cout << "  FAILED with error: " << e.what() << std::endl;
        FAIL() << "get_rotation pattern failed at 100K scale: " << e.what();
    }
}

// Test just the sum operation
TEST(TensorIndexSelectBug, SumAlongDim_100K) {
    std::cout << "Testing sum({1}, true) with 100K elements..." << std::endl;

    auto data = Tensor::randn({100000, 4}, Device::CUDA);

    try {
        std::cout << "  Calling sum({1}, true)..." << std::endl;
        auto result = data.sum({1}, true);
        std::cout << "  SUCCESS! Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]" << std::endl;
        EXPECT_EQ(result.shape()[0], 100000);
        EXPECT_EQ(result.shape()[1], 1);
    } catch (const std::exception& e) {
        std::cout << "  FAILED with error: " << e.what() << std::endl;
        FAIL() << "sum({1}, true) failed at 100K scale: " << e.what();
    }
}

// Test at different thresholds to find exact breaking point
TEST(TensorIndexSelectBug, FindBreakingPoint) {
    std::cout << "Finding exact breaking point for index_select..." << std::endl;

    std::vector<int> sizes = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};

    for (size_t size : sizes) {
        std::cout << "  Testing size " << size << "..." << std::endl;

        auto data = Tensor::randn({size, 4}, Device::CUDA);

        // Select ~99.88% of elements (same ratio as the bug)
        int num_select = static_cast<int>(size * 0.9988);
        std::vector<int> idx_data(num_select);
        for (int i = 0; i < num_select; ++i) {
            idx_data[i] = i;
        }
        auto indices = Tensor::from_vector(idx_data, TensorShape({static_cast<size_t>(num_select)}), Device::CUDA);

        try {
            auto result = data.index_select(0, indices);
            std::cout << "    ✓ Size " << size << " works" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "    ✗ Size " << size << " FAILS: " << e.what() << std::endl;
            // Don't fail the test, just report
        }
    }
}
/* Minimal test using ONLY lfs_tensor library to isolate the bug */
#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace lfs::core;

TEST(MinimalTensorIsolated, FullLikeWorks) {
    std::cout << "\n=== MINIMAL TENSOR ISOLATED TEST ===" << std::endl;

    // Initialize CUDA
    int device_count = -1;
    cudaGetDeviceCount(&device_count);
    printf("[TEST] CUDA device count: %d\n", device_count);

    cudaSetDevice(0);
    int current_device = -1;
    cudaGetDevice(&current_device);
    printf("[TEST] Current CUDA device: %d\n", current_device);

    // Force CUDA runtime initialization
    void* dummy_ptr = nullptr;
    cudaMalloc(&dummy_ptr, 1024);
    cudaFree(dummy_ptr);
    printf("[TEST] CUDA runtime initialized\n");

    try {
        // Create a simple tensor (this will use cuRAND)
        std::cout << "[TEST] Creating tensor..." << std::endl;
        auto w = Tensor::randn({10000}, Device::CUDA);
        std::cout << "[TEST] Tensor created successfully" << std::endl;

        std::cout << "[TEST] Calling full_like (this triggers thrust::fill)..." << std::endl;

        // This should trigger the bug if it exists
        auto two = Tensor::full_like(w, 2.0f);
        std::cout << "[TEST] full_like SUCCESS!" << std::endl;

        EXPECT_EQ(two.shape()[0], 10000);
        std::cout << "[TEST] Test PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[TEST] FAILED: " << e.what() << std::endl;
        FAIL() << "Tensor test failed: " << e.what();
    }
}

// Test with even simpler tensor creation (no randn)
TEST(MinimalTensorIsolated, FullLikeWithZeros) {
    std::cout << "\n=== MINIMAL TENSOR ISOLATED TEST (ZEROS) ===" << std::endl;

    cudaSetDevice(0);

    try {
        // Create tensor with zeros (simpler, no cuRAND)
        std::cout << "[TEST] Creating zeros tensor..." << std::endl;
        auto w = Tensor::zeros({10000}, Device::CUDA);
        std::cout << "[TEST] Zeros tensor created" << std::endl;

        std::cout << "[TEST] Calling full_like..." << std::endl;
        auto two = Tensor::full_like(w, 2.0f);
        std::cout << "[TEST] full_like SUCCESS!" << std::endl;

        EXPECT_EQ(two.shape()[0], 10000);
        std::cout << "[TEST] Test PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[TEST] FAILED: " << e.what() << std::endl;
        FAIL() << "Tensor test failed: " << e.what();
    }
}
