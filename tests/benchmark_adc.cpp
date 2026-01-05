/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Performance benchmarks for ADC
// Measures timing of critical operations to ensure performance matches or exceeds reference

#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "training/strategies/adc.hpp"
#include <chrono>
#include <gtest/gtest.h>

using namespace lfs::training;
using namespace lfs::core;

// Helper to create test SplatData
static SplatData create_benchmark_splat_data(int n_gaussians) {
    std::vector<float> means_data(n_gaussians * 3);
    std::vector<float> sh0_data(n_gaussians * 3, 0.5f);
    std::vector<float> shN_data(n_gaussians * 48, 0.0f);
    std::vector<float> scaling_data(n_gaussians * 3);
    std::vector<float> rotation_data(n_gaussians * 4);

    // Initialize with random-ish values
    for (int i = 0; i < n_gaussians; ++i) {
        means_data[i * 3 + 0] = (i % 100) / 100.0f;
        means_data[i * 3 + 1] = ((i + 33) % 100) / 100.0f;
        means_data[i * 3 + 2] = ((i + 67) % 100) / 100.0f;

        scaling_data[i * 3 + 0] = -2.0f + (i % 10) / 10.0f;
        scaling_data[i * 3 + 1] = -2.0f + ((i + 3) % 10) / 10.0f;
        scaling_data[i * 3 + 2] = -2.0f + ((i + 7) % 10) / 10.0f;

        rotation_data[i * 4 + 0] = 1.0f; // w
        rotation_data[i * 4 + 1] = 0.0f; // x
        rotation_data[i * 4 + 2] = 0.0f; // y
        rotation_data[i * 4 + 3] = 0.0f; // z
    }

    std::vector<float> opacity_data(n_gaussians, 0.5f);

    auto means = Tensor::from_vector(means_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
    auto sh0 = Tensor::from_vector(sh0_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
    auto shN = Tensor::from_vector(shN_data, TensorShape({static_cast<size_t>(n_gaussians), 48}), Device::CUDA);
    auto scaling = Tensor::from_vector(scaling_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
    auto rotation = Tensor::from_vector(rotation_data, TensorShape({static_cast<size_t>(n_gaussians), 4}), Device::CUDA);
    auto opacity = Tensor::from_vector(opacity_data, TensorShape({static_cast<size_t>(n_gaussians), 1}), Device::CUDA);

    return SplatData(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
}

// Helper to time operations
template <typename Func>
double time_operation(Func&& func, int warmup_runs = 3, int timed_runs = 10) {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        func();
    }

    // Synchronize before timing
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < timed_runs; ++i) {
        func();
    }

    // Synchronize after timing
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    return elapsed.count() / timed_runs; // Return average time in ms
}

// Benchmark initialization
TEST(ADCBenchmark, Initialization) {
    const int n_gaussians = 100000;

    double avg_time = time_operation([n_gaussians]() {
        auto splat_data = create_benchmark_splat_data(n_gaussians);
        ADC strategy(splat_data);

        param::OptimizationParameters opt_params;
        opt_params.iterations = 30000;
        strategy.initialize(opt_params);
    },
                                     1, 5);

    std::cout << "Initialization (" << n_gaussians << " Gaussians): "
              << avg_time << " ms" << std::endl;

    // Sanity check: should complete in reasonable time
    EXPECT_LT(avg_time, 1000.0); // Less than 1 second
}

// Benchmark remove operation
TEST(ADCBenchmark, RemoveGaussians) {
    const int n_gaussians = 100000;
    const int n_to_remove = 10000;

    auto splat_data = create_benchmark_splat_data(n_gaussians);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 30000;
    strategy.initialize(opt_params);

    // Create mask
    auto mask = Tensor::zeros_bool({static_cast<size_t>(n_gaussians)}, Device::CPU);
    auto mask_ptr = mask.ptr<unsigned char>();
    for (int i = 0; i < n_to_remove; ++i) {
        mask_ptr[i] = 1;
    }
    mask = mask.to(Device::CUDA);

    double avg_time = time_operation([&strategy, &mask]() {
        // Note: This modifies the strategy, so we can only do it once
        // For proper benchmarking, we'd need to recreate the strategy each time
        // Here we just measure a single operation
        strategy.remove_gaussians(mask);
    },
                                     0, 1);

    std::cout << "RemoveGaussians (" << n_to_remove << " of " << n_gaussians << "): "
              << avg_time << " ms" << std::endl;

    // Soft deletion: tensor size unchanged, active_count reflects logical size
    EXPECT_EQ(strategy.get_model().size(), n_gaussians);           // Physical size unchanged
    EXPECT_EQ(strategy.active_count(), n_gaussians - n_to_remove); // Logical size
    EXPECT_LT(avg_time, 500.0);                                    // Should complete quickly
}

// Benchmark training step
TEST(ADCBenchmark, TrainingStep) {
    const int n_gaussians = 100000;

    auto splat_data = create_benchmark_splat_data(n_gaussians);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 30000;
    opt_params.means_lr = 0.00016f;
    strategy.initialize(opt_params);

    // Initialize densification info
    strategy.get_model()._densification_info = Tensor::zeros({2, static_cast<size_t>(n_gaussians)}, Device::CUDA);

    RenderOutput render_output;

    // Simulate some gradients
    auto& means_grad = strategy.get_optimizer().get_grad(lfs::training::ParamType::Means);
    if (means_grad.is_valid()) {
        means_grad = Tensor::rand_like(means_grad) * 0.001f;
    }

    double avg_time = time_operation([&strategy, &render_output]() {
        strategy.post_backward(100, render_output);
        strategy.step(100);
    });

    std::cout << "Training step (" << n_gaussians << " Gaussians): "
              << avg_time << " ms" << std::endl;

    EXPECT_LT(avg_time, 100.0); // Should be fast
}

// Benchmark full training loop
TEST(ADCBenchmark, FullTrainingLoop_100Iterations) {
    const int n_gaussians = 50000;
    const int n_iterations = 100;

    auto splat_data = create_benchmark_splat_data(n_gaussians);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = n_iterations;
    opt_params.start_refine = 20;
    opt_params.stop_refine = 80;
    opt_params.refine_every = 20;
    opt_params.reset_every = 60;
    strategy.initialize(opt_params);

    strategy.get_model()._densification_info = Tensor::zeros(
        {2, static_cast<size_t>(strategy.get_model().size())}, Device::CUDA);

    RenderOutput render_output;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iterations; ++iter) {
        // Simulate gradients
        auto& means_grad = strategy.get_optimizer().get_grad(lfs::training::ParamType::Means);
        if (means_grad.is_valid()) {
            means_grad = Tensor::rand_like(means_grad) * 0.001f;
        }

        int size_before = strategy.get_model().size();
        strategy.post_backward(iter, render_output);

        // Resize densification info if size changed
        if (strategy.get_model().size() != size_before) {
            strategy.get_model()._densification_info = Tensor::zeros(
                {2, static_cast<size_t>(strategy.get_model().size())},
                Device::CUDA);
        }

        strategy.step(iter);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Full training loop (" << n_gaussians << " Gaussians, "
              << n_iterations << " iterations): " << elapsed.count() << " ms" << std::endl;
    std::cout << "  Average per iteration: " << elapsed.count() / n_iterations << " ms" << std::endl;
    std::cout << "  Final Gaussian count: " << strategy.get_model().size() << std::endl;

    EXPECT_GT(strategy.get_model().size(), 0);
    EXPECT_LT(elapsed.count(), 10000.0); // Should complete in reasonable time
}

// Benchmark scaling with different Gaussian counts
TEST(ADCBenchmark, ScalingTest) {
    std::vector<int> gaussian_counts = {10000, 50000, 100000, 200000};

    std::cout << "\nScaling benchmark:" << std::endl;
    std::cout << "Gaussians\tInit(ms)\tStep(ms)" << std::endl;

    for (int n_gaussians : gaussian_counts) {
        // Initialization time
        double init_time = time_operation([n_gaussians]() {
            auto splat_data = create_benchmark_splat_data(n_gaussians);
            ADC strategy(splat_data);

            param::OptimizationParameters opt_params;
            opt_params.iterations = 30000;
            strategy.initialize(opt_params);
        },
                                          1, 3);

        // Single step time
        auto splat_data = create_benchmark_splat_data(n_gaussians);
        ADC strategy(splat_data);

        param::OptimizationParameters opt_params;
        opt_params.iterations = 30000;
        strategy.initialize(opt_params);

        strategy.get_model()._densification_info = Tensor::zeros(
            {2, static_cast<size_t>(n_gaussians)}, Device::CUDA);

        RenderOutput render_output;

        auto& means_grad = strategy.get_optimizer().get_grad(lfs::training::ParamType::Means);
        if (means_grad.is_valid()) {
            means_grad = Tensor::rand_like(means_grad) * 0.001f;
        }

        double step_time = time_operation([&strategy, &render_output]() {
            strategy.post_backward(100, render_output);
            strategy.step(100);
        });

        std::cout << n_gaussians << "\t\t" << init_time << "\t\t" << step_time << std::endl;

        // Verify reasonable performance
        EXPECT_LT(init_time, 2000.0); // Init should be < 2 seconds
        EXPECT_LT(step_time, 200.0);  // Step should be < 200ms
    }
}

// Benchmark memory efficiency
TEST(ADCBenchmark, MemoryEfficiency) {
    const int n_gaussians = 100000;

    auto splat_data = create_benchmark_splat_data(n_gaussians);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 30000;
    strategy.initialize(opt_params);

    strategy.get_model()._densification_info = Tensor::zeros(
        {2, static_cast<size_t>(n_gaussians)}, Device::CUDA);

    // Calculate theoretical minimum memory per Gaussian
    // Each Gaussian has:
    // - means (3 * 4 bytes) = 12 bytes
    // - sh0 (3 * 4 bytes) = 12 bytes
    // - shN (48 * 4 bytes) = 192 bytes (for degree 3)
    // - scaling (3 * 4 bytes) = 12 bytes
    // - rotation (4 * 4 bytes) = 16 bytes
    // - opacity (1 * 4 bytes) = 4 bytes
    // Total params: 248 bytes
    //
    // Plus gradients (same size): 248 bytes
    // Plus optimizer state (2x params for Adam): 496 bytes
    // Theoretical minimum: ~992 bytes per Gaussian

    const size_t bytes_per_param = 248;
    const size_t bytes_per_grad = 248;
    const size_t bytes_per_optimizer_state = 496;
    const size_t theoretical_min = bytes_per_param + bytes_per_grad + bytes_per_optimizer_state;

    std::cout << "Memory efficiency (" << n_gaussians << " Gaussians):" << std::endl;
    std::cout << "  Theoretical minimum per Gaussian: " << theoretical_min << " bytes" << std::endl;
    std::cout << "  Params: " << bytes_per_param << " bytes" << std::endl;
    std::cout << "  Gradients: " << bytes_per_grad << " bytes" << std::endl;
    std::cout << "  Optimizer state: " << bytes_per_optimizer_state << " bytes" << std::endl;

    // Verify model has correct number of Gaussians
    EXPECT_EQ(strategy.get_model().size(), n_gaussians);

    // Note: cudaMemGetInfo is unreliable due to CUDA memory allocator caching
    // So we just document the theoretical memory usage here
    SUCCEED();
}
