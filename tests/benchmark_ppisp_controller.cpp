/* Benchmark: PPISPController - Our implementation vs LibTorch
 *
 * Tests whether our manual implementation matches LibTorch performance.
 * The controller architecture (matching Python reference):
 *   Conv1x1(3→16) → MaxPool(3,3) → ReLU → Conv1x1(16→32) → ReLU →
 *   Conv1x1(32→64) → AdaptiveAvgPool(5,5) → Flatten →
 *   Concat(exposure) → Linear(1601→128) + ReLU → Linear(128→128) + ReLU →
 *   Linear(128→128) + ReLU → Linear(128→9)
 */

#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

#include "core/tensor.hpp"
#include "training/components/ppisp_controller.hpp"

using lfs::core::DataType;
using lfs::core::Device;

namespace {

    // LibTorch implementation of the same controller architecture (matching Python reference)
    struct LibTorchController : torch::nn::Module {
        LibTorchController() {
            // Conv layers (1x1 convolutions)
            conv1 = register_module("conv1", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(3, 16, 1).bias(true)));
            conv2 = register_module("conv2", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(16, 32, 1).bias(true)));
            conv3 = register_module("conv3", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(32, 64, 1).bias(true)));

            // Pooling
            maxpool = register_module("maxpool", torch::nn::MaxPool2d(
                                                     torch::nn::MaxPool2dOptions(3).stride(3)));
            avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
                                                     torch::nn::AdaptiveAvgPool2dOptions({5, 5})));

            // FC layers: 64*5*5 + 1 = 1601 → 128 → 128 → 128 → 9
            fc1 = register_module("fc1", torch::nn::Linear(1601, 128));
            fc2 = register_module("fc2", torch::nn::Linear(128, 128));
            fc3 = register_module("fc3", torch::nn::Linear(128, 128));
            fc4 = register_module("fc4", torch::nn::Linear(128, 9));
        }

        torch::Tensor forward(torch::Tensor x, torch::Tensor exposure_prior) {
            // x: [B, 3, H, W]
            // Conv1 (no relu) → MaxPool → ReLU → Conv2 → ReLU → Conv3 (no relu) → AvgPool
            x = conv1->forward(x);
            x = maxpool->forward(x);
            x = torch::relu(x);
            x = torch::relu(conv2->forward(x));
            x = conv3->forward(x);
            x = avgpool->forward(x);

            // Flatten: [B, 64, 5, 5] → [B, 1600]
            x = x.flatten(1);

            // Concat exposure prior: [B, 1600] + [B, 1] → [B, 1601]
            x = torch::cat({x, exposure_prior}, 1);

            // FC layers: 3 hidden layers with ReLU + output layer
            x = torch::relu(fc1->forward(x));
            x = torch::relu(fc2->forward(x));
            x = torch::relu(fc3->forward(x));
            x = fc4->forward(x);

            return x;
        }

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::MaxPool2d maxpool{nullptr};
        torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    };

    class PPISPControllerBenchmark : public ::testing::Test {
    protected:
        void SetUp() override {
            // Warmup GPU
            auto warmup = torch::randn({1, 3, 256, 256}, torch::kCUDA);
            cudaDeviceSynchronize();
        }
    };

    TEST_F(PPISPControllerBenchmark, ForwardPassComparison) {
        constexpr int NUM_WARMUP = 10;
        constexpr int NUM_ITERATIONS = 100;
        constexpr int BATCH_SIZE = 1;
        constexpr int HEIGHT = 540;
        constexpr int WIDTH = 960;

        std::cout << "\n=== PPISP Controller Forward Pass Benchmark ===" << std::endl;
        std::cout << "Input: [" << BATCH_SIZE << ", 3, " << HEIGHT << ", " << WIDTH << "]" << std::endl;
        std::cout << "Warmup: " << NUM_WARMUP << ", Iterations: " << NUM_ITERATIONS << std::endl;

        // Create LibTorch model
        auto torch_model = std::make_shared<LibTorchController>();
        torch_model->to(torch::kCUDA);
        torch_model->eval();

        // Create our model
        lfs::training::PPISPController our_model(30000);
        lfs::training::PPISPController::preallocate_shared_buffers(HEIGHT, WIDTH);

        // Create input tensors
        auto torch_input = torch::randn({BATCH_SIZE, 3, HEIGHT, WIDTH}, torch::kCUDA);
        auto torch_exposure = torch::ones({BATCH_SIZE, 1}, torch::kCUDA);

        auto our_input = lfs::core::Tensor::uniform({BATCH_SIZE, 3, HEIGHT, WIDTH}, -1.0f, 1.0f, Device::CUDA);

        // Warmup LibTorch
        {
            torch::NoGradGuard no_grad;
            for (int i = 0; i < NUM_WARMUP; ++i) {
                auto out = torch_model->forward(torch_input, torch_exposure);
            }
            cudaDeviceSynchronize();
        }

        // Warmup our implementation
        for (int i = 0; i < NUM_WARMUP; ++i) {
            auto out = our_model.predict(our_input, 1.0f);
        }
        cudaDeviceSynchronize();

        // Benchmark LibTorch
        double torch_total_ms = 0.0;
        {
            torch::NoGradGuard no_grad;
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                auto out = torch_model->forward(torch_input, torch_exposure);
            }
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            torch_total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }

        // Benchmark our implementation
        double our_total_ms = 0.0;
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                auto out = our_model.predict(our_input, 1.0f);
            }
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            our_total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }

        double torch_avg_ms = torch_total_ms / NUM_ITERATIONS;
        double our_avg_ms = our_total_ms / NUM_ITERATIONS;
        double ratio = our_avg_ms / torch_avg_ms;

        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "LibTorch:        " << torch_avg_ms << " ms/iter" << std::endl;
        std::cout << "Our impl:        " << our_avg_ms << " ms/iter" << std::endl;
        std::cout << "Ratio (ours/torch): " << ratio << "x" << std::endl;

        if (ratio > 2.0) {
            std::cout << "\n⚠️  WARNING: Our implementation is " << ratio << "x slower than LibTorch!" << std::endl;
        } else if (ratio < 0.5) {
            std::cout << "\n✓ Our implementation is " << (1.0 / ratio) << "x faster than LibTorch!" << std::endl;
        } else {
            std::cout << "\n✓ Performance is comparable (within 2x)" << std::endl;
        }

        // Don't fail the test, just report
        EXPECT_GT(torch_avg_ms, 0.0);
        EXPECT_GT(our_avg_ms, 0.0);
    }

    TEST_F(PPISPControllerBenchmark, ForwardPassVariousResolutions) {
        constexpr int NUM_WARMUP = 5;
        constexpr int NUM_ITERATIONS = 50;

        std::vector<std::pair<size_t, size_t>> resolutions = {
            {270, 480},   // Quarter HD
            {540, 960},   // Half HD
            {1080, 1920}, // Full HD
        };

        std::cout << "\n=== Resolution Scaling Benchmark ===" << std::endl;

        auto torch_model = std::make_shared<LibTorchController>();
        torch_model->to(torch::kCUDA);
        torch_model->eval();

        lfs::training::PPISPController our_model(30000);
        lfs::training::PPISPController::preallocate_shared_buffers(1080, 1920);

        for (const auto& [h, w] : resolutions) {
            auto torch_input = torch::randn({1, 3, static_cast<int64_t>(h), static_cast<int64_t>(w)}, torch::kCUDA);
            auto torch_exposure = torch::ones({1, 1}, torch::kCUDA);

            auto our_input = lfs::core::Tensor::uniform({1, 3, h, w}, -1.0f, 1.0f, Device::CUDA);

            // Warmup
            {
                torch::NoGradGuard no_grad;
                for (int i = 0; i < NUM_WARMUP; ++i) {
                    torch_model->forward(torch_input, torch_exposure);
                    our_model.predict(our_input, 1.0f);
                }
                cudaDeviceSynchronize();
            }

            // Benchmark
            double torch_ms, our_ms;
            {
                torch::NoGradGuard no_grad;
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < NUM_ITERATIONS; ++i) {
                    torch_model->forward(torch_input, torch_exposure);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                torch_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
            }
            {
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < NUM_ITERATIONS; ++i) {
                    our_model.predict(our_input, 1.0f);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                our_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
            }

            std::cout << h << "x" << w << ": LibTorch=" << torch_ms << "ms, Ours=" << our_ms
                      << "ms, Ratio=" << (our_ms / torch_ms) << "x" << std::endl;
        }
    }

    TEST_F(PPISPControllerBenchmark, MemoryUsageComparison) {
        std::cout << "\n=== Memory Usage Comparison ===" << std::endl;

        size_t free_before, total;
        cudaMemGetInfo(&free_before, &total);

        // Create LibTorch model
        auto torch_model = std::make_shared<LibTorchController>();
        torch_model->to(torch::kCUDA);
        cudaDeviceSynchronize();

        size_t free_after_torch;
        cudaMemGetInfo(&free_after_torch, &total);
        size_t torch_mem = free_before - free_after_torch;

        // Reset
        torch_model.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_before, &total);

        // Create our model
        lfs::training::PPISPController our_model(30000);
        cudaDeviceSynchronize();

        size_t free_after_ours;
        cudaMemGetInfo(&free_after_ours, &total);
        size_t our_mem = free_before - free_after_ours;

        std::cout << "LibTorch model memory: " << (torch_mem / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Our model memory:      " << (our_mem / 1024.0 / 1024.0) << " MB" << std::endl;

        // Note: Memory measurements may be unreliable due to GPU memory caching
        // This test is informational only - don't fail on zero measurements
    }

} // namespace
