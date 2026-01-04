/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Standalone profiling tool for Adam optimizer
 *
 * Run with:
 *   nsys profile --trace=cuda,nvtx --output=adam_profile ./tests/profile_adam_optimizer
 *   nsys-ui adam_profile.nsys-rep
 */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizers/fused_adam.hpp" // Reference optimizer
#include <chrono>
#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace lfs::core;
using namespace lfs::training;

// NVTX helper macros
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP()      nvtxRangePop()

class ScopedNVTXRange {
    const char* name_;

public:
    explicit ScopedNVTXRange(const char* name) : name_(name) {
        nvtxRangePushA(name_);
    }
    ~ScopedNVTXRange() {
        nvtxRangePop();
    }
};

#define NVTX_SCOPED_RANGE(name) ScopedNVTXRange _nvtx_range_##__LINE__(name)

// Helper to create test splat data
SplatData create_test_splat_data(size_t N, size_t dims) {
    auto means = Tensor::randn({N, dims}, Device::CUDA);
    auto sh0 = Tensor::randn({N, 1, 3}, Device::CUDA);
    auto shN = Tensor::randn({N, 15, 3}, Device::CUDA);
    auto scaling = Tensor::randn({N, 3}, Device::CUDA);
    auto rotation = Tensor::randn({N, 4}, Device::CUDA);
    auto opacity = Tensor::randn({N, 1}, Device::CUDA);
    return SplatData(3, std::move(means), std::move(sh0), std::move(shN),
                     std::move(scaling), std::move(rotation), std::move(opacity), 1.0f);
}

// Convert between tensor types
Tensor from_torch(const torch::Tensor& torch_tensor) {
    auto cpu_t = torch_tensor.cpu().contiguous();
    std::vector<float> vec(cpu_t.data_ptr<float>(),
                           cpu_t.data_ptr<float>() + cpu_t.numel());
    std::vector<size_t> shape;
    for (int i = 0; i < cpu_t.dim(); i++) {
        shape.push_back(cpu_t.size(i));
    }
    auto device = torch_tensor.is_cuda() ? Device::CUDA : Device::CPU;
    return Tensor::from_vector(vec, TensorShape(shape), device);
}

void profile_lfs_optimizer() {
    std::cout << "\n=== Profiling LFS Optimizer ===\n"
              << std::endl;

    const size_t N_initial = 100'000;
    const size_t N_add = 10'000;
    const int n_iterations = 100;
    const int add_every = 10;
    const float lr = 1e-3f;

    NVTX_SCOPED_RANGE("LFS_Optimizer_Setup");

    auto splat_data = create_test_splat_data(N_initial, 3);

    AdamConfig config;
    config.lr = lr;
    config.growth_factor = 1.5f;
    config.initial_capacity = N_initial + (n_iterations / add_every) * N_add;

    AdamOptimizer optimizer(splat_data, config);
    optimizer.allocate_gradients();

    size_t current_N = N_initial;

    std::cout << "Starting profiling loop..." << std::endl;
    std::cout << "Initial size: " << N_initial << std::endl;
    std::cout << "Iterations: " << n_iterations << std::endl;
    std::cout << "Add every: " << add_every << " iterations" << std::endl;
    std::cout << "Add amount: " << N_add << " per addition" << std::endl;
    std::cout << "Pre-allocated capacity: " << config.initial_capacity << std::endl;

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    {
        NVTX_SCOPED_RANGE("LFS_Optimizer_MainLoop");

        for (int iter = 0; iter < n_iterations; iter++) {
            {
                NVTX_SCOPED_RANGE("LFS_Step");
                // OPTIMIZATION: Use in-place normal_() to avoid allocation
                optimizer.get_grad(ParamType::Means).normal_(0.0f, 1.0f);
                optimizer.step(iter + 1);
            }

            if ((iter + 1) % add_every == 0) {
                NVTX_SCOPED_RANGE("LFS_AddParams");
                auto new_means = Tensor::randn({N_add, 3}, Device::CUDA);
                optimizer.add_new_params(ParamType::Means, new_means);
                current_N += N_add;
                std::cout << "  Iteration " << (iter + 1) << ": Added " << N_add
                          << " params (total: " << current_N << ")" << std::endl;
            }
        }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\nLFS Total time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Final Gaussian count: " << current_N << std::endl;
}

void profile_pytorch_optimizer() {
    std::cout << "\n=== Profiling PyTorch Optimizer ===\n"
              << std::endl;

    const size_t N_initial = 100'000;
    const size_t N_add = 10'000;
    const int n_iterations = 100;
    const int add_every = 10;
    const float lr = 1e-3f;

    NVTX_SCOPED_RANGE("PyTorch_Optimizer_Setup");

    auto torch_means = torch::randn({static_cast<long>(N_initial), 3},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                           .requires_grad_(true);

    auto torch_opt = std::make_unique<gs::training::FusedAdam>(
        std::vector<torch::Tensor>{torch_means},
        std::make_unique<gs::training::FusedAdam::Options>(lr));

    size_t current_N = N_initial;

    std::cout << "Starting profiling loop..." << std::endl;
    std::cout << "Initial size: " << N_initial << std::endl;

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    {
        NVTX_SCOPED_RANGE("PyTorch_Optimizer_MainLoop");

        for (int iter = 0; iter < n_iterations; iter++) {
            {
                NVTX_SCOPED_RANGE("PyTorch_Step");
                auto grad = torch::randn({static_cast<long>(current_N), 3},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                torch_means.mutable_grad() = grad;
                torch_opt->step(iter + 1);
            }

            if ((iter + 1) % add_every == 0) {
                NVTX_SCOPED_RANGE("PyTorch_AddParams");
                auto new_means = torch::randn({static_cast<long>(N_add), 3},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                auto extended = torch::cat({torch_means.detach(), new_means}, 0).requires_grad_(true);

                auto& state_map = torch_opt->state();
                auto state_it = state_map.find(torch_means.unsafeGetTensorImpl());
                auto* old_state = static_cast<gs::training::FusedAdam::AdamParamState*>(state_it->second.get());

                auto zeros = torch::zeros({static_cast<long>(N_add), 3}, old_state->exp_avg.options());
                auto new_state = std::make_unique<gs::training::FusedAdam::AdamParamState>();
                new_state->step_count = old_state->step_count;
                new_state->exp_avg = torch::cat({old_state->exp_avg, zeros}, 0);
                new_state->exp_avg_sq = torch::cat({old_state->exp_avg_sq, zeros}, 0);

                state_map.erase(torch_means.unsafeGetTensorImpl());
                torch_opt->param_groups()[0].params()[0] = extended;
                state_map[extended.unsafeGetTensorImpl()] = std::move(new_state);
                torch_means = extended;

                current_N += N_add;
                std::cout << "  Iteration " << (iter + 1) << ": Added " << N_add
                          << " params (total: " << current_N << ")" << std::endl;
            }
        }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\nPyTorch Total time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Final Gaussian count: " << current_N << std::endl;
}

void profile_step_operation_detailed() {
    std::cout << "\n=== Detailed Step Operation Profiling ===\n"
              << std::endl;

    const size_t N = 100'000;
    const int n_steps = 50;
    const float lr = 1e-3f;

    std::cout << "Profiling " << n_steps << " step operations with " << N << " parameters..." << std::endl;

    // LFS
    {
        NVTX_SCOPED_RANGE("LFS_DetailedSteps_Setup");
        auto splat_data = create_test_splat_data(N, 3);

        AdamConfig config;
        config.lr = lr;
        AdamOptimizer optimizer(splat_data, config);
        optimizer.allocate_gradients();

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        {
            NVTX_SCOPED_RANGE("LFS_DetailedSteps_Loop");
            for (int i = 0; i < n_steps; i++) {
                NVTX_SCOPED_RANGE("LFS_SingleStep");
                // OPTIMIZATION: Use in-place normal_() to avoid allocation
                optimizer.get_grad(ParamType::Means).normal_(0.0f, 1.0f);
                optimizer.step(i + 1);
            }
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "LFS: " << elapsed.count() << " ms total, "
                  << (elapsed.count() / n_steps) << " ms per step" << std::endl;
    }

    // PyTorch
    {
        NVTX_SCOPED_RANGE("PyTorch_DetailedSteps_Setup");
        auto torch_means = torch::randn({static_cast<long>(N), 3},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                               .requires_grad_(true);

        auto torch_opt = std::make_unique<gs::training::FusedAdam>(
            std::vector<torch::Tensor>{torch_means},
            std::make_unique<gs::training::FusedAdam::Options>(lr));

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        {
            NVTX_SCOPED_RANGE("PyTorch_DetailedSteps_Loop");
            for (int i = 0; i < n_steps; i++) {
                NVTX_SCOPED_RANGE("PyTorch_SingleStep");
                auto grad = torch::randn({static_cast<long>(N), 3},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                torch_means.mutable_grad() = grad;
                torch_opt->step(i + 1);
            }
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "PyTorch: " << elapsed.count() << " ms total, "
                  << (elapsed.count() / n_steps) << " ms per step" << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "           Adam Optimizer Performance Profiling Tool\n";
    std::cout << "========================================================================\n";
    std::cout << "\nRun with Nsight Systems:\n";
    std::cout << "  nsys profile --trace=cuda,nvtx --output=adam_profile \\\n";
    std::cout << "    ./build/tests/profile_adam_optimizer\n";
    std::cout << "\nView results:\n";
    std::cout << "  nsys-ui adam_profile.nsys-rep\n";
    std::cout << "\n";

    // Run detailed step profiling first (smaller workload)
    profile_step_operation_detailed();

    // Run full optimizer workflows
    profile_lfs_optimizer();
    profile_pytorch_optimizer();

    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "Profiling complete! Check the nsys-rep file for detailed analysis.\n";
    std::cout << "========================================================================\n";
    std::cout << "\n";

    return 0;
}
