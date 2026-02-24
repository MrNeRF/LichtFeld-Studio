/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

using namespace lfs::core;

namespace {

    class LazyTestGuard {
    public:
        LazyTestGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            Tensor::reset_lazy_telemetry();
        }
        ~LazyTestGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            Tensor::reset_lazy_telemetry();
        }
    };

    struct BenchResult {
        double lfs_fused_us = 0.0;
        double lfs_eager_us = 0.0;
        double torch_us = 0.0;
        bool fused_vs_torch = false;
        bool eager_vs_torch = false;
        bool fused_vs_eager = false;

        bool all_verified() const { return fused_vs_torch && eager_vs_torch && fused_vs_eager; }
    };

    void print_header() {
        printf("[fusion-perf] %-45s %10s %10s %10s %10s %10s %s\n",
               "Scenario", "LFS-fused", "LFS-eager", "Torch", "vs-Torch", "vs-Eager", "Correct");
        printf("[fusion-perf] %s\n", std::string(115, '-').c_str());
    }

    void print_row(const char* name, const BenchResult& r) {
        const double vs_torch = (r.lfs_fused_us > 0.0) ? r.torch_us / r.lfs_fused_us : 0.0;
        const double vs_eager = (r.lfs_fused_us > 0.0) ? r.lfs_eager_us / r.lfs_fused_us : 0.0;
        const char* status = r.all_verified() ? "PASS" : "FAIL";
        printf("[fusion-perf] %-45s %8.1fus %8.1fus %8.1fus %8.2fx %8.2fx    %s\n",
               name, r.lfs_fused_us, r.lfs_eager_us, r.torch_us, vs_torch, vs_eager, status);
        if (!r.all_verified()) {
            if (!r.fused_vs_torch)
                printf("[fusion-perf]   -> fused vs torch: FAIL\n");
            if (!r.eager_vs_torch)
                printf("[fusion-perf]   -> eager vs torch: FAIL\n");
            if (!r.fused_vs_eager)
                printf("[fusion-perf]   -> fused vs eager: FAIL\n");
        }
    }

    constexpr int kWarmup = 20;
    constexpr int kIters = 200;

    bool allclose(const float* a, const float* b, size_t n, float atol = 1e-4f, float rtol = 1e-4f) {
        for (size_t i = 0; i < n; ++i) {
            const float diff = std::abs(a[i] - b[i]);
            const float ref = std::max(std::abs(a[i]), std::abs(b[i]));
            if (diff > atol + rtol * ref)
                return false;
        }
        return true;
    }

    bool verify_against_torch(const Tensor& lfs_result, const torch::Tensor& torch_result,
                              float atol = 1e-4f, float rtol = 1e-4f) {
        auto lfs_cpu = lfs_result.to(Device::CPU);
        auto torch_cpu = torch_result.cpu().contiguous();
        if (lfs_cpu.numel() != static_cast<size_t>(torch_cpu.numel()))
            return false;
        return allclose(lfs_cpu.ptr<float>(), torch_cpu.data_ptr<float>(), lfs_cpu.numel(), atol, rtol);
    }

    bool verify_lfs_pair(const Tensor& a, const Tensor& b, float atol = 1e-4f, float rtol = 1e-4f) {
        auto a_cpu = a.to(Device::CPU);
        auto b_cpu = b.to(Device::CPU);
        if (a_cpu.numel() != b_cpu.numel())
            return false;
        return allclose(a_cpu.ptr<float>(), b_cpu.ptr<float>(), a_cpu.numel(), atol, rtol);
    }

    // Benchmark a pointwise chain: LFS fused, LFS eager, and LibTorch.
    // Input tensor is created once. Only chain execution is timed.
    template <typename LfsFn, typename TorchFn>
    BenchResult run_benchmark(const TensorShape& shape, LfsFn lfs_fn, TorchFn torch_fn) {
        BenchResult result;

        auto lfs_input = Tensor::rand(shape, Device::CUDA, DataType::Float32);

        std::vector<int64_t> torch_shape;
        for (size_t i = 0; i < shape.rank(); ++i)
            torch_shape.push_back(static_cast<int64_t>(shape[i]));
        auto torch_input = torch::rand(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(torch_input.data_ptr<float>()),
                   lfs_input.ptr<float>(),
                   lfs_input.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        Tensor fused_out, eager_out;
        torch::Tensor torch_out;

        // LFS fused (lazy + pointwise fusion)
        {
            LazyTestGuard guard;
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

            for (int i = 0; i < kWarmup; ++i) {
                auto y = lfs_fn(lfs_input).contiguous();
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                fused_out = lfs_fn(lfs_input).contiguous();
                cudaDeviceSynchronize();
            }
            const auto end = std::chrono::steady_clock::now();
            result.lfs_fused_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(kIters);
        }

        // LFS eager (lazy without fusion)
        {
            LazyTestGuard guard;

            for (int i = 0; i < kWarmup; ++i) {
                auto y = lfs_fn(lfs_input).contiguous();
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                eager_out = lfs_fn(lfs_input).contiguous();
                cudaDeviceSynchronize();
            }
            const auto end = std::chrono::steady_clock::now();
            result.lfs_eager_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(kIters);
        }

        // LibTorch
        {
            for (int i = 0; i < kWarmup; ++i) {
                auto y = torch_fn(torch_input);
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                torch_out = torch_fn(torch_input);
                cudaDeviceSynchronize();
            }
            const auto end = std::chrono::steady_clock::now();
            result.torch_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(kIters);
        }

        result.fused_vs_torch = verify_against_torch(fused_out, torch_out);
        result.eager_vs_torch = verify_against_torch(eager_out, torch_out);
        result.fused_vs_eager = verify_lfs_pair(fused_out, eager_out);

        return result;
    }

    // Two-input variant for binary chains (e.g. L1 loss)
    template <typename LfsFn, typename TorchFn>
    BenchResult run_benchmark_binary(const TensorShape& shape, LfsFn lfs_fn, TorchFn torch_fn) {
        BenchResult result;

        auto lfs_a = Tensor::rand(shape, Device::CUDA, DataType::Float32);
        auto lfs_b = Tensor::rand(shape, Device::CUDA, DataType::Float32);

        std::vector<int64_t> torch_shape;
        for (size_t i = 0; i < shape.rank(); ++i)
            torch_shape.push_back(static_cast<int64_t>(shape[i]));
        auto torch_a = torch::rand(torch_shape, torch::kCUDA);
        auto torch_b = torch::rand(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(torch_a.data_ptr<float>()), lfs_a.ptr<float>(),
                   lfs_a.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(const_cast<float*>(torch_b.data_ptr<float>()), lfs_b.ptr<float>(),
                   lfs_b.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        Tensor fused_out, eager_out;
        torch::Tensor torch_out;

        {
            LazyTestGuard guard;
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

            for (int i = 0; i < kWarmup; ++i) {
                auto y = lfs_fn(lfs_a, lfs_b).contiguous();
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                fused_out = lfs_fn(lfs_a, lfs_b).contiguous();
                cudaDeviceSynchronize();
            }
            const auto end = std::chrono::steady_clock::now();
            result.lfs_fused_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(kIters);
        }

        {
            LazyTestGuard guard;
            for (int i = 0; i < kWarmup; ++i) {
                auto y = lfs_fn(lfs_a, lfs_b).contiguous();
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                eager_out = lfs_fn(lfs_a, lfs_b).contiguous();
                cudaDeviceSynchronize();
            }
            const auto end = std::chrono::steady_clock::now();
            result.lfs_eager_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(kIters);
        }

        {
            for (int i = 0; i < kWarmup; ++i) {
                auto y = torch_fn(torch_a, torch_b);
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) {
                torch_out = torch_fn(torch_a, torch_b);
                cudaDeviceSynchronize();
            }
            const auto end = std::chrono::steady_clock::now();
            result.torch_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(kIters);
        }

        result.fused_vs_torch = verify_against_torch(fused_out, torch_out);
        result.eager_vs_torch = verify_against_torch(eager_out, torch_out);
        result.fused_vs_eager = verify_lfs_pair(fused_out, eager_out);

        return result;
    }

} // namespace

TEST(TensorFusionPerformanceTest, PointwiseScalarChain) {
    printf("\n[fusion-perf] === 4-op scalar chain: x.add(1).mul(2).sub(0.5).div(3) ===\n");
    print_header();

    auto lfs_fn = [](const Tensor& x) { return x.add(1.0f).mul(2.0f).sub(0.5f).div(3.0f); };
    auto torch_fn = [](const torch::Tensor& x) { return ((x + 1.0f) * 2.0f - 0.5f) / 3.0f; };

    for (auto [label, n] : std::vector<std::pair<const char*, size_t>>{
             {"1M", 1024 * 1024},
             {"4M (HD 3ch)", 4 * 1024 * 1024},
             {"12M (4K 3ch)", 12 * 1024 * 1024},
             {"50M (large scene)", 50 * 1024 * 1024}}) {
        auto r = run_benchmark({n}, lfs_fn, torch_fn);
        char name[128];
        snprintf(name, sizeof(name), "ScalarChain %s", label);
        print_row(name, r);
        EXPECT_TRUE(r.all_verified());
    }
}

TEST(TensorFusionPerformanceTest, MixedUnaryChain) {
    printf("\n[fusion-perf] === 4-op mixed chain: x.add(1).abs().mul(2).sigmoid() ===\n");
    print_header();

    auto lfs_fn = [](const Tensor& x) { return x.add(1.0f).abs().mul(2.0f).sigmoid(); };
    auto torch_fn = [](const torch::Tensor& x) { return ((x + 1.0f).abs() * 2.0f).sigmoid(); };

    for (auto [label, n] : std::vector<std::pair<const char*, size_t>>{
             {"1M", 1024 * 1024},
             {"4M", 4 * 1024 * 1024},
             {"16M", 16 * 1024 * 1024}}) {
        auto r = run_benchmark({n}, lfs_fn, torch_fn);
        char name[128];
        snprintf(name, sizeof(name), "MixedChain %s", label);
        print_row(name, r);
        EXPECT_TRUE(r.all_verified());
    }
}

TEST(TensorFusionPerformanceTest, BroadcastChain) {
    printf("\n[fusion-perf] === Broadcast chain: ((x + b) * 1.25 - 0.5).sigmoid() ===\n");
    print_header();

    const TensorShape shape = {2048, 1024};
    const Tensor lfs_bias = Tensor::rand({1, 1024}, Device::CUDA, DataType::Float32);
    auto torch_bias = torch::zeros({1, 1024}, torch::kCUDA);
    cudaMemcpy(torch_bias.data_ptr<float>(),
               lfs_bias.ptr<float>(),
               lfs_bias.numel() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    auto lfs_fn = [lfs_bias](const Tensor& x) {
        return x.add(lfs_bias).mul(1.25f).sub(0.5f).sigmoid();
    };
    auto torch_fn = [torch_bias](const torch::Tensor& x) {
        return ((x + torch_bias) * 1.25f - 0.5f).sigmoid();
    };

    const auto r = run_benchmark(shape, lfs_fn, torch_fn);
    print_row("BroadcastChain 2048x1024 + [1x1024]", r);
    EXPECT_TRUE(r.all_verified());
}

TEST(TensorFusionPerformanceTest, TransformReduceSum) {
    printf("\n[fusion-perf] === Transform-reduce: x.add(1).mul(2).sum() ===\n");
    print_header();

    auto lfs_fn = [](const Tensor& x) { return x.add(1.0f).mul(2.0f).sum(); };
    auto torch_fn = [](const torch::Tensor& x) { return ((x + 1.0f) * 2.0f).sum(); };

    for (auto [label, n] : std::vector<std::pair<const char*, size_t>>{
             {"1M", 1024 * 1024},
             {"4M", 4 * 1024 * 1024},
             {"12M", 12 * 1024 * 1024},
             {"50M", 50 * 1024 * 1024}}) {
        auto r = run_benchmark({n}, lfs_fn, torch_fn);
        char name[128];
        snprintf(name, sizeof(name), "TransformReduce-Sum %s", label);
        print_row(name, r);
        EXPECT_TRUE(r.all_verified());
    }
}

TEST(TensorFusionPerformanceTest, TransformReduceMean) {
    printf("\n[fusion-perf] === Transform-reduce: x.mul(2).abs().mean() ===\n");
    print_header();

    auto lfs_fn = [](const Tensor& x) { return x.mul(2.0f).abs().mean(); };
    auto torch_fn = [](const torch::Tensor& x) { return (x * 2.0f).abs().mean(); };

    for (auto [label, n] : std::vector<std::pair<const char*, size_t>>{
             {"1M", 1024 * 1024},
             {"4M", 4 * 1024 * 1024},
             {"12M", 12 * 1024 * 1024},
             {"50M", 50 * 1024 * 1024}}) {
        auto r = run_benchmark({n}, lfs_fn, torch_fn);
        char name[128];
        snprintf(name, sizeof(name), "TransformReduce-Mean %s", label);
        print_row(name, r);
        EXPECT_TRUE(r.all_verified());
    }
}

TEST(TensorFusionPerformanceTest, L1Loss) {
    printf("\n[fusion-perf] === L1 loss: (a - b).abs().mean() ===\n");
    print_header();

    auto lfs_fn = [](const Tensor& a, const Tensor& b) { return (a - b).abs().mean(); };
    auto torch_fn = [](const torch::Tensor& a, const torch::Tensor& b) { return (a - b).abs().mean(); };

    for (auto [label, shape] : std::vector<std::pair<const char*, TensorShape>>{
             {"3x512x512 (768K)", {3, 512, 512}},
             {"3x1024x1024 (3M)", {3, 1024, 1024}},
             {"3x1920x1080 (6M)", {3, 1920, 1080}}}) {
        auto r = run_benchmark_binary(shape, lfs_fn, torch_fn);
        char name[128];
        snprintf(name, sizeof(name), "L1Loss %s", label);
        print_row(name, r);
        EXPECT_TRUE(r.all_verified());
    }
}

TEST(TensorFusionPerformanceTest, LongChain) {
    printf("\n[fusion-perf] === Long chains (16M elements) ===\n");
    print_header();

    const TensorShape shape = {16 * 1024 * 1024};

    {
        auto lfs_fn = [](const Tensor& x) {
            return x.add(1.0f).mul(2.0f).sub(0.5f).abs().exp().sigmoid().sqrt().neg();
        };
        auto torch_fn = [](const torch::Tensor& x) {
            return -(((x + 1.0f) * 2.0f - 0.5f).abs().exp().sigmoid().sqrt());
        };
        auto r = run_benchmark(shape, lfs_fn, torch_fn);
        print_row("8-op chain 16M", r);
        EXPECT_TRUE(r.all_verified());
    }

    {
        auto lfs_fn = [](const Tensor& x) {
            return x.add(1.0f).mul(2.0f).sub(0.5f).abs().sigmoid().add(0.1f).mul(0.5f).neg().abs().exp().sigmoid().sqrt().add(0.01f).mul(1.5f).sub(0.1f).abs();
        };
        auto torch_fn = [](const torch::Tensor& x) {
            auto v = ((x + 1.0f) * 2.0f - 0.5f).abs().sigmoid();
            v = (-(v + 0.1f) * 0.5f).abs().exp().sigmoid().sqrt();
            return ((v + 0.01f) * 1.5f - 0.1f).abs();
        };
        auto r = run_benchmark(shape, lfs_fn, torch_fn);
        print_row("16-op chain 16M", r);
        EXPECT_TRUE(r.all_verified());
    }
}

TEST(TensorFusionPerformanceTest, ErrorMap) {
    printf("\n[fusion-perf] === Error map + reduce: x.neg().add(1).mul(0.5).sum() ===\n");
    print_header();

    auto lfs_fn = [](const Tensor& x) { return x.neg().add(1.0f).mul(0.5f).sum(); };
    auto torch_fn = [](const torch::Tensor& x) { return ((-x) + 1.0f).mul(0.5f).sum(); };

    for (auto [label, shape] : std::vector<std::pair<const char*, TensorShape>>{
             {"3x512x512", {3, 512, 512}},
             {"3x1920x1080", {3, 1920, 1080}}}) {
        auto r = run_benchmark(shape, lfs_fn, torch_fn);
        char name[128];
        snprintf(name, sizeof(name), "ErrorMap %s", label);
        print_row(name, r);
        EXPECT_TRUE(r.all_verified());
    }
}

TEST(TensorFusionPerformanceTest, GaussianActivation) {
    printf("\n[fusion-perf] === GS activation: (-x.abs()).exp().mul(opacity) ===\n");
    print_header();

    auto lfs_fn = [](const Tensor& x) { return x.abs().neg().exp().mul(0.8f); };
    auto torch_fn = [](const torch::Tensor& x) { return (-x.abs()).exp() * 0.8f; };

    for (auto [label, n] : std::vector<std::pair<const char*, size_t>>{
             {"100K splats", 100000},
             {"500K splats", 500000},
             {"2M splats", 2000000}}) {
        auto r = run_benchmark({n}, lfs_fn, torch_fn);
        char name[128];
        snprintf(name, sizeof(name), "GaussianActivation %s", label);
        print_row(name, r);
        EXPECT_TRUE(r.all_verified());
    }
}
