/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace {

    constexpr double kPeakFp16 = 82.6;
    constexpr double kPeakFp32 = 48.0;

    bool benches_enabled() {
        if (std::getenv("LFS_NN_BENCH") != nullptr) {
            return true;
        }
        return false;
    }

    float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    struct BenchResult {
        const char* name;
        const char* dtype;
        float ms = 0.0f;
        double tflops = 0.0;
        double frac = 0.0;
    };

    void print_row(const BenchResult& r) {
        std::printf("%-32s %-6s %10.3f %10.3f %8.2f%%\n", r.name, r.dtype, r.ms, r.tflops,
                    r.frac * 100.0);
    }

    float time_op(const int warmup, const int iters, const std::function<void()>& fn) {
        cudaEvent_t start{};
        cudaEvent_t stop{};
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (int i = 0; i < warmup; ++i) {
            fn();
        }
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            fn();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        const float ms = elapsed_ms(start, stop) / static_cast<float>(iters);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms;
    }

    lfs::core::Tensor randn(const std::vector<std::size_t>& shape, lfs::core::DataType dtype) {
        auto t = lfs::core::Tensor::randn(lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
        if (dtype == lfs::core::DataType::Float16) {
            return t.to(lfs::core::DataType::Float16);
        }
        return t;
    }

} // namespace

TEST(NnBench, DISABLED_ReportTable) {
    if (!benches_enabled() && testing::GTEST_FLAG(also_run_disabled_tests) == false) {
        GTEST_SKIP();
    }
    std::printf("\nNN inference benches (under concurrent load if taken before 16:00)\n");
    std::printf("%-32s %-6s %10s %10s %8s\n", "op", "dtype", "ms", "TFLOP/s", "peak");
    const int warmup = 5;
    const int iters = 20;

    auto bench_gemm = [&](const char* name, int m, int n, int k, bool trans_b,
                          lfs::core::DataType dtype, double peak) {
        auto a = randn({static_cast<std::size_t>(m), static_cast<std::size_t>(k)}, dtype);
        auto b = trans_b ? randn({static_cast<std::size_t>(n), static_cast<std::size_t>(k)}, dtype)
                         : randn({static_cast<std::size_t>(k), static_cast<std::size_t>(n)}, dtype);
        const float ms = time_op(warmup, iters, [&] {
            auto c = lfs::core::nn::gemm(a, b, false, trans_b);
            (void)c;
        });
        const double flops = 2.0 * m * n * k;
        const double tflops = flops / (static_cast<double>(ms) * 1e-3) / 1e12;
        BenchResult row{name, dtype == lfs::core::DataType::Float16 ? "fp16" : "fp32", ms, tflops,
                        tflops / peak};
        print_row(row);
    };

    bench_gemm("gemm M1370 K768 N3072", 1370, 3072, 768, true, lfs::core::DataType::Float32,
               kPeakFp32);
    bench_gemm("gemm M1370 K768 N3072", 1370, 3072, 768, true, lfs::core::DataType::Float16,
               kPeakFp16);
    bench_gemm("gemm M1370 K768 N768", 1370, 768, 768, true, lfs::core::DataType::Float32, kPeakFp32);
    bench_gemm("gemm M1370 K768 N768", 1370, 768, 768, true, lfs::core::DataType::Float16, kPeakFp16);

    auto bench_bmm = [&](lfs::core::DataType dtype, double peak) {
        auto a = randn({12, 1370, 64}, dtype);
        auto b = randn({12, 1370, 64}, dtype);
        const float ms = time_op(warmup, iters, [&] {
            auto c = lfs::core::nn::bmm(a, b, false, true);
            (void)c;
        });
        const double flops = 12.0 * 2.0 * 1370.0 * 1370.0 * 64.0;
        const double tflops = flops / (static_cast<double>(ms) * 1e-3) / 1e12;
        BenchResult row{"bmm 12x1370x64 attn product",
                        dtype == lfs::core::DataType::Float16 ? "fp16" : "fp32", ms, tflops,
                        tflops / peak};
        print_row(row);
    };
    bench_bmm(lfs::core::DataType::Float32, kPeakFp32);
    bench_bmm(lfs::core::DataType::Float16, kPeakFp16);

    auto bench_attn = [&](lfs::core::DataType dtype, double peak) {
        auto q = randn({1, 12, 1370, 64}, dtype);
        auto k = randn({1, 12, 1370, 64}, dtype);
        auto v = randn({1, 12, 1370, 64}, dtype);
        const float ms = time_op(warmup, iters, [&] {
            auto o = lfs::core::nn::attention(q, k, v);
            (void)o;
        });
        const double flops = 4.0 * 1.0 * 12.0 * 1370.0 * 1370.0 * 64.0;
        const double tflops = flops / (static_cast<double>(ms) * 1e-3) / 1e12;
        BenchResult row{"attention B1 H12 N1370 d64",
                        dtype == lfs::core::DataType::Float16 ? "fp16" : "fp32", ms, tflops,
                        tflops / peak};
        print_row(row);
    };
    bench_attn(lfs::core::DataType::Float32, kPeakFp32);
    bench_attn(lfs::core::DataType::Float16, kPeakFp16);

    auto bench_conv = [&](lfs::core::DataType dtype, double peak) {
        auto in = randn({1, 256, 148, 148}, dtype);
        auto w = randn({256, 256, 3, 3}, dtype);
        lfs::core::nn::Conv2dParams p;
        p.pad_h = 1;
        p.pad_w = 1;
        const auto bytes = lfs::core::nn::conv2d_workspace_bytes(in.shape(), w.shape(), p, dtype);
        auto ws = lfs::core::Tensor::empty(
            lfs::core::TensorShape{std::vector<std::size_t>{
                (bytes + lfs::core::dtype_size(dtype) - 1) / lfs::core::dtype_size(dtype)}},
            lfs::core::Device::CUDA, dtype);
        const float ms = time_op(warmup, std::max(iters / 4, 5), [&] {
            auto o = lfs::core::nn::conv2d(in, w, nullptr, p, &ws);
            (void)o;
        });
        const double flops = 2.0 * 1.0 * 256.0 * 148.0 * 148.0 * 256.0 * 3.0 * 3.0;
        const double tflops = flops / (static_cast<double>(ms) * 1e-3) / 1e12;
        BenchResult row{"conv2d 3x3 256ch 148x148",
                        dtype == lfs::core::DataType::Float16 ? "fp16" : "fp32", ms, tflops,
                        tflops / peak};
        print_row(row);
    };
    bench_conv(lfs::core::DataType::Float32, kPeakFp32);
    bench_conv(lfs::core::DataType::Float16, kPeakFp16);

    auto bench_resize = [&](lfs::core::DataType dtype) {
        auto in = randn({1, 3, 256, 256}, dtype);
        const float ms = time_op(warmup, iters, [&] {
            auto o = lfs::core::nn::resize2d(in, 512, 512, lfs::core::nn::ResizeMode::Bilinear,
                                             lfs::core::nn::CoordTransform::HalfPixel);
            (void)o;
        });
        BenchResult row{"resize2d bilinear 256->512",
                        dtype == lfs::core::DataType::Float16 ? "fp16" : "fp32", ms, 0.0, 0.0};
        print_row(row);
    };
    bench_resize(lfs::core::DataType::Float32);
    bench_resize(lfs::core::DataType::Float16);
}
