/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/nn.hpp"
#include "core/nn/nn_nvtx.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <numeric>
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

    std::pair<lfs::core::Tensor, lfs::core::Tensor> load_lpips_bench_pair() {
        namespace fs = std::filesystem;
        const char* input = std::getenv("LFS_LPIPS_BENCH_PNG");
        if (!input || !*input)
            throw std::runtime_error("Set LFS_LPIPS_BENCH_PNG to an evaluation comparison PNG");
        const fs::path path(input);
        auto [data, width, height, channels] = lfs::core::load_image(path);
        constexpr int separator = 4;
        if (!data || channels != 3 || width < 36 || height < 16 || (width - separator) % 2 != 0) {
            if (data)
                lfs::core::free_image(data);
            throw std::runtime_error("Invalid LPIPS benchmark comparison PNG: " + path.string());
        }
        const int image_width = (width - separator) / 2;
        std::vector<float> left(3 * static_cast<std::size_t>(height) * image_width);
        std::vector<float> right(left.size());
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < image_width; ++x) {
                const auto pixel = static_cast<std::size_t>(y) * width + x;
                const auto right_pixel = static_cast<std::size_t>(y) * width + image_width + separator + x;
                for (int c = 0; c < 3; ++c) {
                    left[static_cast<std::size_t>(c) * height * image_width +
                         static_cast<std::size_t>(y) * image_width + x] =
                        static_cast<float>(data[3 * pixel + c]) / 255.0f;
                    right[static_cast<std::size_t>(c) * height * image_width +
                          static_cast<std::size_t>(y) * image_width + x] =
                        static_cast<float>(data[3 * right_pixel + c]) / 255.0f;
                }
            }
        }
        lfs::core::free_image(data);
        const lfs::core::TensorShape shape({1, 3, static_cast<std::size_t>(height), static_cast<std::size_t>(image_width)});
        return {lfs::core::Tensor::from_vector(left, shape, lfs::core::Device::CUDA),
                lfs::core::Tensor::from_vector(right, shape, lfs::core::Device::CUDA)};
    }

    struct Distribution {
        double median = 0.0;
        double p90 = 0.0;
    };

    Distribution timed_lpips(lfs::core::nn::models::Lpips& model,
                             const lfs::core::Tensor& pred,
                             const lfs::core::Tensor& target) {
        const bool profile = std::getenv("LFS_NN_BENCH_PROFILE") != nullptr;
        const int warmups = profile ? 0 : 20;
        const int iterations = profile ? 1 : 100;
        for (int i = 0; i < warmups; ++i) {
            const auto value = model.forward(pred, target);
            if (!value) {
                throw std::runtime_error(std::string(value.error().detail()));
            }
        }
        cudaDeviceSynchronize();
        std::vector<cudaEvent_t> starts(iterations), stops(iterations);
        for (int i = 0; i < iterations; ++i) {
            cudaEventCreate(&starts[i]);
            cudaEventCreate(&stops[i]);
            cudaEventRecord(starts[i]);
            {
                lfs::core::nn::NvtxRange range("benchmark/lpips_forward");
                const auto value = model.forward(pred, target);
                if (!value) {
                    throw std::runtime_error(std::string(value.error().detail()));
                }
            }
            cudaEventRecord(stops[i]);
        }
        cudaEventSynchronize(stops.back());
        std::vector<float> samples;
        samples.reserve(starts.size());
        for (int i = 0; i < iterations; ++i) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, starts[i], stops[i]);
            samples.push_back(ms);
            cudaEventDestroy(starts[i]);
            cudaEventDestroy(stops[i]);
        }
        std::sort(samples.begin(), samples.end());
        if (profile)
            return {samples.front(), samples.front()};
        return {samples[49], samples[89]};
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
    bench_gemm("hiera M65536 K112 N112", 65536, 112, 112, true, lfs::core::DataType::Float16,
               kPeakFp16);
    bench_gemm("hiera M65536 K112 N336", 65536, 336, 112, true, lfs::core::DataType::Float16,
               kPeakFp16);
    bench_gemm("hiera M65536 K112 N448", 65536, 448, 112, true, lfs::core::DataType::Float16,
               kPeakFp16);
    bench_gemm("hiera M4096 K448 N448", 4096, 448, 448, true, lfs::core::DataType::Float16,
               kPeakFp16);
    bench_gemm("hiera M4096 K448 N1792", 4096, 1792, 448, true, lfs::core::DataType::Float16,
               kPeakFp16);
    bench_gemm("hiera M4096 K1792 N448", 4096, 448, 1792, true, lfs::core::DataType::Float16,
               kPeakFp16);

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

TEST(NnBench, DISABLED_LpipsFullResolution) {
    if (!benches_enabled() && testing::GTEST_FLAG(also_run_disabled_tests) == false) {
        GTEST_SKIP();
    }
    const char* weights = std::getenv("LFS_LPIPS_WEIGHTS");
    if (!weights || !*weights || !std::getenv("LFS_LPIPS_BENCH_PNG"))
        GTEST_SKIP() << "Set LFS_LPIPS_WEIGHTS and LFS_LPIPS_BENCH_PNG to run the LPIPS benchmark";
    auto pair = load_lpips_bench_pair();
    const char* const only_dtype = std::getenv("LFS_NN_BENCH_DTYPE");
    for (const auto dtype : {lfs::core::DataType::Float32, lfs::core::DataType::Float16}) {
        const char* const label = dtype == lfs::core::DataType::Float32 ? "fp32" : "fp16";
        if (only_dtype != nullptr && std::strcmp(only_dtype, label) != 0) {
            continue;
        }
        auto loaded = lfs::core::nn::models::Lpips::load(weights, lfs::core::Device::CUDA, dtype);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().detail();
        auto model = std::move(*loaded);
        const auto result = timed_lpips(model, pair.first, pair.second);
        std::printf("LPIPS_BENCH native_%s median_ms=%.6f p90_ms=%.6f arena_bytes=%zu workspace_bytes=%zu\n",
                    label, result.median, result.p90, model.arena_bytes(), model.workspace_bytes());
        model.release_activations();
    }
}

TEST(NnBench, DISABLED_LpipsConv64FullResolution) {
    if (!benches_enabled() && testing::GTEST_FLAG(also_run_disabled_tests) == false) {
        GTEST_SKIP();
    }
    const bool fp32 = std::getenv("LFS_NN_BENCH_FP32") != nullptr;
    const auto dtype = fp32 ? lfs::core::DataType::Float32 : lfs::core::DataType::Float16;
    auto input = randn({1, 64, 822, 1237}, dtype);
    auto weight = randn({64, 64, 3, 3}, dtype);
    lfs::core::nn::Conv2dParams params;
    params.pad_h = 1;
    params.pad_w = 1;
    for (int i = 0; i < 20; ++i) {
        auto output = lfs::core::nn::conv2d(input, weight, nullptr, params);
        (void)output;
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < 5; ++i) {
        lfs::core::nn::NvtxRange range("benchmark/lpips_conv64");
        auto output = lfs::core::nn::conv2d(input, weight, nullptr, params);
        (void)output;
    }
    cudaDeviceSynchronize();
}
