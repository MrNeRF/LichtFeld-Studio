/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Standalone tensor launch-overhead bench for Phase 6B-2 P2 (§9 tensor-bench pin).
// Depends only on the public tensor API so the identical source builds on
// both A/B sides. Built alongside tests, NOT registered with ctest.
//
// Diagnostics off: do not set LFS_CUDA_SYNC_DEBUG.
// Output: BENCH,<case>,<median_ms> lines (machine-greppable).

#include "core/tensor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    constexpr int kReps = 5;

    [[nodiscard]] double median_ms(std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        const size_t n = samples.size();
        if (n == 0) {
            return 0.0;
        }
        if (n % 2 == 1) {
            return samples[n / 2];
        }
        return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    }

    void sync_device() {
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
            std::exit(2);
        }
    }

    template <typename Fn>
    [[nodiscard]] double time_ms(Fn&& fn) {
        sync_device();
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        sync_device();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    void print_bench(const char* name, double median) {
        // Machine-greppable: BENCH,<case>,<median_ms>
        std::printf("BENCH,%s,%.6f\n", name, median);
        std::fflush(stdout);
    }

    // (a) sort along middle axis of [64, 1024, 64] float —
    // exercises launch_sort_2d nested host loop (~8192 checked launches).
    void bench_sort_middle_axis() {
        auto t = Tensor::randn({64, 1024, 64}, Device::CUDA);
        // Warmup
        (void)t.sort(/*dim=*/1, /*descending=*/false);
        sync_device();

        std::vector<double> samples;
        samples.reserve(kReps);
        for (int i = 0; i < kReps; ++i) {
            // Fresh-ish data each rep to avoid pure cache effects; sort is in-place
            // on a clone so the source stays unsorted.
            samples.push_back(time_ms([&] {
                auto work = t.clone();
                (void)work.sort(/*dim=*/1, /*descending=*/false);
            }));
        }
        print_bench("sort_middle_axis_64x1024x64", median_ms(std::move(samples)));
    }

    // (b) broadcast_binary add on broadcast-mismatched shapes
    // ([2,3,4] + [2,1,4] → generic broadcast_binary_kernel family).
    void bench_broadcast_binary_add() {
        auto a = Tensor::randn({64, 128, 64}, Device::CUDA);
        auto b = Tensor::randn({64, 1, 64}, Device::CUDA);
        // Warmup
        (void)a.add(b);
        sync_device();

        std::vector<double> samples;
        samples.reserve(kReps);
        for (int i = 0; i < kReps; ++i) {
            samples.push_back(time_ms([&] {
                // Multiple launches per rep so median is stable above timer noise.
                for (int k = 0; k < 32; ++k) {
                    (void)a.add(b);
                }
            }));
        }
        print_bench("broadcast_binary_add", median_ms(std::move(samples)));
    }

    // (c) tight loop of small masking-op launches (per-launch overhead floor).
    void bench_masking_op_loop() {
        auto x = Tensor::randn({256}, Device::CUDA);
        auto y = Tensor::randn({256}, Device::CUDA);
        // Warmup
        auto cond = x.gt(0.0f);
        (void)Tensor::where(cond, x, y);
        sync_device();

        std::vector<double> samples;
        samples.reserve(kReps);
        for (int i = 0; i < kReps; ++i) {
            samples.push_back(time_ms([&] {
                for (int k = 0; k < 256; ++k) {
                    auto c = x.gt(0.0f);
                    (void)Tensor::where(c, x, y);
                }
            }));
        }
        print_bench("masking_ops_tight_loop", median_ms(std::move(samples)));
    }

    void bench_eager_binary_dispatch_tight_loop() {
        auto a = Tensor::randn({16}, Device::CUDA);
        auto b = Tensor::randn({16}, Device::CUDA);
        Tensor c;
        for (int k = 0; k < 64; ++k) {
            c = a.add(b);
        }
        sync_device();

        std::vector<double> samples;
        samples.reserve(kReps);
        for (int i = 0; i < kReps; ++i) {
            samples.push_back(time_ms([&] {
                for (int k = 0; k < 4096; ++k) {
                    c = a.add(b);
                }
            }));
        }
        print_bench("eager_binary_dispatch_tight_loop", median_ms(std::move(samples)));
    }

    void bench_deferred_alias_copy_materialize() {
        auto source = Tensor::randn({262144}, Device::CUDA);
        const auto run = [&] {
            Tensor deferred = source.add(1.0f).mul(2.0f).sub(3.0f);
            std::array<Tensor, 4> aliases{deferred, deferred, deferred, deferred};
            (void)deferred.data_ptr();
            for (const Tensor& alias : aliases) {
                (void)alias.data_ptr();
            }
        };

        run();
        sync_device();

        std::vector<double> samples;
        samples.reserve(kReps);
        for (int i = 0; i < kReps; ++i) {
            samples.push_back(time_ms(run));
        }
        print_bench("deferred_alias_copy_materialize", median_ms(std::move(samples)));
    }

} // namespace

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device available\n");
        return 1;
    }

    bench_sort_middle_axis();
    bench_broadcast_binary_add();
    bench_masking_op_loop();
    bench_eager_binary_dispatch_tight_loop();
    bench_deferred_alias_copy_materialize();
    return 0;
}
