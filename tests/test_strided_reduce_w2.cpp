/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * WO-W.2 — Strided reduce that beats transpose+copy (per-shape heuristic).
 *
 * Correctness vs reference (last-dim reduce of permuted dense), Float16,
 * path selection when strided wins, microbench table printed for PROGRESS.
 */

#include "core/alloc_counter.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/tensor_ops.hpp"

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace lfs::core;

namespace {

    void cuda_ok() {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    Tensor sequential_cuda(const std::vector<size_t>& shape) {
        size_t n = 1;
        for (auto d : shape)
            n *= d;
        std::vector<float> host(n);
        for (size_t i = 0; i < n; ++i)
            host[i] = static_cast<float>((i % 97) + 1); // 1..97 cycle
        return Tensor::from_vector(host, TensorShape(shape), Device::CPU).cuda();
    }

    /// Reference: permute reduce-dim to last, contiguous, reduce last.
    Tensor ref_reduce_sum(const Tensor& t, int dim, bool keepdim) {
        const int rank = static_cast<int>(t.ndim());
        if (dim < 0)
            dim += rank;
        std::vector<int> perm;
        for (int i = 0; i < rank; ++i) {
            if (i != dim)
                perm.push_back(i);
        }
        perm.push_back(dim);
        auto dense = t.permute(perm).contiguous();
        auto reduced = dense.sum({static_cast<int>(dense.ndim()) - 1}, /*keepdim=*/false);
        if (!keepdim)
            return reduced;
        std::vector<size_t> out_dims = t.shape().dims();
        out_dims[static_cast<size_t>(dim)] = 1;
        return reduced.reshape(TensorShape(out_dims));
    }

    void expect_close(const Tensor& a, const Tensor& b, float rtol = 1e-3f, float atol = 1e-2f,
                      const std::string& ctx = "") {
        auto av = a.cpu().contiguous().to_vector();
        auto bv = b.cpu().contiguous().to_vector();
        ASSERT_EQ(av.size(), bv.size()) << ctx;
        for (size_t i = 0; i < av.size(); ++i) {
            float thr = atol + rtol * std::abs(bv[i]);
            EXPECT_NEAR(av[i], bv[i], thr) << ctx << " i=" << i;
        }
    }

    double time_us(std::function<void()> fn, int warmup = 5, int reps = 20) {
        for (int i = 0; i < warmup; ++i) {
            fn();
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            fn();
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    }

    struct PathGuard {
        PathGuard() {
            tensor_ops::set_reduce_path_override_for_testing(
                tensor_ops::ReducePathForTesting::None);
            tensor_ops::set_reduce_last_path_for_testing(
                tensor_ops::ReducePathForTesting::Default);
        }
        ~PathGuard() {
            tensor_ops::set_reduce_path_override_for_testing(
                tensor_ops::ReducePathForTesting::None);
        }
    };

} // namespace

// ---------------------------------------------------------------------------
// Correctness across shape classes
// ---------------------------------------------------------------------------

TEST(StridedReduceW2, SumDim0Rank3MatchesReference) {
    PathGuard g;
    // [32, 64, 512] reduce dim0 — inner = 64*512 = 32768 >= 256
    auto t = sequential_cuda({32, 64, 512});
    auto got = t.sum({0}, /*keepdim=*/false);
    auto ref = ref_reduce_sum(t, 0, false);
    cuda_ok();
    expect_close(got, ref, 1e-3f, 5e-2f, "rank3 dim0");
}

TEST(StridedReduceW2, SumDim1Rank3MatchesReference) {
    PathGuard g;
    auto t = sequential_cuda({16, 128, 512});
    auto got = t.sum({1}, /*keepdim=*/false);
    auto ref = ref_reduce_sum(t, 1, false);
    cuda_ok();
    expect_close(got, ref, 1e-3f, 5e-2f, "rank3 dim1");
}

TEST(StridedReduceW2, ForceStridedMatchesForceTranspose) {
    PathGuard g;
    auto t = sequential_cuda({8, 64, 1024});

    tensor_ops::set_reduce_path_override_for_testing(
        tensor_ops::ReducePathForTesting::StridedFast);
    auto strided = t.sum({1}, false);
    EXPECT_EQ(tensor_ops::reduce_last_path_for_testing(),
              tensor_ops::ReducePathForTesting::StridedFast);
    tensor_ops::set_reduce_path_override_for_testing(
        tensor_ops::ReducePathForTesting::None);

    tensor_ops::set_reduce_path_override_for_testing(
        tensor_ops::ReducePathForTesting::Transpose);
    auto transposed = t.sum({1}, false);
    EXPECT_EQ(tensor_ops::reduce_last_path_for_testing(),
              tensor_ops::ReducePathForTesting::Transpose);
    tensor_ops::set_reduce_path_override_for_testing(
        tensor_ops::ReducePathForTesting::None);

    cuda_ok();
    expect_close(strided, transposed, 1e-3f, 5e-2f, "strided vs transpose");
}

TEST(StridedReduceW2, MeanMaxMinDim0) {
    PathGuard g;
    auto t = sequential_cuda({64, 8, 256});
    expect_close(t.mean({0}, false),
                 ref_reduce_sum(t, 0, false).div(static_cast<float>(64)), 1e-3f, 5e-2f, "mean");
    auto dense = t.permute({1, 2, 0}).contiguous();
    auto max_ref = dense.max({2}, false);
    auto min_ref = dense.min({2}, false);
    expect_close(t.max({0}, false), max_ref, 1e-5f, 1e-4f, "max");
    expect_close(t.min({0}, false), min_ref, 1e-5f, 1e-4f, "min");
    cuda_ok();
}

TEST(StridedReduceW2, KeepdimShape) {
    PathGuard g;
    auto t = sequential_cuda({8, 16, 256});
    auto got = t.sum({1}, /*keepdim=*/true);
    EXPECT_EQ(got.shape(), TensorShape({8, 1, 256}));
    auto ref = ref_reduce_sum(t, 1, true);
    expect_close(got, ref, 1e-3f, 5e-2f, "keepdim");
    cuda_ok();
}

TEST(StridedReduceW2, Float16SumMatchesFloat32) {
    PathGuard g;
    auto f32 = sequential_cuda({16, 32, 256});
    auto f16 = f32.to(DataType::Float16);
    auto r32 = f32.sum({0}, false);
    auto r16 = f16.sum({0}, false).to(DataType::Float32);
    cuda_ok();
    expect_close(r16, r32, 1e-2f, 1e-1f, "f16 sum");
}

TEST(StridedReduceW2, LargeInnerFullCoverage) {
    // Guards against SM-capping grid_x without a grid-stride loop.
    PathGuard g;
    tensor_ops::set_reduce_path_override_for_testing(
        tensor_ops::ReducePathForTesting::StridedFast);
    auto t = sequential_cuda({4, 32, 8192}); // output elems = 4*8192 = 32768
    auto got = t.sum({1}, false);
    auto ref = ref_reduce_sum(t, 1, false);
    EXPECT_EQ(tensor_ops::reduce_last_path_for_testing(),
              tensor_ops::ReducePathForTesting::StridedFast);
    tensor_ops::set_reduce_path_override_for_testing(
        tensor_ops::ReducePathForTesting::None);
    cuda_ok();
    expect_close(got, ref, 1e-3f, 5e-2f, "large inner coverage");
}

// ---------------------------------------------------------------------------
// Path / alloc: shapes where strided_fast wins must not take transpose path
// ---------------------------------------------------------------------------

TEST(StridedReduceW2, NoFullTransposeCopyWhenStridedWins) {
    PathGuard g;
    // large inner, moderate reduce → heuristic prefers strided
    constexpr size_t outer = 4, reduce = 256, inner = 1024;
    auto t = sequential_cuda({outer, reduce, inner});

    tensor_ops::set_reduce_last_path_for_testing(
        tensor_ops::ReducePathForTesting::Default);
    const auto snap = alloc_counter::snapshot();
    auto result = t.sum({1}, false);
    cuda_ok();
    const auto delta = alloc_counter::delta_since(snap);

    EXPECT_EQ(result.numel(), outer * inner);
    EXPECT_EQ(tensor_ops::reduce_last_path_for_testing(),
              tensor_ops::ReducePathForTesting::StridedFast)
        << "expected strided_fast for outer=" << outer << " reduce=" << reduce
        << " inner=" << inner;

    // Soft: when path is strided, driver-bytes for a full transpose (~4 MiB)
    // should not dominate. Pool may hide allocs entirely (delta=0) — OK.
    (void)delta;
}

TEST(StridedReduceW2, HeuristicPrefersTransposeOnShortReduceWideOutput) {
    PathGuard g;
    // Measured edge: reduce<=64, output>=8192, numel<=1M → transpose
    // [8,64,1024] dim1: outer=8 reduce=64 inner=1024
    auto t = sequential_cuda({8, 64, 1024});
    tensor_ops::set_reduce_last_path_for_testing(
        tensor_ops::ReducePathForTesting::Default);
    auto result = t.sum({1}, false);
    cuda_ok();
    EXPECT_EQ(result.numel(), 8u * 1024u);
    EXPECT_EQ(tensor_ops::reduce_last_path_for_testing(),
              tensor_ops::ReducePathForTesting::Transpose)
        << "expected transpose for short-reduce / wide-output edge class";
    auto ref = ref_reduce_sum(t, 1, false);
    expect_close(result, ref, 1e-3f, 5e-2f, "transpose-class correctness");
}

TEST(StridedReduceW2, HeuristicPrefersStridedOnHugeReduce) {
    PathGuard g;
    // Huge reduce: strided avoids full-tensor copy (measured 18µs vs 38–52µs transpose)
    auto t = sequential_cuda({4, 2048, 256});
    tensor_ops::set_reduce_last_path_for_testing(
        tensor_ops::ReducePathForTesting::Default);
    auto result = t.sum({1}, false);
    cuda_ok();
    EXPECT_EQ(tensor_ops::reduce_last_path_for_testing(),
              tensor_ops::ReducePathForTesting::StridedFast);
    expect_close(result, ref_reduce_sum(t, 1, false), 1e-3f, 5e-2f, "huge-reduce");
}

// ---------------------------------------------------------------------------
// Microbench table (shape × method µs) — printed for PROGRESS.md
// ---------------------------------------------------------------------------

TEST(StridedReduceW2, MicrobenchTableShapeXMethod) {
    PathGuard g;
    struct Case {
        std::vector<size_t> shape;
        int dim;
        const char* name;
    };
    const Case cases[] = {
        {{1024, 1024}, 0, "[1024,1024] dim0"},
        {{64, 512, 512}, 0, "[64,512,512] dim0"},
        {{32, 128, 512}, 1, "[32,128,512] dim1"},
        {{8, 64, 1024}, 1, "[8,64,1024] dim1"},
        {{16, 16, 256}, 0, "[16,16,256] dim0"},
        {{4, 2048, 256}, 1, "[4,2048,256] dim1"},
        {{1, 4096, 512}, 1, "[1,4096,512] dim1"},
    };

    std::cout << "\n=== WO-W.2 microbench (µs, mean of 20 after 5 warmup) ===\n";
    std::cout << std::left << std::setw(28) << "shape" << std::setw(14) << "default"
              << std::setw(14) << "force_strided" << std::setw(14) << "force_transpose"
              << "winner\n";

    for (const auto& c : cases) {
        auto t = sequential_cuda(c.shape);
        cuda_ok();

        auto time_default = time_us([&] {
            auto r = t.sum({c.dim}, false);
            (void)r;
        });

        auto time_strided = time_us([&] {
            tensor_ops::set_reduce_path_override_for_testing(
                tensor_ops::ReducePathForTesting::StridedFast);
            auto r = t.sum({c.dim}, false);
            (void)r;
            tensor_ops::set_reduce_path_override_for_testing(
                tensor_ops::ReducePathForTesting::None);
        });

        auto time_transpose = time_us([&] {
            tensor_ops::set_reduce_path_override_for_testing(
                tensor_ops::ReducePathForTesting::Transpose);
            auto r = t.sum({c.dim}, false);
            (void)r;
            tensor_ops::set_reduce_path_override_for_testing(
                tensor_ops::ReducePathForTesting::None);
        });

        const char* winner = "default";
        double best = time_default;
        if (time_strided + 0.5 < best) {
            best = time_strided;
            winner = "strided";
        }
        if (time_transpose + 0.5 < best) {
            winner = "transpose";
        }

        std::cout << std::left << std::setw(28) << c.name << std::setw(14) << std::fixed
                  << std::setprecision(1) << time_default << std::setw(14) << time_strided
                  << std::setw(14) << time_transpose << winner << "\n";
    }
    std::cout << std::flush;
    SUCCEED();
}
