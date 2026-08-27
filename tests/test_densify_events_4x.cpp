/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Densify-event kernels: free-slot fuse, packed counts,
 * positive-median normalize, workspace growth helpers.
 */

#include "core/tensor.hpp"
#include "lfs/training/refine_scratch.hpp"
#include "training/kernels/densification_kernels.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/strategies/strategy_utils.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    std::vector<float> to_host(const Tensor& t) {
        return t.cpu().to_vector();
    }

} // namespace

TEST(DensifyEvents4x, FillFreeSlotsFusedWritesAttrsAndZerosAdam) {
    constexpr size_t N = 8;
    constexpr size_t K = 3;

    auto means = Tensor::zeros({N, 3}, Device::CUDA);
    auto rots = Tensor::zeros({N, 4}, Device::CUDA);
    auto scales = Tensor::zeros({N, 3}, Device::CUDA);
    auto sh0 = Tensor::zeros({N, 1, 3}, Device::CUDA);
    auto opac = Tensor::zeros({N}, Device::CUDA);
    auto free_mask = Tensor::zeros_bool({N}, Device::CUDA);
    // Mark slots 1, 3, 5 free.
    {
        auto host = free_mask.cpu();
        auto* p = host.ptr<bool>();
        p[1] = p[3] = p[5] = true;
        free_mask = host.cuda();
    }

    auto src_means = Tensor::from_vector(
        std::vector<float>{10, 11, 12, 20, 21, 22, 30, 31, 32},
        TensorShape({K, 3}), Device::CUDA);
    auto src_rots = Tensor::from_vector(
        std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
        TensorShape({K, 4}), Device::CUDA);
    auto src_scales = Tensor::from_vector(
        std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f},
        TensorShape({K, 3}), Device::CUDA);
    auto src_sh0 = Tensor::from_vector(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
        TensorShape({K, 1, 3}), Device::CUDA);
    auto src_opac = Tensor::from_vector(
        std::vector<float>{0.5f, 0.6f, 0.7f}, TensorShape({K}), Device::CUDA);

    auto targets = Tensor::zeros({K}, Device::CUDA, DataType::Int64);
    {
        const int64_t host_t[3] = {1, 3, 5};
        ASSERT_EQ(cudaMemcpy(targets.ptr<int64_t>(), host_t, K * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    auto adam0 = Tensor::ones({N}, Device::CUDA);
    auto adam1 = Tensor::ones({N}, Device::CUDA);
    float* adam_ptrs[2] = {adam0.ptr<float>(), adam1.ptr<float>()};

    kernels::launch_fill_free_slots_fused(
        targets.ptr<int64_t>(), K,
        src_means.ptr<float>(), src_rots.ptr<float>(), src_scales.ptr<float>(),
        src_sh0.ptr<float>(), src_opac.ptr<float>(),
        means.ptr<float>(), rots.ptr<float>(), scales.ptr<float>(),
        sh0.ptr<float>(), opac.ptr<float>(),
        /*opacity_dim=*/0, adam_ptrs, 2, free_mask.ptr<bool>(), N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto mh = to_host(means);
    EXPECT_FLOAT_EQ(mh[1 * 3 + 0], 10.f);
    EXPECT_FLOAT_EQ(mh[3 * 3 + 1], 21.f);
    EXPECT_FLOAT_EQ(mh[5 * 3 + 2], 32.f);
    EXPECT_FLOAT_EQ(mh[0], 0.f); // untouched

    auto ah0 = to_host(adam0);
    auto ah1 = to_host(adam1);
    EXPECT_FLOAT_EQ(ah0[1], 0.f);
    EXPECT_FLOAT_EQ(ah0[3], 0.f);
    EXPECT_FLOAT_EQ(ah0[0], 1.f);
    EXPECT_FLOAT_EQ(ah1[5], 0.f);

    auto fm = free_mask.cpu();
    EXPECT_FALSE(fm.ptr<bool>()[1]);
    EXPECT_FALSE(fm.ptr<bool>()[3]);
    EXPECT_FALSE(fm.ptr<bool>()[5]);
    EXPECT_FALSE(fm.ptr<bool>()[0]);
}

TEST(DensifyEvents4x, PackedRefineCountsMatchHost) {
    constexpr size_t N = 16;
    std::vector<uint8_t> b0(N, 0), b1(N, 0);
    std::vector<float> f0(N, 0.f), f1(N, 0.f);
    b0[0] = b0[2] = b0[4] = 1; // 3
    b1[1] = b1[3] = 1;         // 2
    f0[0] = 1.f;
    f0[1] = -1.f;
    f0[5] = 0.5f; // 2 positives
    f1[7] = 3.f;  // 1 positive

    auto bool0 = Tensor::zeros_bool({N}, Device::CUDA);
    auto bool1 = Tensor::zeros_bool({N}, Device::CUDA);
    {
        auto h0 = bool0.cpu();
        auto h1 = bool1.cpu();
        for (size_t i = 0; i < N; ++i) {
            h0.ptr<bool>()[i] = b0[i] != 0;
            h1.ptr<bool>()[i] = b1[i] != 0;
        }
        bool0 = h0.cuda();
        bool1 = h1.cuda();
    }
    auto tf0 = Tensor::from_vector(f0, TensorShape({N}), Device::CUDA);
    auto tf1 = Tensor::from_vector(f1, TensorShape({N}), Device::CUDA);
    auto out = Tensor::zeros({4}, Device::CUDA, DataType::Int64);

    kernels::launch_packed_refine_counts(
        bool0.ptr<bool>(), N, bool1.ptr<bool>(), N,
        tf0.ptr<float>(), N, tf1.ptr<float>(), N,
        out.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int64_t counts[4] = {};
    ASSERT_EQ(cudaMemcpy(counts, out.ptr<int64_t>(), 4 * sizeof(int64_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(counts[0], 3);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(counts[2], 2);
    EXPECT_EQ(counts[3], 1);
}

TEST(DensifyEvents4x, PositiveMedianNormalizeMatchesSortReference) {
    // Values: positives 1,2,3,4,5 → median 3; zeros stay 0; result /3
    std::vector<float> data = {0.f, 1.f, 0.f, 5.f, 2.f, 4.f, 3.f, 0.f, -1.f};
    auto t = Tensor::from_vector(data, TensorShape({data.size()}), Device::CUDA);

    // Reference: positives only sorted mid
    std::vector<float> pos;
    for (float v : data)
        if (v > 0.f)
            pos.push_back(v);
    std::sort(pos.begin(), pos.end());
    const float median = pos[pos.size() / 2];
    std::vector<float> ref = data;
    for (float& v : ref)
        v /= std::max(median, 1e-9f);

    kernels::launch_normalize_by_positive_median(t.ptr<float>(), t.numel());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto got = to_host(t);
    ASSERT_EQ(got.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-5f) << "i=" << i;
    }
}

TEST(DensifyEvents4x, PositiveMedianWorkspaceMatchesAllocatingPath) {
    std::vector<float> data = {0.f, 1.f, 0.f, 5.f, 2.f, 4.f, 3.f, 0.f, -1.f};
    auto allocating = Tensor::from_vector(data, TensorShape({data.size()}), Device::CUDA);
    auto persisted = Tensor::from_vector(data, TensorShape({data.size()}), Device::CUDA);

    kernels::launch_normalize_by_positive_median(allocating.ptr<float>(), allocating.numel());
    PositiveMedianScratch scratch;
    scratch.ensure_n(data.size(), Device::CUDA);
    kernels::launch_normalize_by_positive_median(
        persisted.ptr<float>(), persisted.numel(), nullptr, &scratch);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto got_a = to_host(allocating);
    const auto got_b = to_host(persisted);
    ASSERT_EQ(got_a.size(), got_b.size());
    for (size_t i = 0; i < got_a.size(); ++i) {
        EXPECT_NEAR(got_a[i], got_b[i], 1e-5f) << "i=" << i;
    }
}

TEST(DensifyEvents4x, PositiveMedianWorkspaceZerosWhenNoPositives) {
    std::vector<float> data = {0.f, -1.f, 0.f, -2.f};
    auto t = Tensor::from_vector(data, TensorShape({data.size()}), Device::CUDA);
    PositiveMedianScratch scratch;
    scratch.ensure_n(data.size(), Device::CUDA);
    kernels::launch_normalize_by_positive_median(t.ptr<float>(), t.numel(), nullptr, &scratch);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (float v : to_host(t)) {
        EXPECT_FLOAT_EQ(v, 0.f);
    }
}

TEST(DensifyEvents4x, GumbelScratchMatchesAllocatingPath) {
    constexpr size_t n = 64;
    constexpr size_t k = 7;
    std::vector<float> weights(n, 0.f);
    size_t nnz = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % 3 != 0) {
            weights[i] = static_cast<float>(i + 1);
            ++nnz;
        }
    }
    auto w = Tensor::from_vector(weights, TensorShape({n}), Device::CUDA);
    auto allocating = Tensor::empty({k}, Device::CUDA, DataType::Int64);
    auto persisted = Tensor::empty({k}, Device::CUDA, DataType::Int64);
    constexpr uint64_t seed = 0xC0FFEEULL;

    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), n, k, seed, allocating.ptr<int64_t>());
    GumbelTopKScratch scratch;
    scratch.ensure_n(n, Device::CUDA);
    mrnf_strategy::launch_gumbel_topk(
        w.ptr<float>(), n, k, seed, persisted.ptr<int64_t>(), nullptr, true, &scratch, nnz);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto a = allocating.cpu().to_vector_int64();
    const auto b = persisted.cpu().to_vector_int64();
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "i=" << i;
    }
}

TEST(DensifyEvents4x, GumbelScratchDenseMatchesAllocatingPath) {
    constexpr size_t n = 32;
    constexpr size_t k = 5;
    std::vector<float> weights(n);
    for (size_t i = 0; i < n; ++i) {
        weights[i] = 0.25f + static_cast<float>(i);
    }
    auto w = Tensor::from_vector(weights, TensorShape({n}), Device::CUDA);
    auto allocating = Tensor::empty({k}, Device::CUDA, DataType::Int64);
    auto persisted = Tensor::empty({k}, Device::CUDA, DataType::Int64);
    constexpr uint64_t seed = 42;

    mrnf_strategy::launch_gumbel_topk(
        w.ptr<float>(), n, k, seed, allocating.ptr<int64_t>(), nullptr, false);
    GumbelTopKScratch scratch;
    scratch.ensure_n(n, Device::CUDA);
    mrnf_strategy::launch_gumbel_topk(
        w.ptr<float>(), n, k, seed, persisted.ptr<int64_t>(), nullptr, false, &scratch, n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto a = allocating.cpu().to_vector_int64();
    const auto b = persisted.cpu().to_vector_int64();
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "i=" << i;
    }
}

TEST(DensifyEvents4x, DensifyChildWorkspaceGrowsOnly) {
    DensifyChildWorkspace ws;
    ws.ensure(100, 0, false, false, Device::CUDA);
    const size_t cap1 = ws.capacity;
    EXPECT_GE(cap1, 100u);
    auto m1 = ws.means.ptr<float>();

    ws.ensure(50, 0, false, false, Device::CUDA); // smaller — no realloc
    EXPECT_EQ(ws.capacity, cap1);
    EXPECT_EQ(ws.means.ptr<float>(), m1);

    ws.ensure(cap1 + 10, 0, false, false, Device::CUDA); // grow
    EXPECT_GT(ws.capacity, cap1);
}

TEST(DensifyEvents4x, ScoreBufferAppendZerosInPlace) {
    Tensor scores = Tensor::zeros_direct(TensorShape({4}), /*capacity=*/16, Device::CUDA);
    std::vector<float> init = {1.f, 2.f, 3.f, 4.f};
    ASSERT_EQ(cudaMemcpy(scores.ptr<float>(), init.data(), 4 * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ensure_score_buffer_inplace(scores, 10, Device::CUDA, 16);
    EXPECT_EQ(scores.numel(), 10u);
    EXPECT_GE(scores.capacity(), 16u);
    auto v = to_host(scores);
    EXPECT_FLOAT_EQ(v[0], 1.f);
    EXPECT_FLOAT_EQ(v[3], 4.f);
    EXPECT_FLOAT_EQ(v[4], 0.f);
    EXPECT_FLOAT_EQ(v[9], 0.f);
}

TEST(DensifyEvents4x, DensificationInfoReusesMatchingShape) {
    Tensor info = Tensor::zeros({2, 32}, Device::CUDA);
    float* p0 = info.ptr<float>();
    ensure_densification_info_shape_inplace(info, 32, Device::CUDA);
    EXPECT_EQ(info.ptr<float>(), p0); // same storage
    ensure_densification_info_shape_inplace(info, 40, Device::CUDA);
    EXPECT_EQ(info.shape()[1], 40u);
    EXPECT_NE(info.ptr<float>(), p0); // reallocated for new N
}
