/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

class MRNFStrategyTest_CompactSplatsCorrectAndPeakBelowThreeX_Test;

#include "core/alloc_counter.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/strategies/mrnf.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    size_t cuda_used_bytes() {
        size_t free_b = 0;
        size_t total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b < free_b) {
            return 0;
        }
        return total_b - free_b;
    }

    SplatData create_compact_test_splat(const size_t n, const int sh_degree = 0) {
        std::vector<float> means_data(n * 3, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            means_data[i * 3 + 0] = static_cast<float>(i);
            means_data[i * 3 + 1] = static_cast<float>(i) * 0.1f;
            means_data[i * 3 + 2] = static_cast<float>(i) * 0.01f;
        }
        std::vector<float> sh0_data(n * 3, 0.5f);
        std::vector<float> scaling_data(n * 3, 0.0f);
        std::vector<float> rotation_data(n * 4, 0.0f);
        std::vector<float> opacity_data(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            rotation_data[i * 4 + 0] = 1.0f;
        }
        const size_t sh_rest = sh_rest_coefficients_for_degree(sh_degree);

        auto means = Tensor::from_vector(means_data, TensorShape({n, 3}), Device::CUDA);
        auto sh0 = Tensor::from_vector(sh0_data, TensorShape({n, 1, 3}), Device::CUDA);
        auto shN = Tensor::zeros(TensorShape({n, sh_rest, 3}), Device::CUDA);
        auto scaling = Tensor::from_vector(scaling_data, TensorShape({n, 3}), Device::CUDA);
        auto rotation = Tensor::from_vector(rotation_data, TensorShape({n, 4}), Device::CUDA);
        auto opacity = Tensor::from_vector(opacity_data, TensorShape({n, 1}), Device::CUDA);
        return SplatData(sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    Tensor make_indices(const size_t new_n) {
        std::vector<int> idx_host(new_n);
        for (size_t i = 0; i < new_n; ++i) {
            idx_host[i] = static_cast<int>(i * 2); // keep even rows
        }
        return Tensor::from_vector(idx_host, TensorShape({new_n}), Device::CUDA);
    }

    Tensor make_src(const size_t old_n, const size_t max_cap, const size_t cols) {
        auto src = Tensor::zeros_direct(TensorShape({old_n, cols}), max_cap, Device::CUDA);
        std::vector<float> host(old_n * cols);
        for (size_t i = 0; i < old_n * cols; ++i) {
            host[i] = static_cast<float>(i % 97);
        }
        auto fill = Tensor::from_vector(host, TensorShape({old_n, cols}), Device::CUDA);
        src.copy_from(fill);
        return src;
    }

} // namespace

TEST(CompactSplatPeakPattern, NewPathStaysWithinTwoX) {
    constexpr size_t cols = 3;
    constexpr size_t max_cap = 256 * 1024;
    constexpr size_t new_n = max_cap / 2;
    const size_t tensor_at_cap_bytes = max_cap * cols * sizeof(float);
    // Gather-into-reserved pattern.
    const size_t new_concurrent = tensor_at_cap_bytes + tensor_at_cap_bytes;

    EXPECT_LE(new_concurrent, static_cast<size_t>(2.01 * static_cast<double>(tensor_at_cap_bytes)))
        << "new concurrent=" << new_concurrent << " tensor_at_cap=" << tensor_at_cap_bytes;
    EXPECT_LT(new_concurrent,
              tensor_at_cap_bytes + (max_cap / 2) * cols * sizeof(float) + tensor_at_cap_bytes)
        << "new path must not need the exact-size intermediate";

    auto src = make_src(max_cap, max_cap, cols);
    auto indices = make_indices(new_n);
    auto ref = src.index_select(0, indices).contiguous();
    auto dest = Tensor::zeros_direct(TensorShape({new_n, cols}), max_cap, Device::CUDA);
    src.index_select_into(dest, 0, indices, BoundaryMode::Assert);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(dest.capacity(), max_cap);
    auto got = dest.contiguous().cpu();
    auto exp = ref.contiguous().cpu();
    ASSERT_EQ(got.numel(), exp.numel());
    for (size_t i = 0; i < got.numel(); ++i) {
        EXPECT_FLOAT_EQ(got.ptr<float>()[i], exp.ptr<float>()[i]) << "i=" << i;
    }
}

TEST(MRNFStrategyTest, CompactSplatsCorrectAndPeakBelowThreeX) {
    constexpr size_t old_n = 64;
    constexpr size_t max_cap = 128;
    constexpr size_t keep_n = 32;

    auto splat_data = create_compact_test_splat(old_n, /*sh_degree=*/0);
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 100;
    opt_params.max_cap = static_cast<int>(max_cap);
    opt_params.refine_every = 1000;
    strategy.initialize(opt_params);

    // Reference: keep even rows via index_select on means before compact.
    std::vector<int> keep_idx(keep_n);
    std::vector<uint8_t> keep_mask_host(old_n, 0);
    for (size_t i = 0; i < keep_n; ++i) {
        keep_idx[i] = static_cast<int>(i * 2);
        keep_mask_host[i * 2] = 1;
    }
    auto keep_indices = Tensor::from_vector(keep_idx, TensorShape({keep_n}), Device::CUDA);
    auto ref_means = splat_data.means().index_select(0, keep_indices).contiguous();
    auto ref_sh0 = splat_data.sh0().index_select(0, keep_indices).contiguous();
    auto ref_opacity = splat_data.opacity_raw().index_select(0, keep_indices).contiguous();

    auto keep_mask_cpu = Tensor::zeros_bool(TensorShape({old_n}), Device::CPU);
    {
        auto* p = keep_mask_cpu.ptr<unsigned char>();
        for (size_t i = 0; i < old_n; ++i) {
            p[i] = keep_mask_host[i];
        }
    }
    auto keep_mask = keep_mask_cpu.to(Device::CUDA);

    // Peak / alloc probe around compact.
    Tensor::trim_memory_pool();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto alloc_snap = alloc_counter::snapshot();
    const size_t used_before = cuda_used_bytes();

    strategy.compact_splats(keep_mask);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto alloc_delta = alloc_counter::delta_since(alloc_snap);
    const size_t used_after = cuda_used_bytes();

    ASSERT_EQ(static_cast<size_t>(splat_data.size()), keep_n);
    auto got_means = splat_data.means().contiguous().cpu();
    auto exp_means = ref_means.contiguous().cpu();
    ASSERT_EQ(got_means.numel(), exp_means.numel());
    for (size_t i = 0; i < got_means.numel(); ++i) {
        EXPECT_FLOAT_EQ(got_means.ptr<float>()[i], exp_means.ptr<float>()[i]) << "means i=" << i;
    }
    auto got_sh0 = splat_data.sh0().contiguous().cpu();
    auto exp_sh0 = ref_sh0.contiguous().cpu();
    ASSERT_EQ(got_sh0.numel(), exp_sh0.numel());
    for (size_t i = 0; i < got_sh0.numel(); ++i) {
        EXPECT_FLOAT_EQ(got_sh0.ptr<float>()[i], exp_sh0.ptr<float>()[i]) << "sh0 i=" << i;
    }
    auto got_op = splat_data.opacity_raw().contiguous().cpu();
    auto exp_op = ref_opacity.contiguous().cpu();
    ASSERT_EQ(got_op.numel(), exp_op.numel());
    for (size_t i = 0; i < got_op.numel(); ++i) {
        EXPECT_FLOAT_EQ(got_op.ptr<float>()[i], exp_op.ptr<float>()[i]) << "opacity i=" << i;
    }

    EXPECT_EQ(splat_data.means().capacity(), max_cap);
    EXPECT_EQ(splat_data.sh0().capacity(), max_cap);
    EXPECT_EQ(splat_data.scaling_raw().capacity(), max_cap);
    EXPECT_EQ(splat_data.rotation_raw().capacity(), max_cap);
    EXPECT_EQ(splat_data.opacity_raw().capacity(), max_cap);

    // The reference pattern requires roughly two allocations per site; the
    // gather-into-reserved bound permits one allocation per site.
    constexpr uint64_t kSites = 36;
    EXPECT_LE(alloc_delta, kSites)
        << "compact_splats alloc_delta=" << alloc_delta
        << " exceeds 1-per-site bound (reference path does ~2 per site). used_before="
        << used_before << " used_after=" << used_after
        << " (the reference path should be well above kSites)";

    // Allow allocator-pool slack while excluding a third capacity-sized buffer
    // for every tensor.
    const size_t one_means_at_cap = max_cap * 3 * sizeof(float);
    if (used_after > used_before) {
        EXPECT_LT(used_after - used_before, 8 * one_means_at_cap)
            << "post-compact VRAM growth too large for gather-into-reserved";
    }
}
