/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Fused densification-info fold-and-zero tests.
 * The fused operation must match separate max/add and zero operations.
 */

#include "core/tensor.hpp"
#include "training/kernels/mcmc_kernels.hpp"
#include "training/kernels/mrnf_kernels.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    Tensor make_info(const std::vector<float>& row0, const std::vector<float>& row1) {
        const size_t n = row0.size();
        std::vector<float> flat(n * 2);
        for (size_t i = 0; i < n; ++i) {
            flat[i] = row0[i];
            flat[n + i] = row1[i];
        }
        return Tensor::from_vector(flat, {size_t{2}, n}, Device::CUDA);
    }

    std::vector<float> to_host(const Tensor& t) {
        return t.cpu().to_vector();
    }

} // namespace

TEST(DensificationInfoZeroTest, MrnfFoldMatchesMultiStepReference) {
    constexpr size_t N = 8;
    auto vis = Tensor::zeros({N}, Device::CUDA);
    auto refine_max = Tensor::zeros({N}, Device::CUDA);

    // Reference path: separate max/add + zero each step.
    auto vis_ref = Tensor::zeros({N}, Device::CUDA);
    auto refine_ref = Tensor::zeros({N}, Device::CUDA);

    const std::vector<std::pair<std::vector<float>, std::vector<float>>> steps = {
        {{1, 0, 2, 0, 0, 3, 0, 0}, {0.5f, 0, 1.0f, 0, 0, 0.2f, 0, 0}},
        {{0, 4, 0, 1, 0, 0, 2, 0}, {0.1f, 2.0f, 0, 0.3f, 0, 0, 1.5f, 0}},
        {{1, 1, 1, 1, 1, 1, 1, 1}, {9, 8, 7, 6, 5, 4, 3, 2}},
    };

    for (const auto& [r0, r1] : steps) {
        auto info = make_info(r0, r1);
        auto info_ref = info.clone();

        mrnf_strategy::launch_fold_densification_and_zero(
            vis.ptr<float>(),
            refine_max.ptr<float>(),
            info.ptr<float>(),
            N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        mcmc::launch_elementwise_max_inplace(
            refine_ref.ptr<float>(),
            info_ref.ptr<float>() + N,
            N);
        mrnf_strategy::launch_elementwise_add_inplace(
            vis_ref.ptr<float>(),
            info_ref.ptr<float>(),
            N);
        info_ref.zero_();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        auto info_h = to_host(info);
        for (float v : info_h) {
            EXPECT_FLOAT_EQ(v, 0.f) << "densification_info must be zeroed after fold";
        }
    }

    auto vis_h = to_host(vis);
    auto vis_ref_h = to_host(vis_ref);
    auto ref_h = to_host(refine_max);
    auto ref_ref_h = to_host(refine_ref);
    ASSERT_EQ(vis_h.size(), N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(vis_h[i], vis_ref_h[i]) << "vis i=" << i;
        EXPECT_FLOAT_EQ(ref_h[i], ref_ref_h[i]) << "refine i=" << i;
    }
}

TEST(DensificationInfoZeroTest, McmcMaxMatchesMultiStepReference) {
    constexpr size_t N = 6;
    auto err_max = Tensor::zeros({N}, Device::CUDA);
    auto err_ref = Tensor::zeros({N}, Device::CUDA);

    const std::vector<std::vector<float>> error_rows = {
        {0.1f, 0, 0.5f, 0, 2.0f, 0},
        {0.2f, 1.0f, 0.4f, 0.1f, 0.5f, 3.0f},
        {0, 0, 0, 0, 0, 0},
        {5, 0, 0, 0, 0, 0.01f},
    };

    for (const auto& err : error_rows) {
        std::vector<float> r0(N, 0.f); // unused by MCMC fold but zeroed too
        auto info = make_info(r0, err);
        auto info_ref = info.clone();

        mcmc::launch_max_error_and_zero_densification(
            err_max.ptr<float>(),
            info.ptr<float>(),
            N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        mcmc::launch_elementwise_max_inplace(
            err_ref.ptr<float>(),
            info_ref.ptr<float>() + N,
            N);
        info_ref.zero_();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        for (float v : to_host(info)) {
            EXPECT_FLOAT_EQ(v, 0.f);
        }
    }

    auto a = to_host(err_max);
    auto b = to_host(err_ref);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "error max i=" << i;
    }
}
