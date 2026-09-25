/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Fused densification-info fold-and-zero tests.
 * The fused operation must match separate max/add and zero operations.
 */

#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "training/kernels/mcmc_kernels.hpp"
#include "training/kernels/mrnf_kernels.hpp"

#include <algorithm>
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
        return Tensor::from_vector(flat, {size_t{2}, n}, Device::GPU);
    }

    std::vector<float> to_host(const Tensor& t) {
        return t.cpu().to_vector();
    }

} // namespace

class DensificationInfoZeroTest : public lfs::test::CudaBackendTest {};

TEST_F(DensificationInfoZeroTest, MrnfFoldMatchesMultiStepReference) {
    constexpr size_t N = 8;
    auto vis = Tensor::zeros({N}, Device::GPU);
    auto refine_max = Tensor::zeros({N}, Device::GPU);

    std::vector<float> vis_ref(N, 0.f);
    std::vector<float> refine_ref(N, 0.f);

    const std::vector<std::pair<std::vector<float>, std::vector<float>>> steps = {
        {{1, 0, 2, 0, 0, 3, 0, 0}, {0.5f, 0, 1.0f, 0, 0, 0.2f, 0, 0}},
        {{0, 4, 0, 1, 0, 0, 2, 0}, {0.1f, 2.0f, 0, 0.3f, 0, 0, 1.5f, 0}},
        {{1, 1, 1, 1, 1, 1, 1, 1}, {9, 8, 7, 6, 5, 4, 3, 2}},
    };

    for (const auto& [r0, r1] : steps) {
        auto info = make_info(r0, r1);

        mrnf_strategy::launch_fold_densification_and_zero(
            vis.ptr<float>(),
            refine_max.ptr<float>(),
            info.ptr<float>(),
            N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        for (size_t i = 0; i < N; ++i) {
            refine_ref[i] = std::max(refine_ref[i], r1[i]);
            vis_ref[i] += r0[i];
        }

        auto info_h = to_host(info);
        for (float v : info_h) {
            EXPECT_FLOAT_EQ(v, 0.f) << "densification_info must be zeroed after fold";
        }
    }

    auto vis_h = to_host(vis);
    auto ref_h = to_host(refine_max);
    ASSERT_EQ(vis_h.size(), N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(vis_h[i], vis_ref[i]) << "vis i=" << i;
        EXPECT_FLOAT_EQ(ref_h[i], refine_ref[i]) << "refine i=" << i;
    }
}

TEST_F(DensificationInfoZeroTest, McmcMaxMatchesMultiStepReference) {
    constexpr size_t N = 6;
    auto err_max = Tensor::zeros({N}, Device::GPU);
    std::vector<float> err_ref(N, 0.f);

    const std::vector<std::vector<float>> error_rows = {
        {0.1f, 0, 0.5f, 0, 2.0f, 0},
        {0.2f, 1.0f, 0.4f, 0.1f, 0.5f, 3.0f},
        {0, 0, 0, 0, 0, 0},
        {5, 0, 0, 0, 0, 0.01f},
    };

    for (const auto& err : error_rows) {
        std::vector<float> r0(N, 0.f); // unused by MCMC fold but zeroed too
        auto info = make_info(r0, err);

        mcmc::launch_max_error_and_zero_densification(
            err_max.ptr<float>(),
            info.ptr<float>(),
            N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        for (size_t i = 0; i < N; ++i) {
            err_ref[i] = std::max(err_ref[i], err[i]);
        }

        for (float v : to_host(info)) {
            EXPECT_FLOAT_EQ(v, 0.f);
        }
    }

    auto a = to_host(err_max);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(a[i], err_ref[i]) << "error max i=" << i;
    }
}
