/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;
using namespace lfs::training::kernels;

namespace {

    Tensor make_rgb(int H, int W, float base) {
        auto t = Tensor::empty({1, 3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
        auto cpu = Tensor::empty({1, 3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CPU);
        float* p = cpu.ptr<float>();
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    p[c * H * W + y * W + x] =
                        base + 0.01f * static_cast<float>(c) +
                        0.001f * static_cast<float>(x + y);
                }
            }
        }
        return cpu.to(Device::CUDA);
    }

} // namespace

TEST(PhotometricHygieneTest, FusedLossValueStableAndNoCloneAlloc) {
    constexpr int H = 64;
    constexpr int W = 64;
    constexpr float kSsimWeight = 0.2f;

    auto pred = make_rgb(H, W, 0.4f);
    auto gt = make_rgb(H, W, 0.45f);

    FusedL1SSIMWorkspace ws;
    // Warm once (workspace growth + first forward).
    {
        auto [loss0, ctx0] = fused_l1_ssim_forward(pred, gt, kSsimWeight, ws, true);
        (void)fused_l1_ssim_backward(ctx0, ws);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_GT(loss0.cpu().item<float>(), 0.0f);
    }

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto snap = alloc_counter::snapshot();
    auto [loss1, ctx1] = fused_l1_ssim_forward(pred, gt, kSsimWeight, ws, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto delta = alloc_counter::delta_since(snap);
    EXPECT_EQ(delta, 0u)
        << "steady fused_l1_ssim_forward must not clone/alloc the loss scalar";

    const float v1 = loss1.cpu().item<float>();
    ASSERT_GT(v1, 0.0f);

    // Second step same inputs: loss numerically equal (workspace scalar rewritten).
    auto [loss2, ctx2] = fused_l1_ssim_forward(pred, gt, kSsimWeight, ws, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const float v2 = loss2.cpu().item<float>();
    EXPECT_NEAR(v1, v2, 1e-6f) << "loss equivalence across steps";

    // Backward without prior zero_: grad finite and non-zero.
    auto grad = fused_l1_ssim_backward(ctx2, ws);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto gcpu = grad.cpu();
    float gsum = 0.f;
    const float* gp = gcpu.ptr<float>();
    for (size_t i = 0; i < gcpu.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(gp[i]));
        gsum += std::fabs(gp[i]);
    }
    EXPECT_GT(gsum, 0.0f);
}
