/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "lfs/training/screen_share.cuh"
#include "training/kernels/densification_kernels.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;
using lfs::training::gaussian_screen_share;
using lfs::training::screen_share_cap_active;
using lfs::training::kernels::launch_clip_log_scale_by_screen_share;

class ScreenShareTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

TEST_F(ScreenShareTest, NearCameraBlobApproachesOne) {
    const float share = gaussian_screen_share(
        0.0f, 0.0f, 0.01f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        0.0f);
    EXPECT_GT(share, 0.9f);
    EXPECT_LE(share, 1.0f);
}

TEST_F(ScreenShareTest, FarSmallSplatApproachesZero) {
    const float share = gaussian_screen_share(
        0.0f, 0.0f, 50.0f,
        0.0f, 0.0f, 0.0f,
        -4.0f, -4.0f, -4.0f,
        -2.0f);
    EXPECT_LT(share, 0.05f);
    EXPECT_GE(share, 0.0f);
}

TEST_F(ScreenShareTest, CapActiveRange) {
    EXPECT_FALSE(screen_share_cap_active(0.0f));
    EXPECT_FALSE(screen_share_cap_active(1.0f));
    EXPECT_FALSE(screen_share_cap_active(-1.0f));
    EXPECT_TRUE(screen_share_cap_active(0.3f));
}

TEST_F(ScreenShareTest, HardClipShrinksOversizedLogScale) {
    constexpr int n = 4;
    auto scales = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
    auto share = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA);

    auto scales_cpu = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CPU);
    auto share_cpu = Tensor::zeros({static_cast<size_t>(n)}, Device::CPU);
    float* s = scales_cpu.ptr<float>();
    float* sh = share_cpu.ptr<float>();
    for (int i = 0; i < n; ++i) {
        s[i * 3 + 0] = 0.5f;
        s[i * 3 + 1] = 0.4f;
        s[i * 3 + 2] = 0.3f;
        sh[i] = 0.0f;
    }
    sh[1] = 0.6f;  // over 0.3, ratio=2, log(2)>log(1.5) so clip log(1.5)
    sh[2] = 0.35f; // mild overshoot, log(0.35/0.3) < log(1.5)
    scales = scales_cpu.cuda();
    share = share_cpu.cuda();

    ASSERT_EQ(scales.shape()[0], static_cast<size_t>(n));
    ASSERT_EQ(scales.shape()[1], static_cast<size_t>(3));
    ASSERT_EQ(share.numel(), static_cast<size_t>(n));

    launch_clip_log_scale_by_screen_share(
        scales.ptr<float>(), share.ptr<float>(), nullptr, 0, 0.3f, n);

    auto out = scales.cpu();
    const float* o = out.ptr<float>();
    EXPECT_FLOAT_EQ(o[0], 0.5f);
    EXPECT_FLOAT_EQ(o[1], 0.4f);
    EXPECT_FLOAT_EQ(o[2], 0.3f);

    const float hard = std::log(1.5f);
    EXPECT_NEAR(o[3], 0.5f - hard, 1e-5f);
    EXPECT_NEAR(o[4], 0.4f - hard, 1e-5f);
    EXPECT_NEAR(o[5], 0.3f - hard, 1e-5f);

    const float mild = std::log(0.35f / 0.3f);
    EXPECT_NEAR(o[6], 0.5f - mild, 1e-5f);
    EXPECT_NEAR(o[7], 0.4f - mild, 1e-5f);
    EXPECT_NEAR(o[8], 0.3f - mild, 1e-5f);

    EXPECT_FLOAT_EQ(o[9], 0.5f);
    EXPECT_FLOAT_EQ(o[10], 0.4f);
    EXPECT_FLOAT_EQ(o[11], 0.3f);
}
