/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/ssim.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using lfs::core::Device;
using lfs::core::Tensor;
using lfs::training::kernels::launch_ssim_to_error_map;
using lfs::training::kernels::ssim_forward_map;

namespace {

    Tensor make_synthetic_pair(Tensor& pred, Tensor& gt) {
        constexpr int C = 3;
        constexpr int H = 64;
        constexpr int W = 64;
        pred = Tensor::zeros({1, C, H, W}, Device::GPU);
        gt = Tensor::zeros({1, C, H, W}, Device::GPU);

        auto pred_cpu = Tensor::zeros({1, C, H, W}, Device::CPU);
        auto gt_cpu = Tensor::zeros({1, C, H, W}, Device::CPU);
        float* p = pred_cpu.ptr<float>();
        float* g = gt_cpu.ptr<float>();
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    const int idx = c * H * W + y * W + x;
                    if (x < W / 2) {
                        // Uniform luminance mismatch: black render vs saturated GT.
                        p[idx] = 0.0f;
                        g[idx] = 1.0f;
                    } else {
                        // Textured mismatch: checker vs inverted checker.
                        const float v = (((x / 4) + (y / 4)) & 1) ? 0.85f : 0.15f;
                        p[idx] = v;
                        g[idx] = 1.0f - v;
                    }
                }
            }
        }
        pred = pred_cpu.cuda();
        gt = gt_cpu.cuda();
        return pred;
    }

} // namespace

class DensifyErrorMapTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

TEST_F(DensifyErrorMapTest, ContrastStructureIgnoresUniformLuminanceMismatch) {
    Tensor pred, gt;
    make_synthetic_pair(pred, gt);

    auto maps = ssim_forward_map(pred, gt, /*apply_valid_padding=*/false);
    ASSERT_TRUE(maps.ssim_map.is_valid());
    ASSERT_TRUE(maps.cs_map.is_valid());
    ASSERT_EQ(maps.ssim_map.ndim(), 4);
    ASSERT_EQ(maps.cs_map.ndim(), 4);
    ASSERT_EQ(maps.ssim_map.shape()[1], static_cast<size_t>(3));
    ASSERT_EQ(maps.cs_map.shape()[1], static_cast<size_t>(3));

    auto ssim_err = Tensor::empty({64, 64}, Device::GPU);
    auto cs_err = Tensor::empty({64, 64}, Device::GPU);
    launch_ssim_to_error_map(maps.ssim_map, ssim_err);
    launch_ssim_to_error_map(maps.cs_map, cs_err);

    auto ssim_cpu = ssim_err.cpu();
    auto cs_cpu = cs_err.cpu();
    const float* ssim_p = ssim_cpu.ptr<float>();
    const float* cs_p = cs_cpu.ptr<float>();

    // Interior of the uniform half vs interior of the textured half, away from the
    // 11-pixel SSIM halo and the vertical seam.
    double ssim_uniform = 0.0, cs_uniform = 0.0, ssim_tex = 0.0, cs_tex = 0.0;
    int n_uniform = 0, n_tex = 0;
    for (int y = 16; y < 48; ++y) {
        for (int x = 8; x < 24; ++x) {
            ssim_uniform += ssim_p[y * 64 + x];
            cs_uniform += cs_p[y * 64 + x];
            ++n_uniform;
        }
        for (int x = 40; x < 56; ++x) {
            ssim_tex += ssim_p[y * 64 + x];
            cs_tex += cs_p[y * 64 + x];
            ++n_tex;
        }
    }
    ASSERT_GT(n_uniform, 0);
    ASSERT_GT(n_tex, 0);
    ssim_uniform /= n_uniform;
    cs_uniform /= n_uniform;
    ssim_tex /= n_tex;
    cs_tex /= n_tex;

    EXPECT_GT(ssim_uniform, 0.4f) << "full SSIM error should be high on uniform sky";
    EXPECT_LT(cs_uniform, 0.05f) << "CS error should be ~0 on uniform luminance mismatch";
    EXPECT_GT(cs_tex, 0.3f) << "CS error should be high on textured mismatch";
    EXPECT_GT(ssim_tex, 0.3f) << "full SSIM error should be high on textured mismatch";
    EXPECT_GT(ssim_uniform, cs_uniform * 4.0f);
}
