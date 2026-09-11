/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Banded PPISP forward (launch_ppisp_forward_chw_region) must reproduce the
// full-image pass bit-exactly: vignetting is the only spatially-dependent stage
// and evaluates in full-image coordinates via (y_offset, full_height).

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#include "lfs/kernels/ppisp.cuh"

namespace {

    constexpr int DEFAULT_SEED = 42;
    constexpr int NUM_CAMERAS = 2;
    constexpr int NUM_FRAMES = 5;
    constexpr int CAMERA_IDX = 1;
    constexpr int FRAME_IDX = 3;

    const auto GPU_F32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

    struct TestParams {
        torch::Tensor exposure;
        torch::Tensor vignetting;
        torch::Tensor color;
        torch::Tensor crf;
    };

    TestParams createParams(const int seed) {
        torch::manual_seed(seed);
        return {
            torch::empty({NUM_FRAMES}, GPU_F32).uniform_(-0.5f, 0.5f),
            torch::empty({NUM_CAMERAS, 3, 5}, GPU_F32).uniform_(-0.1f, 0.1f),
            torch::empty({NUM_FRAMES, 8}, GPU_F32).uniform_(-0.1f, 0.1f),
            torch::empty({NUM_CAMERAS, 3, 4}, GPU_F32).uniform_(-0.5f, 0.5f),
        };
    }

    torch::Tensor createImage(const int height, const int width, const int seed) {
        torch::manual_seed(seed);
        return torch::empty({3, height, width}, GPU_F32).uniform_(0.1f, 0.9f);
    }

    torch::Tensor runFullForward(const TestParams& p, const torch::Tensor& rgb_in) {
        const int height = static_cast<int>(rgb_in.size(1));
        const int width = static_cast<int>(rgb_in.size(2));
        auto rgb_out = torch::empty_like(rgb_in);
        lfs::training::kernels::launch_ppisp_forward_chw(
            p.exposure.data_ptr<float>(), p.vignetting.data_ptr<float>(), p.color.data_ptr<float>(),
            p.crf.data_ptr<float>(), rgb_in.data_ptr<float>(), rgb_out.data_ptr<float>(), height, width,
            NUM_CAMERAS, NUM_FRAMES, CAMERA_IDX, FRAME_IDX, nullptr);
        cudaDeviceSynchronize();
        return rgb_out;
    }

    torch::Tensor runBandedForward(const TestParams& p, const torch::Tensor& rgb_in, const int band_rows) {
        const int height = static_cast<int>(rgb_in.size(1));
        const int width = static_cast<int>(rgb_in.size(2));
        auto rgb_out = torch::empty_like(rgb_in);
        for (int y0 = 0; y0 < height; y0 += band_rows) {
            const int band_height = std::min(band_rows, height - y0);
            const auto band_in = rgb_in.slice(1, y0, y0 + band_height).contiguous();
            auto band_out = torch::empty_like(band_in);
            lfs::training::kernels::launch_ppisp_forward_chw_region(
                p.exposure.data_ptr<float>(), p.vignetting.data_ptr<float>(), p.color.data_ptr<float>(),
                p.crf.data_ptr<float>(), band_in.data_ptr<float>(), band_out.data_ptr<float>(), band_height, width,
                y0, height, NUM_CAMERAS, NUM_FRAMES, CAMERA_IDX, FRAME_IDX, nullptr);
            rgb_out.slice(1, y0, y0 + band_height).copy_(band_out);
        }
        cudaDeviceSynchronize();
        return rgb_out;
    }

    class PPISPBandedRegionTest : public ::testing::Test {};

    TEST_F(PPISPBandedRegionTest, EvenBandsMatchFullImageBitExact) {
        const auto params = createParams(DEFAULT_SEED);
        const auto rgb_in = createImage(64, 48, DEFAULT_SEED);

        const auto full = runFullForward(params, rgb_in);
        const auto banded = runBandedForward(params, rgb_in, 16);

        EXPECT_TRUE(torch::equal(full, banded));
    }

    TEST_F(PPISPBandedRegionTest, UnevenLastBandMatchesFullImageBitExact) {
        const auto params = createParams(DEFAULT_SEED);
        const auto rgb_in = createImage(61, 37, DEFAULT_SEED + 1);

        const auto full = runFullForward(params, rgb_in);
        const auto banded = runBandedForward(params, rgb_in, 16);

        EXPECT_TRUE(torch::equal(full, banded));
    }

    TEST_F(PPISPBandedRegionTest, SingleRowBandsMatchFullImageBitExact) {
        const auto params = createParams(DEFAULT_SEED);
        const auto rgb_in = createImage(8, 24, DEFAULT_SEED + 2);

        const auto full = runFullForward(params, rgb_in);
        const auto banded = runBandedForward(params, rgb_in, 1);

        EXPECT_TRUE(torch::equal(full, banded));
    }

    TEST_F(PPISPBandedRegionTest, WholeImageRegionMatchesLegacyLauncher) {
        const auto params = createParams(DEFAULT_SEED);
        const auto rgb_in = createImage(32, 32, DEFAULT_SEED + 3);

        const auto full = runFullForward(params, rgb_in);
        const auto region = runBandedForward(params, rgb_in, static_cast<int>(rgb_in.size(1)));

        EXPECT_TRUE(torch::equal(full, region));
    }

    TEST_F(PPISPBandedRegionTest, BandOutputDependsOnRegionOffset) {
        // Guards against the region parameters being ignored: the same band data
        // must produce different vignetting when anchored at a different offset.
        const auto params = createParams(DEFAULT_SEED);
        const auto rgb_in = createImage(16, 32, DEFAULT_SEED + 4);
        const int width = static_cast<int>(rgb_in.size(2));

        auto out_top = torch::empty_like(rgb_in);
        auto out_bottom = torch::empty_like(rgb_in);
        lfs::training::kernels::launch_ppisp_forward_chw_region(
            params.exposure.data_ptr<float>(), params.vignetting.data_ptr<float>(), params.color.data_ptr<float>(),
            params.crf.data_ptr<float>(), rgb_in.data_ptr<float>(), out_top.data_ptr<float>(), 16, width, 0, 256,
            NUM_CAMERAS, NUM_FRAMES, CAMERA_IDX, FRAME_IDX, nullptr);
        lfs::training::kernels::launch_ppisp_forward_chw_region(
            params.exposure.data_ptr<float>(), params.vignetting.data_ptr<float>(), params.color.data_ptr<float>(),
            params.crf.data_ptr<float>(), rgb_in.data_ptr<float>(), out_bottom.data_ptr<float>(), 16, width, 240,
            256, NUM_CAMERAS, NUM_FRAMES, CAMERA_IDX, FRAME_IDX, nullptr);
        cudaDeviceSynchronize();

        EXPECT_FALSE(torch::equal(out_top, out_bottom));
    }

} // namespace
