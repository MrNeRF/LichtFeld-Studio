/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/kernels/ssim.cuh"
#include "training/kernels/ppisp.cuh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::BilateralGrid;
    using lfs::training::BilateralGridParameterization;
    using lfs::training::PPISP;
    using namespace lfs::training::kernels;

    Tensor load_chw(const std::filesystem::path& path) {
        auto [pixels, width, height, channels] = lfs::core::load_image(path);
        if (!pixels)
            return {};
        const size_t hw = static_cast<size_t>(width) * height;
        std::vector<float> chw(3 * hw);
        for (size_t i = 0; i < hw; ++i) {
            for (size_t c = 0; c < 3; ++c)
                chw[c * hw + i] = static_cast<float>(pixels[i * channels + c]) / 255.0f;
        }
        lfs::core::free_image(pixels);
        return Tensor::from_vector(chw, {size_t{3}, static_cast<size_t>(height), static_cast<size_t>(width)},
                                   Device::CUDA);
    }

    Tensor wave(const lfs::core::TensorShape& shape, const float seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = 0.2f * std::sin(seed * 0.31f * static_cast<float>(i + 3));
        return Tensor::from_vector(values, shape, Device::CUDA);
    }

    std::vector<float> host(const Tensor& tensor) { return tensor.cpu().contiguous().to_vector(); }

    void perturb(BilateralGrid& grid) {
        auto params = host(grid.grids());
        for (size_t i = 0; i < params.size(); ++i)
            params[i] += 0.08f * std::sin(0.17f * static_cast<float>(i + 1));
        grid.grids().copy_from(Tensor::from_vector(params, grid.grids().shape(), Device::CUDA));
    }

    float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size())
            return std::numeric_limits<float>::infinity();
        float diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            const float d = std::abs(a[i] - b[i]);
            if (!std::isfinite(d))
                return std::numeric_limits<float>::infinity();
            diff = std::max(diff, d);
        }
        return diff;
    }

    // Two neighbouring frames of the first scene under the test data directory with quarter-resolution images.
    std::vector<std::filesystem::path> real_frames() {
        std::error_code error;
        std::vector<std::filesystem::path> scenes;
        for (const auto& entry : std::filesystem::directory_iterator(TEST_DATA_DIR, error)) {
            if (std::filesystem::is_directory(entry.path() / "images_4", error))
                scenes.push_back(entry.path() / "images_4");
        }
        std::sort(scenes.begin(), scenes.end());
        for (const auto& scene : scenes) {
            std::vector<std::filesystem::path> frames;
            for (const auto& entry : std::filesystem::directory_iterator(scene, error)) {
                const auto extension = entry.path().extension();
                if (extension == ".JPG" || extension == ".jpg")
                    frames.push_back(entry.path());
            }
            if (frames.size() >= 2) {
                std::sort(frames.begin(), frames.end());
                return {frames[0], frames[1]};
            }
        }
        return {};
    }
} // namespace

using AppearanceInPlace = lfs::test::CudaBackendTest;

// Fails if writing the output or the image gradient over its input changes the result, or if the grid gradient is
// taken after the image gradient overwrote the incoming one.
TEST_F(AppearanceInPlace, BilateralGridMatchesAllocatingPathsOnRealImage) {
    const auto frames = real_frames();
    if (frames.empty())
        GTEST_SKIP() << "no quarter-resolution images in " << TEST_DATA_DIR;
    const auto chw = load_chw(frames[0]);
    ASSERT_TRUE(chw.is_valid()) << frames[0];
    const auto hwc = chw.permute({1, 2, 0}).contiguous();
    for (const auto parameterization :
         {BilateralGridParameterization::Affine, BilateralGridParameterization::ExposureChroma}) {
        for (const auto& image : {chw, hwc}) {
            BilateralGrid grid(1, 16, 16, 8, 20, {}, parameterization);
            perturb(grid);

            const auto output = host(grid.apply(image, 0));
            auto in_place = image.clone();
            grid.apply_in_place(in_place, 0);
            EXPECT_EQ(host(in_place), output);

            const auto grad_output = wave(image.shape(), 13.0f);
            const auto grad = host(grid.backward(image, grad_output, 0));
            const auto grid_grad = host(grid.grad_slice());
            auto grad_in_place = grad_output.clone();
            grid.backward_in_place(image, grad_in_place, 0);
            EXPECT_EQ(host(grad_in_place), grad);
            EXPECT_LT(max_abs_diff(host(grid.grad_slice()), grid_grad), 5e-5f);
        }
    }
}

// Fails if the PPISP backward reads the incoming gradient after overwriting it, for the image or the parameter
// gradients.
TEST_F(AppearanceInPlace, PpispBackwardOverIncomingGradientMatchesOnRealImage) {
    const auto frames = real_frames();
    if (frames.empty())
        GTEST_SKIP() << "no quarter-resolution images in " << TEST_DATA_DIR;
    const auto image = load_chw(frames[0]);
    ASSERT_TRUE(image.is_valid()) << frames[0];
    PPISP ppisp(100);
    ppisp.register_frame(0, 0);
    ppisp.finalize();
    const auto exposure = wave(ppisp.exposure_params().shape(), 3.0f);
    const auto vignetting = wave(ppisp.vignetting_params().shape(), 5.0f);
    const auto color = wave(ppisp.color_params().shape(), 7.0f);
    const auto crf = wave(ppisp.crf_params().shape(), 9.0f);
    const auto grad_output = wave(image.shape(), 13.0f);

    const auto backward = [&](const Tensor& grad_rgb_out, Tensor& grad_rgb_in) {
        std::vector<Tensor> grads;
        for (const auto* param : {&exposure, &vignetting, &color, &crf})
            grads.push_back(Tensor::zeros(param->shape(), Device::CUDA));
        launch_ppisp_backward_chw(exposure.ptr<float>(), vignetting.ptr<float>(), color.ptr<float>(),
                                  crf.ptr<float>(), image.ptr<float>(), grad_rgb_out.ptr<float>(),
                                  grads[0].ptr<float>(), grads[1].ptr<float>(), grads[2].ptr<float>(),
                                  grads[3].ptr<float>(), grad_rgb_in.ptr<float>(), static_cast<int>(image.shape()[1]),
                                  static_cast<int>(image.shape()[2]), 1, 1, 0, 0);
        return grads;
    };
    auto grad_rgb = Tensor::empty(image.shape(), Device::CUDA);
    const auto param_grads = backward(grad_output, grad_rgb);
    auto grad_in_place = grad_output.clone();
    const auto param_grads_in_place = backward(grad_in_place, grad_in_place);

    EXPECT_EQ(host(grad_in_place), host(grad_rgb));
    for (size_t i = 0; i < param_grads.size(); ++i) {
        const auto expected = host(param_grads[i]);
        float scale = 1.0f;
        for (const float value : expected)
            scale = std::max(scale, std::abs(value));
        EXPECT_LT(max_abs_diff(host(param_grads_in_place[i]), expected), 1e-5f * scale) << "parameter " << i;
    }
}

// Fails if the trainer's exposure correction chain departs from the allocating one: grid applied over the PPISP
// output, the loss gradient overwritten in the loss arena, the PPISP output recomputed for the grid backward, and
// the raw-render gradient added last.
TEST_F(AppearanceInPlace, ExposureCorrectionChainMatchesAllocatingChainOnRealImages) {
    const auto frames = real_frames();
    if (frames.empty())
        GTEST_SKIP() << "no quarter-resolution images in " << TEST_DATA_DIR;
    const auto raw = load_chw(frames[0]);
    const auto gt = load_chw(frames[1]);
    ASSERT_TRUE(raw.is_valid() && gt.is_valid());
    ASSERT_EQ(raw.shape(), gt.shape());
    constexpr float ssim_weight = 0.2f;

    PPISP ppisp(100);
    ppisp.register_frame(0, 0);
    ppisp.finalize();
    ppisp.exposure_params().add_(wave(ppisp.exposure_params().shape(), 3.0f));
    ppisp.color_params().add_(wave(ppisp.color_params().shape(), 5.0f));
    BilateralGrid grid(1, 16, 16, 8, 20, {}, BilateralGridParameterization::ExposureChroma);
    perturb(grid);

    const auto ppisp_output = ppisp.apply(raw, 0, 0);
    auto corrected = grid.apply(ppisp_output, 0);
    corrected.clamp_(0.0f, 1.0f);
    DecoupledFusedL1SSIMWorkspace workspace;
    const auto [loss, ctx] = decoupled_fused_l1_ssim_forward(corrected, raw, gt, ssim_weight, workspace, true);
    const auto grad_corrected = decoupled_fused_l1_ssim_backward(ctx, workspace).grad_corrected.squeeze(0);
    auto raw_grad = Tensor::zeros(raw.shape(), Device::CUDA);
    accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), raw_grad);
    const auto grid_grad_rgb = grid.backward(ppisp_output, grad_corrected, 0);
    const auto grid_grad = host(grid.grad_slice());
    const auto expected = host(ppisp.backward(raw, grid_grad_rgb, 0, 0).add_(raw_grad));

    auto in_place = ppisp.apply(raw, 0, 0);
    grid.apply_in_place(in_place, 0);
    in_place.clamp_(0.0f, 1.0f);
    EXPECT_EQ(host(in_place), host(corrected));
    LossWorkspaceArena arena;
    auto& arena_workspace = arena.decoupled();
    const auto [arena_loss, arena_ctx] =
        decoupled_fused_l1_ssim_forward(in_place, raw, gt, ssim_weight, arena_workspace, true);
    auto render_grad = decoupled_fused_l1_ssim_backward(arena_ctx, arena_workspace).grad_corrected.squeeze(0);
    grid.backward_in_place(ppisp.apply(raw, 0, 0), render_grad, 0);
    EXPECT_LT(max_abs_diff(host(grid.grad_slice()), grid_grad), 5e-5f);
    ppisp.backward_in_place(raw, render_grad, 0, 0);
    accumulate_decoupled_raw_gradient(decoupled_raw_gradient(arena_ctx), render_grad);
    EXPECT_EQ(host(render_grad), expected);
}
