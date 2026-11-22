/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Blur-tolerant SSIM: the sigma rule, the plain-SSIM path below it, and the gradient of the blurred branch.

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include "training/losses/photometric_loss.hpp"
#include "training/trainer.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::kernels;

namespace {
    Tensor random_image(const size_t height, const size_t width, const unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(0.05f, 0.95f);
        std::vector<float> values(3 * height * width);
        for (auto& v : values)
            v = uniform(rng);
        return Tensor::from_vector(values, {size_t{3}, height, width}, Device::CUDA);
    }

    // The photo the blurred branch should prefer: a 3x3 box blur of the render.
    Tensor box_blurred(const Tensor& image) {
        const auto cpu = image.cpu();
        const size_t C = cpu.shape()[0], H = cpu.shape()[1], W = cpu.shape()[2];
        std::vector<float> out(C * H * W);
        const float* in = cpu.ptr<float>();
        for (size_t c = 0; c < C; ++c)
            for (size_t y = 0; y < H; ++y)
                for (size_t x = 0; x < W; ++x) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const auto yy = static_cast<long>(y) + dy, xx = static_cast<long>(x) + dx;
                            if (yy < 0 || xx < 0 || yy >= static_cast<long>(H) || xx >= static_cast<long>(W))
                                continue;
                            sum += in[(c * H + yy) * W + xx];
                            ++count;
                        }
                    out[(c * H + y) * W + x] = sum / static_cast<float>(count);
                }
        return Tensor::from_vector(out, {C, H, W}, Device::CUDA);
    }

    void expect_identical(const Tensor& a, const Tensor& b) {
        ASSERT_EQ(a.numel(), b.numel());
        const auto x = a.contiguous().cpu(), y = b.contiguous().cpu();
        for (size_t i = 0; i < x.numel(); ++i)
            ASSERT_EQ(x.ptr<float>()[i], y.ptr<float>()[i]) << "element " << i;
    }

    float sum_of_products(const Tensor& a, const Tensor& b) { return a.mul(b).sum().item<float>(); }
} // namespace

// Catches a change of the blur sigma rule: plain SSIM below a short side of 1,200 pixels, sigma growing with the
// short side above it and clamped at 1.5.
TEST(BlurTolerantSsim, SigmaRule) {
    EXPECT_EQ(blur_tolerant_ssim_sigma(822, 1237), 0.0f);
    EXPECT_EQ(blur_tolerant_ssim_sigma(1199, 4000), 0.0f);
    EXPECT_EQ(blur_tolerant_ssim_sigma(4000, 1199), 0.0f);
    EXPECT_FLOAT_EQ(blur_tolerant_ssim_sigma(1200, 1800), 0.6f);
    EXPECT_FLOAT_EQ(blur_tolerant_ssim_sigma(1643, 2473), 0.8215f);
    EXPECT_FLOAT_EQ(blur_tolerant_ssim_sigma(2473, 1643), 0.8215f);
    EXPECT_EQ(blur_tolerant_ssim_sigma(3000, 4500), 1.5f);
    EXPECT_EQ(blur_tolerant_ssim_sigma(3648, 5472), 1.5f);
    EXPECT_EQ(blur_tolerant_ssim_sigma(10000, 20000), 1.5f);
}

namespace lfs::training {
    struct TrainerBlurTolerantSsimTestAccess {
        static float select(Trainer& trainer, const param::OptimizationParameters& params, const Tensor& target) {
            trainer.select_blur_tolerant_ssim(params, target, false);
            return trainer.blur_sigma_;
        }
        static auto loss(Trainer& trainer, const Tensor& image, const Tensor& target,
                         const param::OptimizationParameters& params, const Tensor& raw) {
            return trainer.compute_photometric_loss_with_gradient(image, target, params, raw);
        }
    };
} // namespace lfs::training

class BlurTolerantSsimTrainerTest : public ::testing::Test {
protected:
    Scene scene;
    std::unique_ptr<lfs::training::Trainer> trainer;
    param::OptimizationParameters params = param::OptimizationParameters::mrnf_defaults();

    void SetUp() override {
        const auto group = scene.addCameraGroup("Training", scene.addGroup("Cameras"), 1);
        scene.addCamera("camera.png", group,
                        std::make_shared<Camera>(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU), 32.0f,
                                                 32.0f, 16.0f, 16.0f, Tensor{}, Tensor{}, CameraModelType::PINHOLE,
                                                 "camera.png", std::filesystem::path{}, std::filesystem::path{}, 32,
                                                 32, 0));
        trainer = std::make_unique<lfs::training::Trainer>(scene);
        params.gradient_loss_weight = 0.0f;
    }
};

// Catches the MRNF loss leaving plain SSIM for photos below the sigma rule's 1,200 pixels, and the blur-tolerant
// loss not being used from there on.
TEST_F(BlurTolerantSsimTrainerTest, SigmaZeroIsPlainSsim) {
    using Access = lfs::training::TrainerBlurTolerantSsimTestAccess;
    for (const auto [height, width] : {std::pair<size_t, size_t>{64, 96}, std::pair<size_t, size_t>{1199, 1216}}) {
        SCOPED_TRACE(height);
        const auto image = random_image(height, width, 7), target = box_blurred(image);
        const auto raw = image.mul(0.9f).add(0.05f);
        ASSERT_EQ(Access::select(*trainer, params, target), 0.0f);

        // Plain loss: the fused L1 + SSIM kernels.
        lfs::training::losses::PhotometricLoss plain;
        const auto reference = plain.forward(image, target, {.lambda_dssim = params.lambda_dssim});
        ASSERT_TRUE(reference);
        const auto result = Access::loss(*trainer, image, target, params, {});
        ASSERT_TRUE(result);
        EXPECT_EQ(result->loss.item<float>(), reference->first.item<float>());
        expect_identical(result->grad_corrected, reference->second.grad_image);
        EXPECT_FALSE(result->raw_gradient);

        // Decoupled loss: the decoupled kernels, the raw-render part still read from their partials.
        DecoupledFusedL1SSIMWorkspace workspace;
        auto [loss, ctx] = decoupled_fused_l1_ssim_forward(image, raw, target, params.lambda_dssim, workspace);
        const auto expected_loss = loss.item<float>();
        auto expected = decoupled_fused_l1_ssim_backward(ctx, workspace).grad_corrected.squeeze(0).clone();
        accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), expected);
        const auto decoupled = Access::loss(*trainer, image, target, params, raw);
        ASSERT_TRUE(decoupled);
        ASSERT_TRUE(decoupled->raw_gradient);
        EXPECT_FALSE(decoupled->raw_gradient->precomputed.is_valid());
        EXPECT_EQ(decoupled->loss.item<float>(), expected_loss);
        auto combined = decoupled->grad_corrected.clone();
        accumulate_decoupled_raw_gradient(*decoupled->raw_gradient, combined);
        expect_identical(combined, expected);
    }

    // From a short side of 1,200 pixels on the loss is the blur-tolerant one.
    const auto image = random_image(1200, 1216, 7), target = box_blurred(image);
    const float sigma = Access::select(*trainer, params, target);
    ASSERT_FLOAT_EQ(sigma, 0.6f);
    LossWorkspaceArena arena;
    const auto expected = blur_tolerant_loss(image, {}, target, {}, params.lambda_dssim, sigma, arena);
    const auto result = Access::loss(*trainer, image, target, params, {});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->loss.item<float>(), expected.loss.item<float>());
    expect_identical(result->grad_corrected, expected.grad_corrected);
    lfs::training::losses::PhotometricLoss plain;
    const auto plain_loss = plain.forward(image, target, {.lambda_dssim = params.lambda_dssim});
    ASSERT_TRUE(plain_loss);
    EXPECT_LT(result->loss.item<float>(), plain_loss->first.item<float>()); // the photo is the blurred render
}

// Catches a wrong gradient of the blur-tolerant SSIM: directional derivatives of the loss match <gradient, d> for
// the plain, decoupled and weighted forms, with the blurred branch in force (the photo is a blur of the render). The
// SSIM weight is 1, so no L1 kink sits inside the finite difference.
TEST(BlurTolerantSsim, GradientMatchesDirectionalDerivative) {
    constexpr size_t H = 40, W = 48;
    constexpr float sigma = 0.8f, eps = 1e-3f;
    const auto image = random_image(H, W, 3), target = box_blurred(image);
    const auto raw_base = random_image(H, W, 5).mul(0.2f).add(image.mul(0.8f)).contiguous();
    std::vector<float> signs(3 * H * W), weights(H * W);
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> uniform(0.5f, 1.5f);
    for (auto& s : signs)
        s = rng() % 2 == 0 ? 1.0f : -1.0f;
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = (i / W) % 9 == 4 ? 0.0f : uniform(rng);
    const auto direction = Tensor::from_vector(signs, {size_t{3}, H, W}, Device::CUDA);
    const auto weight = Tensor::from_vector(weights, {H, W}, Device::CUDA);

    for (const bool decoupled : {false, true}) {
        for (const bool masked : {false, true}) {
            SCOPED_TRACE(std::string(decoupled ? "decoupled" : "plain") + (masked ? ", weighted" : ""));
            const Tensor w = masked ? weight : Tensor{};
            LossWorkspaceArena arena;
            const auto value = [&](const Tensor& corrected, const Tensor& raw) {
                return blur_tolerant_loss(corrected.contiguous(), decoupled ? raw.contiguous() : Tensor{}, target, w,
                                          1.0f, sigma, arena)
                    .loss.item<float>();
            };
            const auto result = blur_tolerant_loss(image, decoupled ? raw_base : Tensor{}, target, w, 1.0f, sigma, arena);
            const auto grad_corrected = result.grad_corrected.squeeze(0).clone();
            ASSERT_EQ(static_cast<bool>(result.raw), decoupled);
            auto grad_raw = Tensor::zeros_like(grad_corrected);
            if (result.raw)
                accumulate_decoupled_raw_gradient(*result.raw, grad_raw);

            const float numeric_c =
                (value(image.add(direction.mul(eps)), raw_base) - value(image.sub(direction.mul(eps)), raw_base)) /
                (2.0f * eps);
            const float analytic_c = sum_of_products(grad_corrected, direction);
            EXPECT_NEAR(numeric_c, analytic_c, 0.03f * std::abs(analytic_c) + 1e-5f);
            if (decoupled) {
                const float numeric_r =
                    (value(image, raw_base.add(direction.mul(eps))) - value(image, raw_base.sub(direction.mul(eps)))) /
                    (2.0f * eps);
                const float analytic_r = sum_of_products(grad_raw, direction);
                EXPECT_NEAR(numeric_r, analytic_r, 0.03f * std::abs(analytic_r) + 1e-5f);
            }
        }
    }
}
