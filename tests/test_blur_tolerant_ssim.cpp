/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Blur-tolerant SSIM term and multi-scale photometric loss: the switches off are today's loss,
// the gradients match finite differences (blend, blur adjoint, borders, weight maps, pooling),
// and the blended term rewards a render sharper than a soft photo.

#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include "training/losses/photometric_loss.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::kernels;

namespace {

    bool cuda_available() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    std::vector<float> random_texture(const int C, const int H, const int W, const unsigned seed,
                                      const float low = 0.15f, const float high = 0.85f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(low, high);
        std::vector<float> data(static_cast<size_t>(C) * H * W);
        for (auto& v : data)
            v = dist(rng);
        return data;
    }

    Tensor image_tensor(const std::vector<float>& data, const int C, const int H, const int W) {
        return Tensor::from_vector(data, {static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W)},
                                   Device::CUDA);
    }

    std::vector<float> blurred(const std::vector<float>& data, const int C, const int H, const int W,
                               const float sigma = BLUR_TOLERANT_SSIM_SIGMA) {
        return normalized_gaussian_blur(image_tensor(data, C, H, W), Tensor{}, sigma).cpu().to_vector();
    }

    struct LossCase {
        int C = 3, H = 40, W = 48;
        std::vector<float> target;
        std::vector<float> weight; // empty: unweighted
        std::vector<float> raw;    // non-empty: decoupled loss, the prediction is the corrected image
        float ssim_weight = 0.8f;
        int levels = 1;
        bool blur_tolerant = true;
        float denominator = 0.0f;
    };

    struct Evaluation {
        float loss = 0.0f;
        std::vector<float> grad;
        std::vector<float> grad_raw;
        float mask_sum = 0.0f;
    };

    Evaluation evaluate(const LossCase& c, const std::vector<float>& prediction) {
        LossWorkspaceArena arena;
        PhotometricPyramidWorkspace workspace;
        const Tensor weight = c.weight.empty()
                                  ? Tensor{}
                                  : Tensor::from_vector(c.weight, {static_cast<size_t>(c.H), static_cast<size_t>(c.W)},
                                                        Device::CUDA);
        const auto x = image_tensor(prediction, c.C, c.H, c.W);
        const auto y = image_tensor(c.target, c.C, c.H, c.W);
        if (!c.raw.empty()) {
            const auto result =
                decoupled_photometric_pyramid_loss(x, image_tensor(c.raw, c.C, c.H, c.W), y, weight, c.ssim_weight,
                                                   c.levels, c.blur_tolerant, arena, workspace, c.denominator);
            return {result.loss.item<float>(), result.grad.cpu().to_vector(), result.grad_raw.cpu().to_vector(),
                    result.mask_sum_value};
        }
        const auto result = photometric_pyramid_loss(x, y, weight, c.ssim_weight, c.levels, c.blur_tolerant, arena,
                                                     workspace, c.denominator);
        return {result.loss.item<float>(), result.grad.cpu().to_vector(), {}, result.mask_sum_value};
    }

    // Central differences: one pixel at a time and along one random direction over the whole image.
    // With perturb_raw (decoupled loss) the raw render is differentiated and x is the fixed corrected image.
    void expect_gradient_matches_finite_differences(LossCase c, std::vector<float> x,
                                                    const std::vector<std::array<int, 3>>& pixels,
                                                    const char* label, const bool perturb_raw = false) {
        constexpr float pixel_eps = 1.0e-2f;
        // Keep the probed pixels off the L1 kink.
        for (const auto& [ch, py, px] : pixels) {
            const size_t i = (static_cast<size_t>(ch) * c.H + py) * c.W + px;
            if (std::abs(x[i] - c.target[i]) < 2.5f * pixel_eps)
                x[i] = c.target[i] + (c.target[i] < 0.5f ? 2.5f : -2.5f) * pixel_eps;
        }
        const auto corrected = x;
        const auto evaluate_at = [&](const std::vector<float>& v) {
            if (!perturb_raw)
                return evaluate(c, v);
            LossCase moved = c;
            moved.raw = v;
            return evaluate(moved, corrected);
        };
        if (perturb_raw)
            x = c.raw;
        const auto evaluated = evaluate_at(x);
        Evaluation analytic{.loss = evaluated.loss, .grad = perturb_raw ? evaluated.grad_raw : evaluated.grad};
        ASSERT_TRUE(std::isfinite(analytic.loss)) << label;
        ASSERT_EQ(analytic.grad.size(), x.size()) << label;

        for (const auto& [ch, py, px] : pixels) {
            const size_t i = (static_cast<size_t>(ch) * c.H + py) * c.W + px;
            auto plus = x;
            auto minus = x;
            plus[i] += pixel_eps;
            minus[i] -= pixel_eps;
            const double fd =
                (static_cast<double>(evaluate_at(plus).loss) - evaluate_at(minus).loss) / (2.0 * pixel_eps);
            const double g = analytic.grad[i];
            EXPECT_NEAR(g, fd, 0.06 * std::abs(fd) + 1.0e-5)
                << label << " pixel (" << ch << ", " << py << ", " << px << ")";
        }

        // Along a direction over every pixel, the SSIM term alone: a step moving all pixels crosses L1
        // kinks. The quotient is only good to a few percent: below ~1e-3 the float rounding of SSIM in
        // low-variance windows (the blurred branch against a soft photo) dominates it, above it the
        // curvature of the soft maximum (temperature 50) does.
        c.ssim_weight = 1.0f;
        const auto ssim_only = evaluate_at(x);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> sign(0, 1);
        std::vector<float> direction(x.size());
        for (auto& v : direction)
            v = sign(rng) ? 1.0f : -1.0f;
        constexpr float dir_eps = 2.0e-3f;
        auto plus = x;
        auto minus = x;
        double projected = 0.0;
        const auto& ssim_grad = perturb_raw ? ssim_only.grad_raw : ssim_only.grad;
        for (size_t i = 0; i < x.size(); ++i) {
            plus[i] += dir_eps * direction[i];
            minus[i] -= dir_eps * direction[i];
            projected += static_cast<double>(ssim_grad[i]) * direction[i];
        }
        const double fd = (static_cast<double>(evaluate_at(plus).loss) - evaluate_at(minus).loss) / (2.0 * dir_eps);
        EXPECT_NEAR(projected, fd, 0.08 * std::abs(fd) + 3.0e-5) << label << " directional derivative";
    }

    // Left half: the photo is the blurred render (the blurred branch wins). Right half: the photo is
    // the render plus noise (the sharp branch wins).
    LossCase two_region_case(std::vector<float>& x) {
        LossCase c;
        x = random_texture(c.C, c.H, c.W, 11);
        const auto soft = blurred(x, c.C, c.H, c.W);
        std::mt19937 rng(5);
        std::normal_distribution<float> noise(0.0f, 0.03f);
        c.target.resize(x.size());
        for (int ch = 0; ch < c.C; ++ch)
            for (int y = 0; y < c.H; ++y)
                for (int xx = 0; xx < c.W; ++xx) {
                    const size_t i = (static_cast<size_t>(ch) * c.H + y) * c.W + xx;
                    c.target[i] = xx < c.W / 2 ? soft[i] : std::clamp(x[i] + noise(rng), 0.0f, 1.0f);
                }
        return c;
    }

    std::vector<float> pool_cpu(const std::vector<float>& in, const int planes, const int H, const int W) {
        const int h = (H + 1) / 2, w = (W + 1) / 2;
        std::vector<float> out(static_cast<size_t>(planes) * h * w);
        for (int p = 0; p < planes; ++p)
            for (int Y = 0; Y < h; ++Y)
                for (int X = 0; X < w; ++X) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b) {
                            const int y = 2 * Y + a, x = 2 * X + b;
                            if (y < H && x < W) {
                                sum += in[(static_cast<size_t>(p) * H + y) * W + x];
                                ++count;
                            }
                        }
                    out[(static_cast<size_t>(p) * h + Y) * w + X] = sum / static_cast<float>(count);
                }
        return out;
    }

    void expect_bitwise_equal(const Tensor& a, const Tensor& b, const char* label) {
        const auto va = a.cpu().to_vector();
        const auto vb = b.cpu().to_vector();
        ASSERT_EQ(va.size(), vb.size()) << label;
        for (size_t i = 0; i < va.size(); ++i)
            ASSERT_EQ(va[i], vb[i]) << label << " at " << i;
    }

} // namespace

class BlurTolerantSSIMTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available())
            GTEST_SKIP() << "No CUDA device available";
    }
};

TEST(LossPyramidLevelsTest, FollowsTheShortSide) {
    EXPECT_EQ(loss_pyramid_levels(822, 1237), 1);
    EXPECT_EQ(loss_pyramid_levels(1919, 6000), 1);
    EXPECT_EQ(loss_pyramid_levels(1920, 1920), 2);
    EXPECT_EQ(loss_pyramid_levels(3286, 4946), 2);
    EXPECT_EQ(loss_pyramid_levels(3933, 5246), 3);
    EXPECT_EQ(loss_pyramid_levels(5246, 3933), 3);
    EXPECT_EQ(loss_pyramid_levels(3840, 3840), 3);
    EXPECT_EQ(loss_pyramid_levels(7679, 9000), 3);
    EXPECT_EQ(loss_pyramid_levels(7680, 7680), 4);
    EXPECT_EQ(loss_pyramid_levels(30000, 30000), 4);
}

// Catches either switch changing the loss while it has nothing to do: both off, or the pyramid on
// below 1920 px (one scale), must be today's fused / pure-SSIM / L1 loss bit for bit.
TEST_F(BlurTolerantSSIMTest, OffAndSingleScaleAreTodaysLoss) {
    constexpr int C = 3, H = 64, W = 80;
    const auto x = image_tensor(random_texture(C, H, W, 1), C, H, W);
    const auto y = image_tensor(random_texture(C, H, W, 2), C, H, W);
    for (const float lambda : {0.0f, 0.2f, 1.0f}) {
        lfs::training::losses::PhotometricLoss off_loss;
        lfs::training::losses::PhotometricLoss pyramid_loss;
        auto off = off_loss.forward(x, y, {.lambda_dssim = lambda});
        auto pyramid = pyramid_loss.forward(x, y, {.lambda_dssim = lambda, .loss_pyramid = true});
        ASSERT_TRUE(off.has_value()) << off.error();
        ASSERT_TRUE(pyramid.has_value()) << pyramid.error();
        expect_bitwise_equal(off->first, pyramid->first, "loss");
        expect_bitwise_equal(off->second.grad_image, pyramid->second.grad_image, "gradient");
        EXPECT_EQ(off_loss.last_forward_used_pure_ssim(), lambda == 1.0f);
        EXPECT_EQ(pyramid_loss.last_forward_used_pure_ssim(), lambda == 1.0f);
        EXPECT_TRUE(pyramid_loss.pyramid_workspace().levels.empty());
    }

    // The fused path is the kernel pair it always was.
    LossWorkspaceArena arena;
    auto& ws = arena.fused();
    auto [loss, ctx] = fused_l1_ssim_forward(x, y, 0.2f, ws, true);
    const float direct_loss = loss.item<float>();
    const auto direct_grad = fused_l1_ssim_backward(ctx, ws).cpu().to_vector();
    lfs::training::losses::PhotometricLoss off_loss;
    auto off = off_loss.forward(x, y, {.lambda_dssim = 0.2f});
    ASSERT_TRUE(off.has_value());
    EXPECT_EQ(off->first.item<float>(), direct_loss);
    EXPECT_EQ(off->second.grad_image.cpu().to_vector(), direct_grad);

    // Weighted: one scale without the blur term is the masked fused loss.
    std::vector<float> weight(static_cast<size_t>(H) * W, 1.0f);
    for (int i = 0; i < H * W; i += 3)
        weight[i] = 0.0f;
    const auto weight_tensor = Tensor::from_vector(weight, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
    LossWorkspaceArena masked_arena;
    auto& masked_ws = masked_arena.masked_fused();
    auto [masked_loss, masked_ctx] = masked_fused_l1_ssim_forward(x, y, weight_tensor, 0.2f, masked_ws);
    const float masked_direct = masked_loss.item<float>();
    const auto masked_grad = masked_fused_l1_ssim_backward(masked_ctx, masked_ws).cpu().to_vector();
    LossWorkspaceArena pyramid_arena;
    PhotometricPyramidWorkspace pyramid_ws;
    const auto single = photometric_pyramid_loss(x, y, weight_tensor, 0.2f, 1, false, pyramid_arena, pyramid_ws);
    EXPECT_EQ(single.loss.item<float>(), masked_direct);
    EXPECT_EQ(single.grad.cpu().to_vector(), masked_grad);
    EXPECT_EQ(single.mask_sum_value, masked_ctx.mask_sum_value);
    EXPECT_TRUE(pyramid_ws.levels.empty());
}

// Catches a blur adjoint that is not the transpose of the blur, or a blur that lets pixels outside
// the image or the weight map in.
TEST_F(BlurTolerantSSIMTest, BlurAdjointIsTheTransposeOverKeptPixels) {
    constexpr int C = 2, H = 23, W = 31;
    auto x = random_texture(C, H, W, 3);
    const auto g = random_texture(C, H, W, 4, -1.0f, 1.0f);
    std::vector<float> weight(static_cast<size_t>(H) * W, 1.0f);
    for (int y = 6; y < 15; ++y)
        for (int xx = 10; xx < 19; ++xx)
            weight[y * W + xx] = 0.0f;
    weight[2 * W + 3] = 0.5f; // a fractional weight is kept like a full one
    const auto weight_tensor = Tensor::from_vector(weight, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);

    for (const bool weighted : {false, true}) {
        for (const float sigma : {1.5f, 0.75f, 0.1875f}) {
            const Tensor w = weighted ? weight_tensor : Tensor{};
            const auto gx = normalized_gaussian_blur(image_tensor(x, C, H, W), w, sigma).cpu().to_vector();
            const auto gtg = normalized_gaussian_blur_adjoint(image_tensor(g, C, H, W), w, sigma).cpu().to_vector();
            double lhs = 0.0, rhs = 0.0;
            for (size_t i = 0; i < x.size(); ++i) {
                lhs += static_cast<double>(gx[i]) * g[i];
                rhs += static_cast<double>(x[i]) * gtg[i];
            }
            EXPECT_NEAR(lhs, rhs, 1.0e-5 * std::abs(lhs) + 1.0e-5) << "weighted " << weighted << " sigma " << sigma;

            // A constant stays constant at the border and next to the hole.
            std::vector<float> constant(x.size(), 0.375f);
            const auto blurred_constant =
                normalized_gaussian_blur(image_tensor(constant, C, H, W), w, sigma).cpu().to_vector();
            for (size_t i = 0; i < constant.size(); ++i) {
                if (weighted && weight[i % (static_cast<size_t>(H) * W)] == 0.0f)
                    continue;
                ASSERT_NEAR(blurred_constant[i], 0.375f, 1.0e-6f) << i;
            }
            if (!weighted)
                continue;
            // Weight-0 pixels neither enter the blur nor receive gradient.
            auto poisoned = x;
            for (size_t i = 0; i < poisoned.size(); ++i)
                if (weight[i % (static_cast<size_t>(H) * W)] == 0.0f)
                    poisoned[i] = 100.0f;
            const auto gp = normalized_gaussian_blur(image_tensor(poisoned, C, H, W), w, sigma).cpu().to_vector();
            for (size_t i = 0; i < gp.size(); ++i) {
                if (weight[i % (static_cast<size_t>(H) * W)] == 0.0f) {
                    EXPECT_EQ(gtg[i], 0.0f) << i;
                    continue;
                }
                ASSERT_FLOAT_EQ(gp[i], gx[i]) << i;
            }
        }
    }
}

// Catches a wrong blend weight, a missing blur adjoint, or a blurred-branch gradient leaking L1:
// the gradient must match finite differences where each branch wins, at the image border, and
// with a weight map (including its boundary and a hole that must get no gradient).
TEST_F(BlurTolerantSSIMTest, GradientMatchesFiniteDifferences) {
    std::vector<float> x;
    auto c = two_region_case(x);

    // The setup really has both branches winning: compare plain SSIM of the render and of its blur.
    {
        const auto s0 = ssim_forward_map(image_tensor(x, c.C, c.H, c.W), image_tensor(c.target, c.C, c.H, c.W), false)
                            .ssim_map.cpu()
                            .to_vector();
        const auto xb = blurred(x, c.C, c.H, c.W);
        const auto s1 = ssim_forward_map(image_tensor(xb, c.C, c.H, c.W), image_tensor(c.target, c.C, c.H, c.W), false)
                            .ssim_map.cpu()
                            .to_vector();
        int blurred_wins = 0, sharp_wins = 0, left = 0, right = 0;
        for (int ch = 0; ch < c.C; ++ch)
            for (int y = 6; y < c.H - 6; ++y)
                for (int xx = 6; xx < c.W - 6; ++xx) {
                    const size_t i = (static_cast<size_t>(ch) * c.H + y) * c.W + xx;
                    if (xx < c.W / 2 - 6) {
                        ++left;
                        blurred_wins += s1[i] > s0[i] + 0.05f;
                    } else if (xx >= c.W / 2 + 6) {
                        ++right;
                        sharp_wins += s0[i] > s1[i] + 0.05f;
                    }
                }
        EXPECT_GT(blurred_wins, left * 9 / 10);
        EXPECT_GT(sharp_wins, right * 9 / 10);
    }

    const std::vector<std::array<int, 3>> pixels = {
        {0, 20, 8},  // blurred branch wins
        {1, 20, 40}, // sharp branch wins
        {2, 20, 24}, // where the regions meet
        {0, 0, 0},   // image corner: border blur normalisation and the valid-padding crop
        {1, 39, 47},
        {2, 3, 45},
    };
    expect_gradient_matches_finite_differences(c, x, pixels, "unweighted");

    // Weight map: a hole, a fractional band and a rim.
    c.weight.assign(static_cast<size_t>(c.H) * c.W, 1.0f);
    for (int y = 12; y < 26; ++y)
        for (int xx = 14; xx < 30; ++xx)
            c.weight[y * c.W + xx] = 0.0f;
    for (int y = 30; y < 34; ++y)
        for (int xx = 0; xx < c.W; ++xx)
            c.weight[y * c.W + xx] = 0.5f;
    const std::vector<std::array<int, 3>> weighted_pixels = {
        {0, 20, 8},
        {1, 20, 40},
        {2, 11, 20}, // just outside the hole
        {0, 12, 13},
        {1, 31, 5}, // beside the hole, in the fractional band
        {2, 0, 47},
        {0, 18, 20}, // a corner; inside the hole
    };
    expect_gradient_matches_finite_differences(c, x, weighted_pixels, "weighted");
    const auto weighted = evaluate(c, x);
    for (int ch = 0; ch < c.C; ++ch)
        for (int y = 12; y < 26; ++y)
            for (int xx = 14; xx < 30; ++xx)
                ASSERT_EQ(weighted.grad[(static_cast<size_t>(ch) * c.H + y) * c.W + xx], 0.0f);
}

// Catches a soft maximum that is not the plain SSIM for equal branches: a render that is constant
// per channel is its own blur, so the loss equals the plain loss.
TEST_F(BlurTolerantSSIMTest, EqualBranchesGiveThePlainSSIM) {
    constexpr int C = 3, H = 40, W = 48;
    std::vector<float> x(static_cast<size_t>(C) * H * W);
    for (int ch = 0; ch < C; ++ch)
        std::fill_n(x.begin() + static_cast<size_t>(ch) * H * W, H * W, 0.3f + 0.2f * ch);
    LossCase c;
    c.target = random_texture(C, H, W, 9);
    c.ssim_weight = 1.0f;
    const float tolerant = evaluate(c, x).loss;
    c.blur_tolerant = false;
    const float plain = evaluate(c, x).loss;
    EXPECT_NEAR(tolerant, plain, 1.0e-6f);
}

// Catches the term not rewarding sharpness: a sharp render against a blurred photo scores better
// than under plain SSIM; a render blurrier than the photo scores as plain SSIM, at most ln2 / 50 lower.
TEST_F(BlurTolerantSSIMTest, SharperRenderScoresHigherThanUnderPlainSSIM) {
    constexpr int C = 3, H = 40, W = 48;
    const auto sharp = random_texture(C, H, W, 21);
    const auto soft = blurred(sharp, C, H, W);
    LossCase c;
    c.ssim_weight = 1.0f; // loss = 1 - mean S
    c.target = soft;
    const float tolerant = evaluate(c, sharp).loss;
    c.blur_tolerant = false;
    const float plain = evaluate(c, sharp).loss;
    EXPECT_LT(tolerant, 0.5f * plain) << "tolerant " << tolerant << " plain " << plain;
    EXPECT_LT(tolerant, 0.05f);

    c.target = sharp;
    c.blur_tolerant = true;
    const float tolerant_blurry = evaluate(c, soft).loss;
    c.blur_tolerant = false;
    const float plain_blurry = evaluate(c, soft).loss;
    EXPECT_GE(tolerant_blurry, plain_blurry - 1.0e-4f);
    EXPECT_LE(tolerant_blurry, plain_blurry + std::log(2.0f) / BLUR_TOLERANT_SSIM_TEMPERATURE + 1.0e-4f);
}

// Catches pooling that is not a 2x2 average (odd edges average what they have) or a scale average
// that is not 1 / K: two scales equal the mean of today's loss at full and at pooled resolution.
TEST_F(BlurTolerantSSIMTest, PyramidIsTheMeanOfTheLossOverPooledScales) {
    constexpr int C = 3, H = 45, W = 37;
    const auto x = random_texture(C, H, W, 31);
    const auto y = random_texture(C, H, W, 32);
    const int h = (H + 1) / 2, w = (W + 1) / 2;
    const auto px = pool_cpu(x, C, H, W);
    const auto py = pool_cpu(y, C, H, W);
    std::vector<float> weight(static_cast<size_t>(H) * W);
    std::mt19937 rng(33);
    std::uniform_int_distribution<int> keep(0, 3);
    for (auto& v : weight)
        v = keep(rng) == 0 ? 0.0f : 1.0f;
    const auto pw = pool_cpu(weight, 1, H, W);

    for (const bool weighted : {false, true}) {
        LossCase c;
        c.C = C;
        c.H = H;
        c.W = W;
        c.target = y;
        c.ssim_weight = 0.2f;
        c.levels = 2;
        c.blur_tolerant = false;
        if (weighted)
            c.weight = weight;
        const float pyramid = evaluate(c, x).loss;

        LossWorkspaceArena arena;
        float full = 0.0f, coarse = 0.0f;
        if (weighted) {
            auto& ws = arena.masked_fused();
            full = masked_fused_l1_ssim_forward(image_tensor(x, C, H, W), image_tensor(y, C, H, W),
                                                Tensor::from_vector(weight, {size_t{H}, size_t{W}}, Device::CUDA),
                                                0.2f, ws)
                       .first.item<float>();
            coarse = masked_fused_l1_ssim_forward(image_tensor(px, C, h, w), image_tensor(py, C, h, w),
                                                  Tensor::from_vector(pw, {size_t(h), size_t(w)}, Device::CUDA), 0.2f,
                                                  ws)
                         .first.item<float>();
        } else {
            auto& ws = arena.fused();
            full = fused_l1_ssim_forward(image_tensor(x, C, H, W), image_tensor(y, C, H, W), 0.2f, ws)
                       .first.item<float>();
            coarse = fused_l1_ssim_forward(image_tensor(px, C, h, w), image_tensor(py, C, h, w), 0.2f, ws)
                         .first.item<float>();
        }
        EXPECT_NEAR(pyramid, 0.5f * (full + coarse), 2.0e-6f) << "weighted " << weighted;
    }
}

// Catches a pooling adjoint that drops or doubles gradient (odd last row and column, three scales)
// and the two switches not composing: gradients match finite differences with and without the
// blur term, with and without a weight map.
TEST_F(BlurTolerantSSIMTest, PyramidGradientMatchesFiniteDifferences) {
    LossCase c;
    c.H = 45;
    c.W = 37;
    const auto x = random_texture(c.C, c.H, c.W, 41);
    const auto soft = blurred(x, c.C, c.H, c.W);
    c.target = soft;
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.03f);
    for (size_t i = 0; i < c.target.size(); ++i)
        if ((i % c.W) >= static_cast<size_t>(c.W / 2))
            c.target[i] = std::clamp(x[i] + noise(rng), 0.0f, 1.0f);

    const std::vector<std::array<int, 3>> pixels = {
        {0, 44, 36}, // odd last row and column
        {1, 44, 10},
        {2, 20, 36},
        {0, 22, 9},
        {1, 0, 0},
    };
    for (const int levels : {2, 3}) {
        for (const bool blur : {false, true}) {
            for (const bool weighted : {false, true}) {
                c.levels = levels;
                c.blur_tolerant = blur;
                c.weight.clear();
                if (weighted) {
                    c.weight.assign(static_cast<size_t>(c.H) * c.W, 1.0f);
                    for (int y = 10; y < 20; ++y)
                        for (int xx = 5; xx < 15; ++xx)
                            c.weight[y * c.W + xx] = 0.0f;
                }
                const auto label = std::string("levels ") + std::to_string(levels) + (blur ? " blur" : "") +
                                   (weighted ? " weighted" : "");
                expect_gradient_matches_finite_differences(c, x, pixels, label.c_str());
            }
        }
    }
}

// Catches a fixed loss denominator (thin-structure weighting) not following the scale's pixel count.
TEST_F(BlurTolerantSSIMTest, FixedDenominatorScalesWithThePixelCount) {
    LossCase c;
    c.H = 44;
    c.W = 52;
    const auto x = random_texture(c.C, c.H, c.W, 51);
    c.target = random_texture(c.C, c.H, c.W, 52);
    c.weight.assign(static_cast<size_t>(c.H) * c.W, 1.0f);
    c.levels = 2;
    c.blur_tolerant = false;
    const float normalized = evaluate(c, x).loss;
    c.denominator = static_cast<float>(3 * c.H * c.W);
    const auto fixed = evaluate(c, x);
    EXPECT_NEAR(fixed.loss, normalized, 2.0e-6f);
    EXPECT_EQ(fixed.mask_sum, c.denominator);
}

// Catches the decoupled pyramid loss (appearance correction) drifting from today's decoupled loss
// when it has nothing to do: one scale without the blur term gives the same loss, corrected-image
// gradient and raw-render gradient bit for bit, with and without a weight map.
TEST_F(BlurTolerantSSIMTest, DecoupledOffIsTodaysDecoupledLoss) {
    constexpr int C = 3, H = 64, W = 80;
    const auto raw = image_tensor(random_texture(C, H, W, 61), C, H, W);
    const auto corrected = image_tensor(random_texture(C, H, W, 62), C, H, W);
    const auto y = image_tensor(random_texture(C, H, W, 63), C, H, W);
    std::vector<float> weight(static_cast<size_t>(H) * W, 1.0f);
    for (int i = 0; i < H * W; i += 3)
        weight[i] = i % 2 ? 0.0f : 0.5f;
    const auto weight_tensor = Tensor::from_vector(weight, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);

    for (const bool weighted : {false, true}) {
        LossWorkspaceArena arena;
        float direct_loss = 0.0f, direct_mask_sum = 0.0f;
        std::vector<float> direct_grad;
        auto direct_raw = Tensor::zeros(raw.shape(), Device::CUDA);
        if (weighted) {
            auto& ws = arena.masked_decoupled();
            auto [loss, ctx] = masked_decoupled_fused_l1_ssim_forward(corrected, raw, y, weight_tensor, 0.2f, ws);
            direct_loss = loss.item<float>();
            direct_mask_sum = ctx.mask_sum_value;
            direct_grad = masked_decoupled_fused_l1_ssim_backward(ctx, ws).grad_corrected.cpu().to_vector();
            accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), direct_raw);
        } else {
            auto& ws = arena.decoupled();
            auto [loss, ctx] = decoupled_fused_l1_ssim_forward(corrected, raw, y, 0.2f, ws);
            direct_loss = loss.item<float>();
            direct_grad = decoupled_fused_l1_ssim_backward(ctx, ws).grad_corrected.cpu().to_vector();
            accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), direct_raw);
        }

        LossWorkspaceArena pyramid_arena;
        PhotometricPyramidWorkspace pyramid_ws;
        const auto single = decoupled_photometric_pyramid_loss(corrected, raw, y, weighted ? weight_tensor : Tensor{},
                                                               0.2f, 1, false, pyramid_arena, pyramid_ws);
        EXPECT_EQ(single.loss.item<float>(), direct_loss) << "weighted " << weighted;
        EXPECT_EQ(single.grad.cpu().to_vector(), direct_grad) << "weighted " << weighted;
        EXPECT_EQ(single.mask_sum_value, direct_mask_sum) << "weighted " << weighted;
        // The raw gradient is carried precomputed and added as is.
        auto carried = Tensor::zeros(raw.shape(), Device::CUDA);
        accumulate_decoupled_raw_gradient(decoupled_raw_gradient(single, raw), carried);
        EXPECT_EQ(carried.cpu().to_vector(), direct_raw.cpu().to_vector()) << "weighted " << weighted;
    }
}

// Catches the decoupled blurred branch blurring only one image or scaling its partials wrongly:
// with the corrected image equal to the raw render the decoupled loss is the plain loss, and the two
// gradients add up to the plain gradient (blur term and pyramid, with and without weights).
TEST_F(BlurTolerantSSIMTest, DecoupledWithEqualImagesIsThePlainLoss) {
    std::vector<float> x;
    auto c = two_region_case(x);
    std::vector<float> weight(static_cast<size_t>(c.H) * c.W, 1.0f);
    for (int y = 12; y < 26; ++y)
        for (int xx = 14; xx < 30; ++xx)
            weight[y * c.W + xx] = 0.0f;
    for (const bool blur : {false, true}) {
        for (const int levels : {1, 2}) {
            for (const bool weighted : {false, true}) {
                c.blur_tolerant = blur;
                c.levels = levels;
                c.weight = weighted ? weight : std::vector<float>{};
                c.raw.clear();
                const auto plain = evaluate(c, x);
                c.raw = x;
                const auto decoupled = evaluate(c, x);
                const auto label = std::string("levels ") + std::to_string(levels) + (blur ? " blur" : "") +
                                   (weighted ? " weighted" : "");
                EXPECT_NEAR(decoupled.loss, plain.loss, 2.0e-6f) << label;
                ASSERT_EQ(decoupled.grad_raw.size(), plain.grad.size()) << label;
                // The two paths round their SSIM partials to fp16 differently (one partial or a split pair),
                // which shows most in the low-variance windows of the blurred branch.
                double scale = 0.0, deviation = 0.0, rms = 0.0;
                for (size_t i = 0; i < plain.grad.size(); ++i) {
                    scale = std::max(scale, static_cast<double>(std::abs(plain.grad[i])));
                    const double d = static_cast<double>(decoupled.grad[i]) + decoupled.grad_raw[i] - plain.grad[i];
                    deviation = std::max(deviation, std::abs(d));
                    rms += d * d;
                }
                rms = std::sqrt(rms / static_cast<double>(plain.grad.size()));
                std::printf("[ equal images ] %s: max deviation %.3g, rms %.3g of max |g| %.3g\n", label.c_str(),
                            deviation, rms, scale);
                EXPECT_LT(deviation, 3.0e-2 * scale) << label;
                EXPECT_LT(rms, 3.0e-3 * scale) << label;
            }
        }
    }
}

// Catches a wrong decoupled blurred-branch gradient: the corrected image (a gain and offset away from
// the raw render) and the raw render both match finite differences, with the pyramid and a weight map.
TEST_F(BlurTolerantSSIMTest, DecoupledGradientMatchesFiniteDifferences) {
    std::vector<float> raw;
    auto c = two_region_case(raw);
    std::vector<float> corrected(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        corrected[i] = 0.9f * raw[i] + 0.06f;
    c.raw = raw;
    const std::vector<std::array<int, 3>> pixels = {{0, 20, 8}, {1, 20, 40}, {2, 20, 24}, {0, 0, 0}, {1, 39, 47}};
    std::vector<float> weight(static_cast<size_t>(c.H) * c.W, 1.0f);
    for (int y = 12; y < 26; ++y)
        for (int xx = 14; xx < 30; ++xx)
            weight[y * c.W + xx] = 0.0f;
    for (const int levels : {1, 2}) {
        for (const bool weighted : {false, true}) {
            c.levels = levels;
            c.weight = weighted ? weight : std::vector<float>{};
            const auto label = std::string("decoupled levels ") + std::to_string(levels) + (weighted ? " weighted" : "");
            expect_gradient_matches_finite_differences(c, corrected, pixels, (label + " corrected").c_str());
            expect_gradient_matches_finite_differences(c, corrected, pixels, (label + " raw").c_str(),
                                                       /*perturb_raw=*/true);
        }
    }
}

namespace {

    // CPU contrast-structure error of one scale: 1 - mean_c(w0 cs(x, y) + w1 cs(b, y)), 11-tap Gaussian
    // windows (sigma 1.5) with zero padding, w0 the soft-maximum weight of the sharp branch; b empty: cs(x, y).
    std::vector<double> cs_error_cpu(const std::vector<float>& x, const std::vector<float>& b,
                                     const std::vector<float>& y, const int C, const int H, const int W) {
        double g[11];
        double gsum = 0.0;
        for (int d = -5; d <= 5; ++d)
            gsum += g[d + 5] = std::exp(-0.5 * d * d / (1.5 * 1.5));
        for (double& v : g)
            v /= gsum;
        constexpr double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
        std::vector<double> out(static_cast<size_t>(H) * W, 0.0);
        for (int ch = 0; ch < C; ++ch)
            for (int py = 0; py < H; ++py)
                for (int px = 0; px < W; ++px) {
                    double s[8] = {};
                    for (int dy = -5; dy <= 5; ++dy)
                        for (int dx = -5; dx <= 5; ++dx) {
                            const int yy = py + dy, xx = px + dx;
                            if (yy < 0 || yy >= H || xx < 0 || xx >= W)
                                continue;
                            const size_t i = (static_cast<size_t>(ch) * H + yy) * W + xx;
                            const double w = g[dy + 5] * g[dx + 5];
                            const double X = x[i], B = b.empty() ? 0.0 : b[i], Y = y[i];
                            s[0] += w * X, s[1] += w * X * X, s[2] += w * X * Y;
                            s[3] += w * B, s[4] += w * B * B, s[5] += w * B * Y;
                            s[6] += w * Y, s[7] += w * Y * Y;
                        }
                    const double my = s[6], vy = s[7] - my * my;
                    const auto branch = [&](const double m, const double sq, const double cross, double& cs) {
                        cs = (2.0 * (cross - m * my) + C2) / (sq - m * m + vy + C2);
                        return (2.0 * m * my + C1) / (m * m + my * my + C1) * cs;
                    };
                    double cs0 = 0.0, cs1 = 0.0;
                    const double S0 = branch(s[0], s[1], s[2], cs0);
                    double blend = cs0;
                    if (!b.empty()) {
                        const double S1 = branch(s[3], s[4], s[5], cs1);
                        const double w0 = 1.0 / (1.0 + std::exp(-BLUR_TOLERANT_SSIM_TEMPERATURE * (S0 - S1)));
                        blend = w0 * cs0 + (1.0 - w0) * cs1;
                    }
                    out[static_cast<size_t>(py) * W + px] += blend / C;
                }
        for (auto& v : out)
            v = std::max(1.0 - v, 0.0);
        return out;
    }

} // namespace

// Catches a densification error map that is not the loss's own: per scale the soft-maximum blend of
// the two branches' contrast-structure terms, averaged over the scales with each coarse value
// repeated over the pixels it pools; none without either switch.
TEST_F(BlurTolerantSSIMTest, ErrorMapBlendsTheBranchesOverTheScales) {
    std::vector<float> x;
    const LossCase c = two_region_case(x);
    for (const int levels : {1, 2}) {
        for (const bool blur : {false, true}) {
            const std::string label = "levels " + std::to_string(levels) + (blur ? " blur" : " plain");
            LossWorkspaceArena arena;
            PhotometricPyramidWorkspace workspace;
            workspace.want_cs_error_map = true;
            (void)photometric_pyramid_loss(image_tensor(x, c.C, c.H, c.W), image_tensor(c.target, c.C, c.H, c.W),
                                           Tensor{}, 0.2f, levels, blur, arena, workspace);
            if (levels == 1 && !blur) {
                EXPECT_FALSE(workspace.cs_error_map_ready) << label;
                continue;
            }
            ASSERT_TRUE(workspace.cs_error_map_ready) << label;
            const auto map = workspace.cs_error_map.cpu().to_vector();
            ASSERT_EQ(map.size(), static_cast<size_t>(c.H) * c.W) << label;

            std::vector<double> expected(map.size(), 0.0);
            auto xs = x, ys = c.target;
            int H = c.H, W = c.W;
            for (int s = 0; s < levels; ++s) {
                if (s > 0) {
                    xs = pool_cpu(xs, c.C, H, W);
                    ys = pool_cpu(ys, c.C, H, W);
                    H = (H + 1) / 2, W = (W + 1) / 2;
                }
                const auto bs = blur ? blurred(xs, c.C, H, W, BLUR_TOLERANT_SSIM_SIGMA / static_cast<float>(1 << s))
                                     : std::vector<float>{};
                const auto level = cs_error_cpu(xs, bs, ys, c.C, H, W);
                for (int py = 0; py < c.H; ++py)
                    for (int px = 0; px < c.W; ++px)
                        expected[static_cast<size_t>(py) * c.W + px] +=
                            level[static_cast<size_t>(py >> s) * W + (px >> s)] / levels;
            }
            double max_err = 0.0;
            for (size_t i = 0; i < map.size(); ++i)
                max_err = std::max(max_err, std::abs(map[i] - expected[i]));
            EXPECT_LT(max_err, 2.0e-4) << label;
        }
    }

    // The blend differs from the sharp branch alone where the photo is the blurred render.
    LossWorkspaceArena arena;
    PhotometricPyramidWorkspace workspace;
    workspace.want_cs_error_map = true;
    (void)photometric_pyramid_loss(image_tensor(x, c.C, c.H, c.W), image_tensor(c.target, c.C, c.H, c.W), Tensor{},
                                   0.2f, 1, true, arena, workspace);
    const auto tolerant = workspace.cs_error_map.cpu().to_vector();
    const auto plain = cs_error_cpu(x, {}, c.target, c.C, c.H, c.W);
    double left_tolerant = 0.0, left_plain = 0.0;
    for (int py = 0; py < c.H; ++py)
        for (int px = 8; px < c.W / 2 - 8; ++px) {
            left_tolerant += tolerant[static_cast<size_t>(py) * c.W + px];
            left_plain += plain[static_cast<size_t>(py) * c.W + px];
        }
    EXPECT_LT(left_tolerant, 0.5 * left_plain);
}

namespace {

    struct PyramidRun {
        float loss = 0.0f;
        float mask_sum = 0.0f;
        std::vector<float> grad, grad_raw, ssim_map, cs_map;
    };

    // One pyramid loss call with the fused or the per-branch (reference) blur-tolerant kernels; the
    // scale-0 SSIM and CS maps come from the arena workspace the loss used.
    PyramidRun run_pyramid(const Tensor& x, const Tensor& raw, const Tensor& y, const Tensor& weight,
                           const float ssim_weight, const int levels, const bool reference,
                           const float denominator = 0.0f) {
        LossWorkspaceArena arena;
        PhotometricPyramidWorkspace workspace;
        workspace.reference_kernels = reference;
        const bool weighted = weight.is_valid();
        PhotometricPyramidResult result;
        if (raw.is_valid())
            result = decoupled_photometric_pyramid_loss(x, raw, y, weight, ssim_weight, levels, true, arena, workspace,
                                                        denominator);
        else
            result = photometric_pyramid_loss(x, y, weight, ssim_weight, levels, true, arena, workspace, denominator);
        PyramidRun run{.loss = result.loss.item<float>(), .mask_sum = result.mask_sum_value, .grad = result.grad.cpu().to_vector()};
        if (result.grad_raw.is_valid())
            run.grad_raw = result.grad_raw.cpu().to_vector();
        const auto maps = [&](const auto& ws) {
            run.ssim_map = ws.ssim_map.cpu().to_vector();
            run.cs_map = ws.cs_map.cpu().to_vector();
        };
        if (raw.is_valid() && weighted)
            maps(arena.masked_decoupled());
        else if (raw.is_valid())
            maps(arena.decoupled());
        else if (weighted)
            maps(arena.masked_fused());
        else
            maps(arena.fused());
        return run;
    }

    // ||a - b|| / ||b||, and the largest element difference relative to the largest |b|.
    void expect_close(const std::vector<float>& a, const std::vector<float>& b, const double relative,
                      const std::string& label) {
        ASSERT_EQ(a.size(), b.size()) << label;
        ASSERT_FALSE(b.empty()) << label;
        double diff = 0.0, norm = 0.0, max_diff = 0.0, max_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            ASSERT_TRUE(std::isfinite(a[i])) << label << " at " << i;
            const double d = static_cast<double>(a[i]) - b[i];
            diff += d * d;
            norm += static_cast<double>(b[i]) * b[i];
            max_diff = std::max(max_diff, std::abs(d));
            max_b = std::max(max_b, std::abs(static_cast<double>(b[i])));
        }
        ASSERT_GT(norm, 0.0) << label;
        EXPECT_LE(std::sqrt(diff / norm), relative) << label;
        EXPECT_LE(max_diff, 10.0 * relative * max_b) << label;
    }

} // namespace

// Catches a fused blur-tolerant level that is not the per-branch kernels' loss: loss, render and raw
// gradients and the scale-0 maps agree within 1e-5 relative for the plain, weighted, decoupled and
// weighted decoupled losses, over one to three scales with odd edges, a hole in the weight map
// (whole ignored tiles), fractional weights, a byte photo and a fixed denominator.
TEST_F(BlurTolerantSSIMTest, FusedLevelsMatchThePerBranchKernels) {
    constexpr int C = 3, H = 75, W = 93;
    const auto render = random_texture(C, H, W, 21);
    auto photo = blurred(render, C, H, W);
    {
        std::mt19937 rng(22);
        std::normal_distribution<float> noise(0.0f, 0.04f);
        for (size_t i = 0; i < photo.size(); ++i)
            if ((i % W) >= static_cast<size_t>(W / 2)) // right half: sharp photo with noise
                photo[i] = std::clamp(render[i] + noise(rng), 0.0f, 1.0f);
    }
    std::vector<float> raw(render.size());
    for (size_t i = 0; i < raw.size(); ++i)
        raw[i] = std::clamp(1.1f * render[i] - 0.04f, 0.0f, 1.0f);
    std::vector<float> weight(static_cast<size_t>(H) * W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            weight[static_cast<size_t>(y) * W + x] = (y >= 20 && y < 60 && x >= 16 && x < 64) ? 0.0f
                                                                                              : 1.0f + 0.5f * ((x * 7 + y * 3) % 5) / 4.0f;
    std::vector<float> photo_levels(photo.size());
    for (size_t i = 0; i < photo.size(); ++i)
        photo_levels[i] = static_cast<float>(std::lround(photo[i] * 255.0f));

    const auto x = image_tensor(render, C, H, W);
    const auto y = image_tensor(photo, C, H, W);
    const auto y_bytes = image_tensor(photo_levels, C, H, W).to(DataType::UInt8);
    const auto r = image_tensor(raw, C, H, W);
    const auto w = Tensor::from_vector(weight, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
    for (const bool decoupled : {false, true}) {
        for (const bool weighted : {false, true}) {
            for (const int levels : {1, 2, 3}) {
                for (const bool bytes : {false, true}) {
                    if (bytes && levels == 2)
                        continue;
                    const float denominator = weighted && levels == 3 ? 0.9f * static_cast<float>(C * H * W) : 0.0f;
                    const std::string label = std::string(decoupled ? "decoupled" : "plain") +
                                              (weighted ? " weighted" : "") + " levels " + std::to_string(levels) +
                                              (bytes ? " byte photo" : "");
                    const Tensor& photo_tensor = bytes ? y_bytes : y;
                    const Tensor raw_tensor = decoupled ? r : Tensor{};
                    const Tensor weight_tensor = weighted ? w : Tensor{};
                    const auto fused = run_pyramid(x, raw_tensor, photo_tensor, weight_tensor, 0.2f, levels, false,
                                                   denominator);
                    const auto reference = run_pyramid(x, raw_tensor, photo_tensor, weight_tensor, 0.2f, levels,
                                                       true, denominator);
                    EXPECT_NEAR(fused.loss, reference.loss, 1.0e-5 * std::abs(reference.loss)) << label;
                    EXPECT_EQ(fused.mask_sum, reference.mask_sum) << label;
                    expect_close(fused.grad, reference.grad, 1.0e-5, label + " gradient");
                    if (decoupled)
                        expect_close(fused.grad_raw, reference.grad_raw, 1.0e-5, label + " raw gradient");
                    else
                        EXPECT_TRUE(fused.grad_raw.empty()) << label;
                    expect_close(fused.ssim_map, reference.ssim_map, 1.0e-5, label + " SSIM map");
                    expect_close(fused.cs_map, reference.cs_map, 1.0e-5, label + " CS map");
                }
            }
        }
    }
}

// Catches pyramid buffers that keep data or shape from an earlier photo: one workspace and arena
// take photos of different sizes (larger, smaller, larger again, as the photos of a capture differ)
// and each loss and gradient equals that of fresh buffers.
TEST_F(BlurTolerantSSIMTest, FusedLevelsReuseTheirBuffersAcrossPhotoSizes) {
    constexpr int C = 3;
    LossWorkspaceArena arena;
    PhotometricPyramidWorkspace workspace;
    unsigned seed = 40;
    for (const auto [H, W] : {std::pair{75, 93}, std::pair{61, 80}, std::pair{70, 91}}) {
        const auto render = random_texture(C, H, W, seed++);
        const auto photo = blurred(render, C, H, W);
        std::vector<float> raw(render.size());
        for (size_t i = 0; i < raw.size(); ++i)
            raw[i] = std::clamp(0.9f * render[i] + 0.05f, 0.0f, 1.0f);
        std::vector<float> weight(static_cast<size_t>(H) * W);
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = 1.0f + 0.25f * static_cast<float>(i % 3);
        const auto x = image_tensor(render, C, H, W);
        const auto y = image_tensor(photo, C, H, W);
        const auto r = image_tensor(raw, C, H, W);
        const auto w = Tensor::from_vector(weight, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
        const float denominator = static_cast<float>(C * H * W);
        const std::string label = std::to_string(W) + "x" + std::to_string(H);
        const auto reused = decoupled_photometric_pyramid_loss(x, r, y, w, 0.2f, 3, true, arena, workspace,
                                                               denominator);
        const float loss = reused.loss.item<float>();
        const auto grad = reused.grad.cpu().to_vector();
        const auto grad_raw = reused.grad_raw.cpu().to_vector();
        const auto fresh = run_pyramid(x, r, y, w, 0.2f, 3, false, denominator);
        EXPECT_NEAR(loss, fresh.loss, 1.0e-6 * std::abs(fresh.loss)) << label;
        expect_close(grad, fresh.grad, 1.0e-6, label + " gradient");
        expect_close(grad_raw, fresh.grad_raw, 1.0e-6, label + " raw gradient");
    }
}

namespace {
    // Wall time per call of fn, in ms, after warmup calls.
    template <typename Fn>
    double time_ms(Fn&& fn, const int warmup = 3, const int iterations = 10) {
        for (int i = 0; i < warmup; ++i)
            fn();
        cudaDeviceSynchronize();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            fn();
        cudaDeviceSynchronize();
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(stop - start).count() / iterations;
    }
} // namespace

// Timing at the full resolution of a 20 Mpx photo (not a check; run with
// --gtest_also_run_disabled_tests): plain SSIM against the blur-tolerant pyramid with the per-branch
// and the fused kernels, for the plain, weighted, decoupled and weighted decoupled losses.
TEST_F(BlurTolerantSSIMTest, DISABLED_FullResolutionTiming) {
    constexpr int C = 3, H = 3933, W = 5246;
    std::vector<float> render(static_cast<size_t>(C) * H * W), photo(render.size()), raw(render.size());
    std::vector<float> weight(static_cast<size_t>(H) * W);
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 0.03f);
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t i = (static_cast<size_t>(c) * H + y) * W + x;
                const float v = 0.5f + 0.25f * std::sin(0.013f * x + 0.7f * c) * std::cos(0.021f * y) +
                                0.1f * std::sin(0.37f * x + 0.29f * y);
                photo[i] = std::clamp(v + noise(rng), 0.0f, 1.0f);
                render[i] = std::clamp(v, 0.0f, 1.0f);
                raw[i] = std::clamp(0.95f * v + 0.02f, 0.0f, 1.0f);
            }
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = 1.0f + 0.5f * static_cast<float>((i * 2654435761u) % 7) / 6.0f;
    const auto x = image_tensor(render, C, H, W);
    const auto y = image_tensor(photo, C, H, W);
    const auto r = image_tensor(raw, C, H, W);
    const auto w = Tensor::from_vector(weight, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
    const int levels = loss_pyramid_levels(H, W);
    const float denominator = static_cast<float>(C) * H * W;

    // A profiler range, so a trace can capture just the timed calls.
    cudaProfilerStart();
    for (const bool decoupled : {false, true}) {
        for (const bool weighted : {false, true}) {
            const Tensor raw_tensor = decoupled ? r : Tensor{};
            const Tensor weight_tensor = weighted ? w : Tensor{};
            const float den = weighted ? denominator : 0.0f;
            LossWorkspaceArena arena;
            PhotometricPyramidWorkspace plain_ws, reference_ws, fused_ws;
            reference_ws.reference_kernels = true;
            const auto call = [&](PhotometricPyramidWorkspace& ws, const int K, const bool blur) {
                if (decoupled) {
                    auto result = decoupled_photometric_pyramid_loss(x, raw_tensor, y, weight_tensor, 0.2f, K, blur,
                                                                     arena, ws, den);
                    auto grad = result.grad;
                    accumulate_decoupled_raw_gradient(decoupled_raw_gradient(result, raw_tensor), grad);
                } else {
                    (void)photometric_pyramid_loss(x, y, weight_tensor, 0.2f, K, blur, arena, ws, den);
                }
            };
            // Plain SSIM: one scale without the blur term (the decoupled loss through its own kernels).
            const auto plain_call = [&] {
                if (decoupled) {
                    if (weighted) {
                        auto& ws = arena.masked_decoupled();
                        auto [loss, ctx] = masked_decoupled_fused_l1_ssim_forward(x, raw_tensor, y, weight_tensor, 0.2f,
                                                                                  ws, den);
                        auto grads = masked_decoupled_fused_l1_ssim_backward(ctx, ws);
                        accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), grads.grad_corrected);
                    } else {
                        auto& ws = arena.decoupled();
                        auto [loss, ctx] = decoupled_fused_l1_ssim_forward(x, raw_tensor, y, 0.2f, ws);
                        auto grads = decoupled_fused_l1_ssim_backward(ctx, ws);
                        accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), grads.grad_corrected);
                    }
                } else {
                    call(plain_ws, 1, false);
                }
            };
            const double plain = time_ms(plain_call);
            const double reference = time_ms([&] { call(reference_ws, levels, true); });
            const double fused = time_ms([&] { call(fused_ws, levels, true); });
            std::printf("[timing] %s%s %dx%d K=%d: plain %.2f ms, blur-tolerant pyramid per-branch %.2f ms "
                        "(+%.2f), fused %.2f ms (+%.2f)\n",
                        decoupled ? "decoupled" : "plain", weighted ? " weighted" : "", W, H, levels, plain,
                        reference, reference - plain, fused, fused - plain);
        }
    }
    cudaProfilerStop();
}
