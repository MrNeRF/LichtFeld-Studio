/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/losses/regularization.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;

namespace {

    Camera make_camera(int w, int h) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 4};
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        return Camera(R, T, /*fx=*/100.f, /*fy=*/100.f, /*cx=*/w * 0.5f, /*cy=*/h * 0.5f,
                      Tensor(), Tensor(), CameraModelType::PINHOLE, "test", "",
                      std::filesystem::path{}, w, h, 0);
    }

    std::unique_ptr<SplatData> make_splat(int n, int sh_degree = 0) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                p[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
                p[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
                p[i * 3 + 2] = 0.0f;
            }
            means = cpu.to(Device::CUDA);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.5f, Device::CUDA);
        const size_t sh_rest = sh_degree > 0 ? static_cast<size_t>(sh_degree * (sh_degree + 2)) : 0;
        auto shN = Tensor::zeros({static_cast<size_t>(n), sh_rest, 3}, Device::CUDA);
        // Varied raw scales so mean(exp(s)) is non-trivial.
        auto scaling_cpu = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CPU);
        float* sp = scaling_cpu.ptr<float>();
        for (int i = 0; i < n; ++i) {
            sp[i * 3 + 0] = -2.0f + 0.01f * static_cast<float>(i);
            sp[i * 3 + 1] = -1.8f - 0.005f * static_cast<float>(i % 7);
            sp[i * 3 + 2] = -2.2f + 0.003f * static_cast<float>(i % 5);
        }
        auto scaling = scaling_cpu.to(Device::CUDA);
        std::vector<float> rot(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rot[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::CUDA);
        auto opacity_cpu = Tensor::zeros({static_cast<size_t>(n)}, Device::CPU);
        float* op = opacity_cpu.ptr<float>();
        for (int i = 0; i < n; ++i) {
            op[i] = 1.5f + 0.02f * static_cast<float>(i % 11);
        }
        auto opacity = opacity_cpu.to(Device::CUDA);
        auto splat = std::make_unique<SplatData>(sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
        if (sh_degree > 0)
            splat->set_active_sh_degree(sh_degree);
        return splat;
    }

    void cleanup_arena() {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    float relative_delta(float a, float b) {
        const float denom = std::max(std::max(std::fabs(a), std::fabs(b)), 1e-12f);
        return std::fabs(a - b) / denom;
    }

    double effective_rank_reference(const double x, const double y, const double z) {
        const double peak = std::max({x, y, z});
        const double q[3] = {std::exp(2.0 * (x - peak)), std::exp(2.0 * (y - peak)), std::exp(2.0 * (z - peak))};
        const double sum = q[0] + q[1] + q[2];
        double entropy = 0.0;
        for (const double value : q) {
            const double p = value / sum;
            entropy -= p * std::log(p);
        }
        return std::max(-std::log(std::exp(entropy) - 0.99999), 0.0);
    }

} // namespace

class FusedRegLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        bg_ = Tensor::zeros({3}, Device::CUDA);
        camera_ = std::make_unique<Camera>(make_camera(64, 64));
        splat_ = make_splat(64);
    }

    void TearDown() override {
        splat_.reset();
        camera_.reset();
        cleanup_arena();
    }

    Tensor bg_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<SplatData> splat_;
};

// Equivalence: fused backward loss scalars match the legacy loss-only kernels.
TEST_F(FusedRegLossTest, FusedBackwardLossMatchesLossOnly) {
    constexpr float kScaleWeight = 0.01f;
    constexpr float kOpacityWeight = 0.02f;

    auto scale_ref = losses::ScaleRegularization::forward_loss_only(
        splat_->scaling_raw(), {.weight = kScaleWeight});
    ASSERT_TRUE(scale_ref.has_value()) << scale_ref.error();
    auto opacity_ref = losses::OpacityRegularization::forward_loss_only(
        splat_->opacity_raw(), {.weight = kOpacityWeight});
    ASSERT_TRUE(opacity_ref.has_value()) << opacity_ref.error();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const float scale_old = scale_ref->cpu().item<float>();
    const float opacity_old = opacity_ref->cpu().item<float>();
    ASSERT_GT(scale_old, 0.0f);
    ASSERT_GT(opacity_old, 0.0f);

    auto result = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value()) << std::string(result.error().user_message());

    AdamConfig cfg{.lr = 0.0f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(*splat_, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    auto scale_loss = Tensor::zeros({1}, Device::CUDA);
    auto opacity_loss = Tensor::zeros({1}, Device::CUDA);

    FastGSFusedExtraGradients extra;
    extra.scale_reg_weight = kScaleWeight;
    extra.opacity_reg_weight = kOpacityWeight;
    extra.scale_reg_loss_out = scale_loss.ptr<float>();
    extra.opacity_reg_loss_out = opacity_loss.ptr<float>();

    auto grad_out = Tensor::zeros_like(result->first.image);
    fast_rasterize_backward(result->second, grad_out, *splat_, opt, {}, {},
                            DensificationType::None, /*iteration=*/1, extra);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const float scale_new = scale_loss.cpu().item<float>();
    const float opacity_new = opacity_loss.cpu().item<float>();

    EXPECT_LT(relative_delta(scale_old, scale_new), 1e-5f)
        << "scale reg: old=" << scale_old << " fused=" << scale_new;
    EXPECT_LT(relative_delta(opacity_old, opacity_new), 1e-5f)
        << "opacity reg: old=" << opacity_old << " fused=" << opacity_new;
}

TEST_F(FusedRegLossTest, LogScaleAndEffectiveRankMatchCpuReference) {
    constexpr float kScaleWeight = 0.01f;
    constexpr float kNormalizer = 2.0f;
    constexpr float kRankWeight = 0.001f;
    constexpr float kLearningRate = 0.01f;
    constexpr float kEpsilon = 0.1f;
    constexpr double logs[3] = {0.0, -0.5, -1.5};
    auto scale_cpu = Tensor::empty({64, 3}, Device::CPU);
    for (size_t i = 0; i < 64; ++i)
        for (size_t axis = 0; axis < 3; ++axis)
            scale_cpu.ptr<float>()[i * 3 + axis] = static_cast<float>(logs[axis]);
    splat_->scaling_raw() = scale_cpu.to(Device::CUDA);

    AdamConfig cfg{.lr = kLearningRate, .beta1 = 0.9, .beta2 = 0.999, .eps = kEpsilon};
    AdamOptimizer opt(*splat_, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);
    auto fwd = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
    auto scale_loss = Tensor::zeros({1}, Device::CUDA);
    auto rank_loss = Tensor::zeros({1}, Device::CUDA);
    FastGSFusedExtraGradients extra;
    extra.scale_reg_weight = kScaleWeight;
    extra.scale_reg_log = true;
    extra.scale_reg_normalizer = kNormalizer;
    extra.scale_reg_loss_out = scale_loss.ptr<float>();
    extra.erank_reg_weight = kRankWeight;
    extra.erank_reg_loss_out = rank_loss.ptr<float>();
    auto grad_out = Tensor::zeros_like(fwd->first.image);
    fast_rasterize_backward(fwd->second, grad_out, *splat_, opt, {}, {},
                            DensificationType::None, 1, extra);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const double rank = effective_rank_reference(logs[0], logs[1], logs[2]);
    constexpr double h = 1.0e-4;
    const auto after = splat_->scaling_raw().cpu();
    const float expected_scale_loss = kScaleWeight * 0.01f / (3.0f * kNormalizer) *
                                      static_cast<float>(logs[0] + logs[1] + logs[2]);
    EXPECT_NEAR(scale_loss.cpu().item<float>(), expected_scale_loss, 1.0e-7f);
    EXPECT_NEAR(rank_loss.cpu().item<float>(), kRankWeight * static_cast<float>(rank), 1.0e-5f);
    for (int axis = 0; axis < 3; ++axis) {
        double plus[3] = {logs[0], logs[1], logs[2]};
        double minus[3] = {logs[0], logs[1], logs[2]};
        plus[axis] += h;
        minus[axis] -= h;
        const double rank_grad = (effective_rank_reference(plus[0], plus[1], plus[2]) -
                                  effective_rank_reference(minus[0], minus[1], minus[2])) /
                                 (2.0 * h);
        const double scale_grad = kScaleWeight * 0.01 / (64.0 * 3.0 * kNormalizer);
        const double grad = scale_grad + kRankWeight * rank_grad / 64.0;
        const double step = -kLearningRate * grad / (std::abs(grad) + kEpsilon);
        EXPECT_NEAR(after.ptr<float>()[axis], logs[axis] + step, 2.0e-5);
    }
}

TEST_F(FusedRegLossTest, DirectColorRegularizationMatchesCpuReference) {
    constexpr int kSplatCount = 300;
    constexpr float kWeight = 0.001f;
    constexpr float kLearningRate = 0.01f;
    constexpr float kEpsilon = 0.1f;
    constexpr float kCoefficient = 3.0f;
    constexpr float kLimit = 0.5f / 0.28209479177387814f;
    const float delta = kCoefficient - kLimit;
    auto model = make_splat(kSplatCount);
    auto sh0_cpu = model->sh0().cpu();
    for (int i = 0; i < kSplatCount; ++i)
        sh0_cpu.ptr<float>()[i * 3] = kCoefficient;
    model->sh0() = sh0_cpu.to(Device::CUDA);

    AdamConfig cfg{.lr = kLearningRate, .beta1 = 0.9, .beta2 = 0.999, .eps = kEpsilon};
    AdamOptimizer opt(*model, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);
    auto fwd = fast_rasterize_forward(*camera_, *model, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
    auto loss = Tensor::zeros({1}, Device::CUDA);
    FastGSFusedExtraGradients extra;
    extra.dc_reg_weight = kWeight;
    extra.dc_reg_loss_out = loss.ptr<float>();
    auto grad_out = Tensor::zeros_like(fwd->first.image);
    fast_rasterize_backward(fwd->second, grad_out, *model, opt, {}, {},
                            DensificationType::None, 1, extra);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const float* old = sh0_cpu.ptr<float>();
    const auto after = model->sh0().cpu();
    const float grad = 2.0f * kWeight * delta / 3.0f;
    const float expected_step = -kLearningRate * grad / (std::fabs(grad) + kEpsilon);
    EXPECT_NEAR(loss.cpu().item<float>(), kSplatCount * kWeight * delta * delta / 3.0f, 1.0e-4f);
    EXPECT_NEAR(after.ptr<float>()[0], old[0] + expected_step, 2.0e-5f);
}

TEST_F(FusedRegLossTest, HigherOrderColorRegularizationMatchesCpuReference) {
    constexpr int kSplatCount = 300;
    constexpr float kWeight = 0.1f;
    constexpr float kCoefficient = 0.25f;
    constexpr float kLearningRate = 0.01f;
    constexpr float kEpsilon = 0.1f;
    auto model = make_splat(kSplatCount, 1);
    model->shN().fill_(kCoefficient);

    AdamConfig cfg{.lr = kLearningRate, .beta1 = 0.9, .beta2 = 0.999, .eps = kEpsilon};
    AdamOptimizer opt(*model, cfg);
    opt.allocate_gradients(kSplatCount);
    auto fused = opt.prepare_fastgs_fused_adam(1001);
    ASSERT_TRUE(fused.enabled);
    ASSERT_TRUE(fused.shN.enabled);
    ASSERT_EQ(fused.shN.joint_bits, 8);
    opt.zero_grad(0);
    auto fwd = fast_rasterize_forward(*camera_, *model, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
    auto loss = Tensor::zeros({1}, Device::CUDA);
    FastGSFusedExtraGradients extra;
    extra.sh_rest_reg_weight = kWeight;
    extra.sh_rest_reg_loss_out = loss.ptr<float>();
    auto grad_out = Tensor::zeros_like(fwd->first.image);
    fast_rasterize_backward(fwd->second, grad_out, *model, opt, {}, {},
                            DensificationType::None, 1001, extra);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const float grad = 2.0f * kWeight * kCoefficient /
                       (kSplatCount * 9.0f);
    const float step = -kLearningRate * grad / (std::fabs(grad) + kEpsilon);
    EXPECT_NEAR(loss.cpu().item<float>(), kWeight * kCoefficient * kCoefficient, 1.0e-6f);
    const auto after = model->shN_canonical().cpu();
    EXPECT_NEAR(after.ptr<float>()[0], kCoefficient + step, 2.0e-5f);
}

// Alloc counter: steady fused path reuses persistent scalars (zero_ only)
// and must issue 0 driver allocs for the reg-loss path. Legacy
// forward_loss_only still does empty({num_blocks})+empty({1}) per call
// (regularization.cpp) — kept for gsplat / freeze paths; pool hits may hide
// those in alloc_counter, so we assert the fused side only.
TEST_F(FusedRegLossTest, FusedPathHasNoPerCallRegLossAllocs) {
    constexpr float kScaleWeight = 0.01f;
    constexpr float kOpacityWeight = 0.02f;
    constexpr float kDcWeight = 0.001f;
    constexpr float kShRestWeight = 0.001f;

    auto model = make_splat(64, 1);
    model->shN().fill_(0.25f);
    auto sh0_cpu = model->sh0().cpu();
    for (size_t i = 0; i < 64; ++i)
        sh0_cpu.ptr<float>()[i * 3] = 3.0f;
    model->sh0() = sh0_cpu.to(Device::CUDA);

    AdamConfig cfg{.lr = 0.0f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(*model, cfg);
    opt.allocate_gradients();
    const auto fused = opt.prepare_fastgs_fused_adam(1001);
    ASSERT_TRUE(fused.enabled);
    opt.zero_grad(0);

    // Persistent scalars (one-time alloc, outside the measured window).
    auto scale_loss = Tensor::zeros({1}, Device::CUDA);
    auto opacity_loss = Tensor::zeros({1}, Device::CUDA);
    auto dc_loss = Tensor::zeros({1}, Device::CUDA);
    auto sh_rest_loss = Tensor::zeros({1}, Device::CUDA);

    auto run_bwd = [&](int iter) {
        auto fwd = fast_rasterize_forward(*camera_, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
        auto grad_out = Tensor::zeros_like(fwd->first.image);
        scale_loss.zero_();
        opacity_loss.zero_();
        dc_loss.zero_();
        sh_rest_loss.zero_();
        FastGSFusedExtraGradients extra;
        extra.scale_reg_weight = kScaleWeight;
        extra.opacity_reg_weight = kOpacityWeight;
        extra.dc_reg_weight = kDcWeight;
        extra.sh_rest_reg_weight = kShRestWeight;
        extra.scale_reg_loss_out = scale_loss.ptr<float>();
        extra.opacity_reg_loss_out = opacity_loss.ptr<float>();
        extra.dc_reg_loss_out = dc_loss.ptr<float>();
        extra.sh_rest_reg_loss_out = sh_rest_loss.ptr<float>();
        fast_rasterize_backward(fwd->second, grad_out, *model, opt, {}, {},
                                DensificationType::None, iter, extra);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // Drop forward cache so the next step is a clean same-size run.
        fwd->second.release_forward_context();
    };

    // Warm: settle sort buffers / any first-touch caches.
    ASSERT_NO_FATAL_FAILURE(run_bwd(1001));
    ASSERT_GT(scale_loss.cpu().item<float>(), 0.0f);
    ASSERT_GT(opacity_loss.cpu().item<float>(), 0.0f);

    // Steady fused step: zero_ + fused bwd only — no empty for reg loss.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto snap = alloc_counter::snapshot();
    ASSERT_NO_FATAL_FAILURE(run_bwd(1002));
    const auto fused_delta = alloc_counter::delta_since(snap);

    EXPECT_EQ(fused_delta, 0u)
        << "fused reg-loss path must not allocate (got " << fused_delta
        << " driver allocs); legacy path does empty({num_blocks})+empty({1}) per call";

    EXPECT_GT(scale_loss.cpu().item<float>(), 0.0f);
    EXPECT_GT(opacity_loss.cpu().item<float>(), 0.0f);
    EXPECT_GT(dc_loss.cpu().item<float>(), 0.0f);
    EXPECT_GT(sh_rest_loss.cpu().item<float>(), 0.0f);
}
