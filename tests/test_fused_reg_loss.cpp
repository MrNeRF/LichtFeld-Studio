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

    std::unique_ptr<SplatData> make_splat(int n) {
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
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA);
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
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    void cleanup_arena() {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    float relative_delta(float a, float b) {
        const float denom = std::max(std::max(std::fabs(a), std::fabs(b)), 1e-12f);
        return std::fabs(a - b) / denom;
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

// Alloc counter: steady fused path reuses persistent scalars (zero_ only)
// and must issue 0 driver allocs for the reg-loss path. Legacy
// forward_loss_only still does empty({num_blocks})+empty({1}) per call
// (regularization.cpp) — kept for gsplat / freeze paths; pool hits may hide
// those in alloc_counter, so we assert the fused side only.
TEST_F(FusedRegLossTest, FusedPathHasNoPerCallRegLossAllocs) {
    constexpr float kScaleWeight = 0.01f;
    constexpr float kOpacityWeight = 0.02f;

    AdamConfig cfg{.lr = 0.0f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(*splat_, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    // Persistent scalars (one-time alloc, outside the measured window).
    auto scale_loss = Tensor::zeros({1}, Device::CUDA);
    auto opacity_loss = Tensor::zeros({1}, Device::CUDA);

    auto run_bwd = [&](int iter) {
        auto fwd = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
        auto grad_out = Tensor::zeros_like(fwd->first.image);
        scale_loss.zero_();
        opacity_loss.zero_();
        FastGSFusedExtraGradients extra;
        extra.scale_reg_weight = kScaleWeight;
        extra.opacity_reg_weight = kOpacityWeight;
        extra.scale_reg_loss_out = scale_loss.ptr<float>();
        extra.opacity_reg_loss_out = opacity_loss.ptr<float>();
        fast_rasterize_backward(fwd->second, grad_out, *splat_, opt, {}, {},
                                DensificationType::None, iter, extra);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // Drop forward cache so the next step is a clean same-size run.
        fwd->second.release_forward_context();
    };

    // Warm: settle sort buffers / any first-touch caches.
    ASSERT_NO_FATAL_FAILURE(run_bwd(1));
    ASSERT_GT(scale_loss.cpu().item<float>(), 0.0f);
    ASSERT_GT(opacity_loss.cpu().item<float>(), 0.0f);

    // Steady fused step: zero_ + fused bwd only — no empty for reg loss.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto snap = alloc_counter::snapshot();
    ASSERT_NO_FATAL_FAILURE(run_bwd(2));
    const auto fused_delta = alloc_counter::delta_since(snap);

    EXPECT_EQ(fused_delta, 0u)
        << "fused reg-loss path must not allocate (got " << fused_delta
        << " driver allocs); legacy path does empty({num_blocks})+empty({1}) per call";

    EXPECT_GT(scale_loss.cpu().item<float>(), 0.0f);
    EXPECT_GT(opacity_loss.cpu().item<float>(), 0.0f);
}
