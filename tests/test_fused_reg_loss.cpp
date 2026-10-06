/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "fast_raster_test_helpers.hpp"
#include "io/formats/ply.hpp"
#include "training/losses/regularization.hpp"
#include "training/optimizer/adam_optimizer.hpp"

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
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
        return Camera(R, T, /*fx=*/100.f, /*fy=*/100.f, /*cx=*/w * 0.5f, /*cy=*/h * 0.5f,
                      Tensor(), Tensor(), CameraModelType::PINHOLE, "test", "",
                      std::filesystem::path{}, w, h, 0);
    }

    std::unique_ptr<SplatData> make_splat(int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::GPU);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                p[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
                p[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
                p[i * 3 + 2] = 0.0f;
            }
            means = cpu.to(Device::GPU);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.5f, Device::GPU);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::GPU);
        // Varied raw scales so mean(exp(s)) is non-trivial.
        auto scaling_cpu = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CPU);
        float* sp = scaling_cpu.ptr<float>();
        for (int i = 0; i < n; ++i) {
            sp[i * 3 + 0] = -2.0f + 0.01f * static_cast<float>(i);
            sp[i * 3 + 1] = -1.8f - 0.005f * static_cast<float>(i % 7);
            sp[i * 3 + 2] = -2.2f + 0.003f * static_cast<float>(i % 5);
        }
        auto scaling = scaling_cpu.to(Device::GPU);
        std::vector<float> rot(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rot[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::GPU);
        auto opacity_cpu = Tensor::zeros({static_cast<size_t>(n)}, Device::CPU);
        float* op = opacity_cpu.ptr<float>();
        for (int i = 0; i < n; ++i) {
            op[i] = 1.5f + 0.02f * static_cast<float>(i % 11);
        }
        auto opacity = opacity_cpu.to(Device::GPU);
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

class FusedRegLossTest : public lfs::test::CudaBackendTest {
protected:
    void SetUp() override {
        LFS_CUDA_BACKEND_OR_RETURN();
        bg_ = Tensor::zeros({3}, Device::GPU);
        camera_ = std::make_unique<Camera>(make_camera(64, 64));
        splat_ = make_splat(64);
    }

    void TearDown() override {
        if (IsSkipped()) {
            return;
        }
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

    auto scale_loss = Tensor::zeros({1}, Device::GPU);
    auto opacity_loss = Tensor::zeros({1}, Device::GPU);

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
    auto scale_loss = Tensor::zeros({1}, Device::GPU);
    auto opacity_loss = Tensor::zeros({1}, Device::GPU);

    std::size_t backward_allocs = 0;
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
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto snap = alloc_counter::snapshot();
        fast_rasterize_backward(fwd->second, grad_out, *splat_, opt, {}, {},
                                DensificationType::None, iter, extra);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        backward_allocs = alloc_counter::delta_since(snap);
        // Drop forward cache so the next step is a clean same-size run.
        fwd->second.release_forward_context();
    };

    // Warm: settle sort buffers / any first-touch caches.
    ASSERT_NO_FATAL_FAILURE(run_bwd(1));
    ASSERT_GT(scale_loss.cpu().item<float>(), 0.0f);
    ASSERT_GT(opacity_loss.cpu().item<float>(), 0.0f);

    // Steady fused step: the fused backward issues no empty for reg loss.
    ASSERT_NO_FATAL_FAILURE(run_bwd(2));

    EXPECT_EQ(backward_allocs, 0u)
        << "fused reg-loss path must not allocate (got " << backward_allocs
        << " driver allocs); legacy path does empty({num_blocks})+empty({1}) per call";

    EXPECT_GT(scale_loss.cpu().item<float>(), 0.0f);
    EXPECT_GT(opacity_loss.cpu().item<float>(), 0.0f);
}

// Retained trained vertices exercise the clamp with real geometry and colours.
TEST_F(FusedRegLossTest, ClampedTrainedColorAllowsOnlyImageDrivenRecovery) {
    const auto path = std::filesystem::path(PROJECT_ROOT_PATH) /
                      "tests/data/clamped_colour_regression.ply.fixture";
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    auto loaded = lfs::io::load_ply(path);
    ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
    auto model = std::make_unique<SplatData>(std::move(loaded->value));
    model->set_active_sh_degree(0);
    ASSERT_EQ(model->size(), 32);

    std::vector<float> rotation = {
        0.980588226f, 0.079929263f, -0.179047602f,
        -0.0259451419f, 0.958005654f, 0.285573136f,
        0.194354266f, -0.275384239f, 0.941482841f};
    std::vector<float> translation = {-0.339415499f, -1.93373719f, 3.83564182f};
    auto R = Tensor::from_blob(rotation.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto T = Tensor::from_blob(translation.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    Camera camera(R, T, 64.f, 64.f, 32.f, 32.f,
                  Tensor(), Tensor(), CameraModelType::PINHOLE, "regression", "",
                  std::filesystem::path{}, 64, 64, 0);
    const auto original = model->sh0().cpu();
    for (size_t i = 0; i < 32; ++i)
        ASSERT_LT(0.5f + 0.28209479177387814f * original.ptr<float>()[i * 3 + 2], -1.f);

    auto blue_sum = [](const Tensor& image) {
        const auto cpu = image.cpu();
        double sum = 0.;
        for (size_t i = 2 * 64 * 64; i < 3 * 64 * 64; ++i)
            sum += cpu.ptr<float>()[i];
        return sum;
    };
    constexpr float h = 0.001f;
    auto plus = original.clone();
    auto minus = original.clone();
    for (size_t i = 0; i < 32; ++i) {
        plus.ptr<float>()[i * 3 + 2] += h;
        minus.ptr<float>()[i * 3 + 2] -= h;
    }
    double loss_plus = 0.;
    double loss_minus = 0.;
    model->sh0() = plus.to(Device::CUDA);
    {
        auto forward_plus = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(forward_plus.has_value());
        loss_plus = blue_sum(forward_plus->first.image);
    }
    model->sh0() = minus.to(Device::CUDA);
    {
        auto forward_minus = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(forward_minus.has_value());
        loss_minus = blue_sum(forward_minus->first.image);
    }
    const double finite_difference = (loss_plus - loss_minus) / (2. * h);
    RecordProperty("finite_difference", std::to_string(finite_difference));
    EXPECT_NEAR(finite_difference, 0., 1e-8);

    auto update = [&](float image_gradient) {
        model->sh0() = original.to(Device::CUDA);
        AdamConfig config{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 0.1f};
        AdamOptimizer optimizer(*model, config);
        optimizer.allocate_gradients();
        optimizer.zero_grad(0);
        auto forward = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        EXPECT_TRUE(forward.has_value());
        auto grad = Tensor::zeros({3, 64, 64}, Device::CPU);
        for (size_t i = 2 * 64 * 64; i < 3 * 64 * 64; ++i)
            grad.ptr<float>()[i] = image_gradient;
        fast_rasterize_backward(forward->second, grad.to(Device::CUDA), *model, optimizer,
                                {}, {}, DensificationType::None, 1);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto after = model->sh0().cpu();
        float largest = 0.f;
        for (size_t i = 0; i < 32; ++i) {
            const float change = after.ptr<float>()[i * 3 + 2] - original.ptr<float>()[i * 3 + 2];
            if (std::fabs(change) > std::fabs(largest))
                largest = change;
        }
        return largest;
    };
    const float darker = update(1.f);
    const float brighter = update(-1.f);
    RecordProperty("darker_change", std::to_string(darker));
    RecordProperty("brighter_change", std::to_string(brighter));
    EXPECT_NEAR(darker, 0.f, 1e-6f);
    EXPECT_GT(brighter, 1e-4f);

    // Positive colours retain the actual derivative of the forward renderer.
    auto positive = original.clone();
    for (size_t i = 0; i < 32; ++i)
        positive.ptr<float>()[i * 3 + 2] = (0.2f - 0.5f) / 0.28209479177387814f;
    plus = positive.clone();
    minus = positive.clone();
    for (size_t i = 0; i < 32; ++i) {
        plus.ptr<float>()[i * 3 + 2] += h;
        minus.ptr<float>()[i * 3 + 2] -= h;
    }
    model->sh0() = plus.to(Device::CUDA);
    {
        auto forward = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(forward.has_value());
        loss_plus = blue_sum(forward->first.image);
    }
    model->sh0() = minus.to(Device::CUDA);
    double coverage = 0.;
    {
        auto forward = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(forward.has_value());
        loss_minus = blue_sum(forward->first.image);
        const auto alpha = forward->first.alpha.cpu();
        for (size_t i = 0; i < alpha.numel(); ++i)
            coverage += alpha.ptr<float>()[i];
    }
    const double positive_difference = (loss_plus - loss_minus) / (2. * h);
    RecordProperty("positive_finite_difference", std::to_string(positive_difference));
    RecordProperty("visible_alpha_sum", std::to_string(coverage));
    ASSERT_GT(coverage, 1.);
    EXPECT_NEAR(positive_difference, 0.28209479177387814 * coverage, 0.02);

    plus = positive.clone();
    minus = positive.clone();
    plus.ptr<float>()[2] += h;
    minus.ptr<float>()[2] -= h;
    model->sh0() = plus.to(Device::CUDA);
    {
        auto fwd = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(fwd.has_value());
        loss_plus = blue_sum(fwd->first.image);
    }
    model->sh0() = minus.to(Device::CUDA);
    {
        auto fwd = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(fwd.has_value());
        loss_minus = blue_sum(fwd->first.image);
    }
    const float first_gradient = static_cast<float>((loss_plus - loss_minus) / (2. * h));
    ASSERT_GT(first_gradient, 0.f);
    model->sh0() = positive.to(Device::CUDA);
    {
        AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 0.1f};
        AdamOptimizer opt(*model, cfg);
        opt.allocate_gradients();
        opt.zero_grad(0);
        auto fwd = fast_rasterize_forward(camera, *model, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(fwd.has_value());
        auto grad = Tensor::zeros({3, 64, 64}, Device::CPU);
        for (size_t i = 2 * 64 * 64; i < 3 * 64 * 64; ++i)
            grad.ptr<float>()[i] = 1.f;
        fast_rasterize_backward(fwd->second, grad.to(Device::CUDA), *model, opt,
                                {}, {}, DensificationType::None, 1);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto after = model->sh0().cpu();
        const float expected_step = -0.01f * first_gradient / (std::fabs(first_gradient) + 0.1f);
        const float actual_step = after.ptr<float>()[2] - positive.ptr<float>()[2];
        RecordProperty("positive_first_gradient", std::to_string(first_gradient));
        RecordProperty("positive_expected_step", std::to_string(expected_step));
        RecordProperty("positive_actual_step", std::to_string(actual_step));
        EXPECT_NEAR(actual_step, expected_step, 2e-5f);
    }
}
