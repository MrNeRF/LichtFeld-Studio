/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/popspa.cuh"
#include "training/losses/regularization.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <format>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;

    double sigmoid_derivative_reference(const double x) {
        return 1.0 / (2.0 + 2.0 * std::cosh(x));
    }

    std::unique_ptr<SplatData> make_splat(const float logit) {
        return std::make_unique<SplatData>(
            0, Tensor::zeros({1, 3}, Device::CUDA),
            Tensor::full({1, 1, 3}, 0.5f, Device::CUDA),
            Tensor::zeros({1, 0, 3}, Device::CUDA),
            Tensor::full({1, 3}, -1.7f, Device::CUDA),
            Tensor::from_vector({1.f, 0.f, 0.f, 0.f}, {1, 4}, Device::CUDA),
            Tensor::full({1}, logit, Device::CUDA), 1.f);
    }

    Camera make_camera() {
        return Camera(
            Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA),
            Tensor::from_vector({0.f, 0.f, 3.f}, {3}, Device::CUDA),
            32.f, 32.f, 16.f, 16.f, Tensor(), Tensor(),
            lfs::core::CameraModelType::PINHOLE, "opacity-gradient", "", std::filesystem::path{}, 32, 32, 0);
    }

    class OpacityGradientTest : public ::testing::Test {
        void TearDown() override {
            release_gsplat_rasterizer_thread_local_caches();
            GlobalArenaManager::instance().get_arena().full_reset();
        }
    };

    TEST_F(OpacityGradientTest, RegularizationMatchesDoubleReferenceAtBothTails) {
        const std::vector<float> logits = {-80.f, -40.f, -20.f, -17.f, -10.f, -1.f, 0.f,
                                           1.f, 10.f, 12.f, 16.f, 17.f, 20.f, 40.f, 80.f};
        auto raw = Tensor::from_vector(logits, {logits.size()}, Device::CUDA);
        auto grad = Tensor::zeros_like(raw);
        auto result = losses::OpacityRegularization::forward(
            raw, grad, {.weight = static_cast<float>(logits.size())});
        ASSERT_TRUE(result.has_value()) << result.error();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto host = grad.cpu();
        double max_relative_error = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            SCOPED_TRACE(logits[i]);
            const double expected = sigmoid_derivative_reference(logits[i]);
            const double actual = host.ptr<float>()[i];
            max_relative_error = std::max(max_relative_error, std::abs(actual / expected - 1.0));
            EXPECT_GT(actual, 0.0);
            EXPECT_NEAR(actual / expected, 1.0, 5e-6);
        }
        RecordProperty("max_relative_error", std::format("{:.9e}", max_relative_error));
    }

    TEST_F(OpacityGradientTest, PopspaPenaltyPreservesSaturatedOpacityGradient) {
        constexpr float logit = 17.f, rho = 2.f;
        auto raw = Tensor::full({1}, logit, Device::CUDA);
        auto proxy = Tensor::zeros_like(raw);
        auto dual = Tensor::zeros_like(raw);
        auto active = Tensor::from_vector(std::vector<bool>{true}, {1}, Device::CUDA);
        auto frozen = Tensor::from_vector(std::vector<bool>{false}, {1}, Device::CUDA);
        auto grad = Tensor::zeros_like(raw);
        auto loss = Tensor::zeros({1}, Device::CUDA);
        // A zero proxy asks POPSpa to remove this saturated Gaussian.
        kernels::launch_popspa_penalty(raw.ptr<float>(), proxy.ptr<float>(), dual.ptr<float>(),
                                       active.ptr<bool>(), frozen.ptr<bool>(), grad.ptr<float>(),
                                       loss.ptr<float>(), 1, rho, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const double opacity = 1.0 / (1.0 + std::exp(-static_cast<double>(logit)));
        const double expected = rho * opacity * sigmoid_derivative_reference(logit);
        const float actual = grad.cpu().item<float>();
        RecordProperty("penalty_gradient", std::format("{:.9e}", actual));
        EXPECT_GT(actual, 0.f);
        EXPECT_NEAR(actual / expected, 1.0, 5e-6);
        EXPECT_FLOAT_EQ(loss.cpu().item<float>(), 0.5f * rho);
    }

    TEST_F(OpacityGradientTest, FusedRegularizationMovesSaturatedOpacity) {
        auto camera = make_camera();
        auto model = make_splat(17.f);
        auto bg = Tensor::zeros({3}, Device::CUDA);
        AdamOptimizer optimizer(*model, {.lr = 0.f});
        optimizer.set_param_lr(lfs::training::ParamType::Opacity, 0.01);
        optimizer.allocate_gradients();
        optimizer.zero_grad(0);
        auto fwd = fast_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
        FastGSFusedExtraGradients extra;
        extra.opacity_reg_weight = 0.02f;
        fast_rasterize_backward(fwd->second, Tensor::zeros_like(fwd->first.image),
                                *model, optimizer, {}, {}, DensificationType::None, 1, extra);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const float after = model->opacity_raw().cpu().item<float>();
        RecordProperty("raw_opacity_after", std::to_string(after));
        EXPECT_NEAR(after, 16.99f, 2e-5f);
    }

    class OpacityRasterizerTest : public OpacityGradientTest,
                                  public ::testing::WithParamInterface<bool> {};

    TEST_P(OpacityRasterizerTest, SaturatedPhotometricTrainingRecovers) {
        const bool gut = GetParam();
        auto camera = make_camera();
        auto model = make_splat(17.f);
        auto bg = Tensor::zeros({3}, Device::CUDA);
        AdamOptimizer optimizer(*model, {.lr = 0.f});
        optimizer.set_param_lr(lfs::training::ParamType::Opacity, 0.05);
        optimizer.allocate_gradients();
        float initial_loss = 0.f;
        float final_loss = 0.f;
        // A fixed Gaussian against a black target isolates opacity optimization.
        // Its image is positive, so mean(image) is exactly the L1 image loss.
        for (int step = 1; step <= 400; ++step) {
            optimizer.zero_grad(step);
            if (gut) {
                auto fwd = gsplat_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, 1.f,
                                                    false, GsplatRenderMode::RGB, true);
                ASSERT_TRUE(fwd.has_value()) << fwd.error();
                final_loss = fwd->first.image.mean().cpu().item<float>();
                auto grad = Tensor::full(fwd->first.image.shape(),
                                         1.f / static_cast<float>(fwd->first.image.numel()), Device::CUDA);
                gsplat_rasterize_backward(fwd->second, grad,
                                          Tensor::zeros_like(fwd->first.alpha), *model, optimizer);
                optimizer.step(step);
            } else {
                auto fwd = fast_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, false);
                ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
                final_loss = fwd->first.image.mean().cpu().item<float>();
                auto grad = Tensor::full(fwd->first.image.shape(),
                                         1.f / static_cast<float>(fwd->first.image.numel()), Device::CUDA);
                fast_rasterize_backward(fwd->second, grad, *model, optimizer, {}, {},
                                        DensificationType::None, step);
            }
            if (step == 1)
                initial_loss = final_loss;
            ASSERT_TRUE(std::isfinite(final_loss));
        }
        const auto final_render = gut
                                      ? gsplat_rasterize(camera, *model, bg, 1.f, false, GsplatRenderMode::RGB, true)
                                      : fast_rasterize(camera, *model, bg);
        final_loss = final_render.image.mean().cpu().item<float>();
        const float after = model->opacity_raw().cpu().item<float>();
        RecordProperty("initial_loss", std::to_string(initial_loss));
        RecordProperty("final_loss", std::to_string(final_loss));
        RecordProperty("raw_opacity_after", std::to_string(after));
        EXPECT_LT(after, 0.f);
        EXPECT_LT(final_loss, initial_loss * 0.1f);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, OpacityRasterizerTest, ::testing::Bool(),
                             [](const auto& info) { return info.param ? "Gsplat" : "FastGS"; });
} // namespace
