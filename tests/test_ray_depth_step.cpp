/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The ray-depth coordinate rests on two properties of both training rasterizers: an image does not change when a
// splat is scaled about the camera centre, and the gradient that reaches a splat through the image, alpha and
// normal maps satisfies (x - c).g_x + sum(g_log_scale) = 0 per splat. Only a rendered-depth gradient breaks it.

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/ray_depth_step.cuh"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <vector>

namespace {

    using namespace lfs::core;
    using namespace lfs::training;

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr float kFocal = 60.0f;
    constexpr size_t kSplats = 12;
    // World-to-camera translation with R = I: the camera centre is -T.
    constexpr float kT[3] = {0.3f, -0.2f, 1.0f};

    bool cuda_available() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    Camera make_camera(const float* t, const std::string& name = "ray_depth") {
        auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CPU);
        auto T = Tensor::from_vector({t[0], t[1], t[2]}, {3}, Device::CPU);
        return Camera(R, T, kFocal, kFocal, kWidth / 2.0f, kHeight / 2.0f, Tensor(), Tensor(),
                      lfs::core::CameraModelType::PINHOLE, name, "", std::filesystem::path{}, kWidth, kHeight, 0);
    }

    struct Scene {
        std::vector<float> means, sh0, shN, scales, rotations, opacities;
    };

    // Splats from 2 to 25 units in front of the camera, sorted by depth, of a few pixels each (or larger, with
    // relative_size), with view-dependent colour. One sits across the left image border.
    Scene make_scene(const float relative_size = 0.05f) {
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        Scene s;
        for (size_t i = 0; i < kSplats; ++i) {
            const float depth = 2.0f + 23.0f * static_cast<float>(i) / static_cast<float>(kSplats - 1);
            const float lateral_x = i == 3 ? -0.52f : 0.35f * u(rng);
            const float lateral_y = 0.3f * u(rng);
            s.means.insert(s.means.end(), {lateral_x * depth - kT[0], lateral_y * depth - kT[1], depth - kT[2]});
            s.sh0.insert(s.sh0.end(), {u(rng), u(rng), u(rng)});
            for (int k = 0; k < 9; ++k)
                s.shN.push_back(0.4f * u(rng));
            const float log_size = std::log(relative_size * depth);
            s.scales.insert(s.scales.end(), {log_size + 0.4f * u(rng), log_size + 0.4f * u(rng), log_size - 0.5f + 0.3f * u(rng)});
            s.rotations.insert(s.rotations.end(), {1.0f, 0.5f * u(rng), 0.5f * u(rng), 0.5f * u(rng)});
            s.opacities.push_back(1.0f + u(rng));
        }
        return s;
    }

    SplatData to_splats(const Scene& s) {
        const size_t n = s.opacities.size();
        return SplatData(1,
                         Tensor::from_vector(s.means, {n, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(s.sh0, {n, size_t{1}, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(s.shN, {n, size_t{3}, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(s.scales, {n, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(s.rotations, {n, size_t{4}}, Device::CUDA),
                         Tensor::from_vector(s.opacities, {n, size_t{1}}, Device::CUDA),
                         1.0f);
    }

    std::vector<float> render(const Scene& scene, const bool gut) {
        auto model = to_splats(scene);
        auto camera = make_camera(kT);
        auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
        const auto output = gut ? gsplat_rasterize(camera, model, bg, 1.0f, false, GsplatRenderMode::RGB, true)
                                : fast_rasterize(camera, model, bg);
        return output.image.cpu().contiguous().to_vector();
    }

    // Low-frequency loss weights [3, H, W]: L = <image, w>.
    std::vector<float> weights() {
        std::vector<float> w(3 * kWidth * kHeight);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < kHeight; ++y)
                for (int x = 0; x < kWidth; ++x)
                    w[(c * kHeight + y) * kWidth + x] =
                        static_cast<float>(std::sin(2.0 * std::numbers::pi * (x + 0.5) / kWidth + 1.3 * c) *
                                           std::cos(2.0 * std::numbers::pi * (y + 0.5) / kHeight - 0.7 * c));
        return w;
    }

    struct Gradients {
        std::vector<float> means, scales;
    };

    // Loss terms on top of L = <image, w> / numel.
    enum class Extra {
        None,
        DepthNormalConsistency,                 // with its gradients into the normal, alpha and depth maps
        DepthNormalConsistencyWithoutDepthPath, // the same with the rendered-depth gradient zeroed
    };

    // Adds the depth-normal consistency gradients of one view to grad_normal, grad_alpha and grad_depth.
    void add_depth_normal_consistency(const RenderOutput& output, const float* camera_rays, Tensor& grad_normal,
                                      Tensor& grad_alpha, Tensor& grad_depth) {
        const auto hw = TensorShape({size_t{kHeight}, size_t{kWidth}});
        auto normal = output.normal.contiguous();
        auto depth = output.depth.contiguous().reshape(hw);
        auto alpha = output.alpha.contiguous().reshape(hw);
        grad_normal = Tensor::zeros(normal.shape(), Device::CUDA);
        grad_alpha = Tensor::zeros(hw, Device::CUDA);
        grad_depth = Tensor::zeros(hw, Device::CUDA);
        auto loss = Tensor::zeros({1}, Device::CUDA);
        auto partials = Tensor::zeros(
            {kernels::normal_consistency_partial_count(static_cast<size_t>(kWidth) * kHeight)}, Device::CUDA);
        kernels::launch_normal_consistency_loss(normal.ptr<float>(), depth.ptr<float>(), alpha.ptr<float>(),
                                                grad_normal.ptr<float>(), grad_depth.ptr<float>(),
                                                grad_alpha.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(),
                                                kWidth, kHeight, kFocal, kFocal, kWidth / 2.0f, kHeight / 2.0f, 1.0f,
                                                nullptr, nullptr, camera_rays);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_GT(grad_depth.abs().sum().cpu().item<float>(), 0.0f) << "the consistency term must reach the depth map";
    }

    Gradients photometric_gradients(const Scene& scene, const bool gut, const Extra extra = Extra::None) {
        auto model = to_splats(scene);
        auto camera = make_camera(kT);
        auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
        AdamOptimizer optimizer(model, {.lr = 0.0f});
        optimizer.allocate_gradients();
        optimizer.zero_grad(0);
        auto grad_image = Tensor::from_vector(weights(), {size_t{3}, size_t{kHeight}, size_t{kWidth}}, Device::CUDA) *
                          (1.0f / static_cast<float>(3 * kWidth * kHeight));
        const bool consistency = extra != Extra::None;
        Tensor grad_normal, grad_alpha, grad_depth;
        const auto add_consistency = [&](const RenderOutput& output, const float* camera_rays) {
            add_depth_normal_consistency(output, camera_rays, grad_normal, grad_alpha, grad_depth);
            if (extra == Extra::DepthNormalConsistencyWithoutDepthPath)
                grad_depth.zero_();
        };
        if (gut) {
            auto forward = gsplat_rasterize_forward(camera, model, bg, 0, 0, 0, 0, 1.0f, false,
                                                    consistency ? GsplatRenderMode::RGB_D_N : GsplatRenderMode::RGB,
                                                    true);
            EXPECT_TRUE(forward.has_value());
            if (!forward)
                return {};
            if (consistency)
                add_consistency(forward->first, forward->second.camera_rays.ptr<float>());
            else
                grad_alpha = Tensor::zeros_like(forward->first.alpha);
            gsplat_rasterize_backward(forward->second, grad_image, grad_alpha, model, optimizer, {}, {}, {},
                                      grad_depth, grad_normal);
        } else {
            auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, consistency, true);
            EXPECT_TRUE(forward.has_value());
            if (!forward)
                return {};
            if (consistency)
                add_consistency(forward->first, nullptr);
            fast_rasterize_backward(forward->second, grad_image, model, optimizer, grad_alpha, {},
                                    DensificationType::None, 0, {}, grad_depth, grad_normal, true);
        }
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return {optimizer.get_grad(ParamType::Means).cpu().to_vector(),
                optimizer.get_grad(ParamType::Scaling).cpu().to_vector()};
    }

    // Largest per-splat |(x - c).g_x + sum(g_log_scale)| against the size of the two terms; counts the splats
    // the loss reaches. against_full_gradient: the same residual against |x - c| |g_x|.
    double worst_relative_residual(const Scene& scene, const Gradients& g, int& checked,
                                   double* against_full_gradient = nullptr) {
        const float centre[3] = {-kT[0], -kT[1], -kT[2]};
        double worst = 0.0, worst_full = 0.0;
        checked = 0;
        for (size_t i = 0; i < kSplats; ++i) {
            double radial = 0.0, scale = 0.0, distance2 = 0.0, gradient2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double offset = static_cast<double>(scene.means[3 * i + k]) - centre[k];
                radial += offset * g.means[3 * i + k];
                scale += g.scales[3 * i + k];
                distance2 += offset * offset;
                gradient2 += static_cast<double>(g.means[3 * i + k]) * g.means[3 * i + k];
            }
            const double magnitude = std::abs(radial) + std::abs(scale);
            if (magnitude < 1e-9)
                continue;
            ++checked;
            worst = std::max(worst, std::abs(radial + scale) / magnitude);
            worst_full = std::max(worst_full, std::abs(radial + scale) / std::sqrt(distance2 * gradient2));
        }
        if (against_full_gradient != nullptr)
            *against_full_gradient = worst_full;
        return worst;
    }

    class RayDepthRasterizerTest : public ::testing::TestWithParam<bool> {
    protected:
        void SetUp() override {
            if (!cuda_available())
                GTEST_SKIP() << "CUDA not available";
        }
        void TearDown() override {
            release_gsplat_rasterizer_thread_local_caches();
            GlobalArenaManager::instance().get_arena().full_reset();
        }
    };

    // The five nearest splats move in, the five farthest out and the two between them stay, so the
    // depth order stays.
    TEST_P(RayDepthRasterizerTest, MoveAboutTheCameraLeavesTheImageUnchanged) {
        const bool gut = GetParam();
        const auto scene = make_scene();
        const auto before = render(scene, gut);
        auto moved = scene;
        const float centre[3] = {-kT[0], -kT[1], -kT[2]};
        for (size_t i = 0; i < kSplats; ++i) {
            if (i == kSplats / 2 - 1 || i == kSplats / 2)
                continue;
            ray_depth::move(&moved.means[3 * i], &moved.scales[3 * i], centre, i < kSplats / 2 ? -0.5f : 0.5f);
        }
        const auto after = render(moved, gut);
        ASSERT_EQ(before.size(), after.size());
        ASSERT_FALSE(before.empty());
        float max_diff = 0.0f;
        double mean_abs = 0.0;
        for (size_t k = 0; k < before.size(); ++k) {
            max_diff = std::max(max_diff, std::abs(before[k] - after[k]));
            mean_abs += std::abs(before[k] - 0.3f);
        }
        RecordProperty("max_pixel_difference", std::format("{:.3e}", max_diff));
        EXPECT_GT(mean_abs / static_cast<double>(before.size()), 0.01) << "the splats must cover the image";
        EXPECT_LT(max_diff, 2e-5f);
    }

    TEST_P(RayDepthRasterizerTest, PhotometricGradientIsInvariantToScalingAboutTheCamera) {
        const bool gut = GetParam();
        const auto scene = make_scene();
        const auto g = photometric_gradients(scene, gut);
        ASSERT_EQ(g.means.size(), 3 * kSplats);
        ASSERT_EQ(g.scales.size(), 3 * kSplats);
        int checked = 0;
        double against_full_gradient = 0.0;
        const double worst = worst_relative_residual(scene, g, checked, &against_full_gradient);
        RecordProperty("worst_relative_residual", std::format("{:.3e}", worst));
        RecordProperty("worst_residual_against_distance_times_gradient", std::format("{:.3e}", against_full_gradient));
        std::printf("worst relative residual: %.3e (against |x - c| |g_x|: %.3e) over %d splats\n", worst,
                    against_full_gradient, checked);
        EXPECT_LT(worst, 2e-3);
        EXPECT_GE(checked, static_cast<int>(kSplats) - 2) << "the loss must reach the splats";
    }

    // Depth-normal consistency reaches a splat through the normal, alpha and depth maps. Only the depth path
    // breaks the identity; with the rendered-depth gradient zeroed it holds again.
    TEST_P(RayDepthRasterizerTest, OnlyTheRenderedDepthPathBreaksTheInvariance) {
        const bool gut = GetParam();
        const auto scene = make_scene(0.3f);
        int with_depth_checked = 0, without_depth_checked = 0;
        const double with_depth = worst_relative_residual(
            scene, photometric_gradients(scene, gut, Extra::DepthNormalConsistency), with_depth_checked);
        const double without_depth = worst_relative_residual(
            scene, photometric_gradients(scene, gut, Extra::DepthNormalConsistencyWithoutDepthPath),
            without_depth_checked);
        RecordProperty("worst_relative_residual_with_depth_path", std::format("{:.3e}", with_depth));
        RecordProperty("worst_relative_residual_without_depth_path", std::format("{:.3e}", without_depth));
        std::printf("worst relative residual with / without the depth path: %.3e / %.3e over %d / %d splats\n",
                    with_depth, without_depth, with_depth_checked, without_depth_checked);
        EXPECT_GT(with_depth, 1e-2);
        EXPECT_LT(without_depth, 2e-3);
        EXPECT_GE(without_depth_checked, static_cast<int>(kSplats) - 2) << "the loss must reach the splats";
    }

    // One splat at distance kTargetDistance on the first camera's ray renders the targets of three cameras a
    // hundredth of that distance apart. The trained splat starts at half the distance and half the size, which the
    // first camera cannot tell apart. Returns the distance from the first camera at the checkpoints.
    constexpr float kTargetDistance = 10.0f;
    constexpr float kBaseline = 0.1f;
    constexpr int kSteps = 300;

    std::vector<float> train_far_splat(const bool gut, const float ray_depth_lr) {
        const float direction[3] = {0.05f / std::sqrt(1.0034f), 0.03f / std::sqrt(1.0034f), 1.0f / std::sqrt(1.0034f)};
        const auto splat_at = [&](const float distance, const float log_size) {
            Scene s;
            s.means = {distance * direction[0], distance * direction[1], distance * direction[2]};
            s.sh0 = {0.8f, -0.3f, 0.2f};
            s.shN.assign(9, 0.0f);
            s.scales = {log_size, log_size, log_size};
            s.rotations = {1.0f, 0.0f, 0.0f, 0.0f};
            s.opacities = {1.5f};
            return s;
        };
        // Camera centres (0, 0, 0), (-B, 0, 0), (B, 0.5 B, 0); R = I, so T = -centre.
        const float translations[3][3] = {{0.0f, 0.0f, 0.0f}, {kBaseline, 0.0f, 0.0f}, {-kBaseline, -0.5f * kBaseline, 0.0f}};
        std::vector<Camera> cameras;
        for (int i = 0; i < 3; ++i)
            cameras.push_back(make_camera(translations[i], std::format("ray_depth_{}", i)));
        auto bg = Tensor::zeros({3}, Device::CUDA);

        const float target_log_size = std::log(0.05f * kTargetDistance);
        auto target_model = to_splats(splat_at(kTargetDistance, target_log_size));
        std::vector<Tensor> targets;
        for (auto& camera : cameras)
            targets.push_back((gut ? gsplat_rasterize(camera, target_model, bg, 1.0f, false, GsplatRenderMode::RGB, true)
                                   : fast_rasterize(camera, target_model, bg))
                                  .image.clone());

        auto model = to_splats(splat_at(0.5f * kTargetDistance, target_log_size - std::log(2.0f)));
        AdamOptimizer optimizer(model, {.lr = 0.0f});
        optimizer.set_param_lr(ParamType::Means, 1e-3);
        optimizer.set_param_lr(ParamType::Scaling, 5e-3);
        optimizer.allocate_gradients();
        if (ray_depth_lr > 0.0f) {
            auto state = Tensor::full({size_t{1}, size_t{ray_depth::kStateStride}}, ray_depth::kUnobserved, Device::CUDA);
            optimizer.set_ray_depth(state, ray_depth_lr);
        }
        std::vector<float> distances;
        for (int step = 1; step <= kSteps; ++step) {
            auto& camera = cameras[step % 3];
            const auto& target = targets[step % 3];
            optimizer.zero_grad(step);
            const auto l1_gradient = [&](const Tensor& image) {
                return (image - target).sign() * (1.0f / static_cast<float>(image.numel()));
            };
            if (gut) {
                auto forward = gsplat_rasterize_forward(camera, model, bg, 0, 0, 0, 0, 1.0f, false,
                                                        GsplatRenderMode::RGB, true);
                EXPECT_TRUE(forward.has_value());
                if (!forward)
                    return {};
                gsplat_rasterize_backward(forward->second, l1_gradient(forward->first.image),
                                          Tensor::zeros_like(forward->first.alpha), model, optimizer);
                optimizer.step(step);
            } else {
                auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false);
                EXPECT_TRUE(forward.has_value());
                if (!forward)
                    return {};
                fast_rasterize_backward(forward->second, l1_gradient(forward->first.image), model, optimizer, {}, {},
                                        DensificationType::None, step);
            }
            if (step % (kSteps / 5) == 0) {
                const auto x = model.means().cpu().to_vector();
                distances.push_back(std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
            }
        }
        return distances;
    }

    TEST_P(RayDepthRasterizerTest, FarSplatSettlesAtTheDepthTheViewsAgreeOn) {
        const bool gut = GetParam();
        const auto with = train_far_splat(gut, 0.01f);
        const auto without = train_far_splat(gut, 0.0f);
        ASSERT_EQ(with.size(), 5u);
        ASSERT_EQ(without.size(), 5u);
        std::string curves;
        for (size_t i = 0; i < with.size(); ++i)
            curves += std::format("{}{}: {:.3f} / {:.3f}", i ? ", " : "", (i + 1) * kSteps / 5, with[i], without[i]);
        RecordProperty("distance_with_without", curves);
        std::printf("distance at steps (with / without): %s\n", curves.c_str());
        EXPECT_NEAR(with.back(), kTargetDistance, 0.1f * kTargetDistance);
        EXPECT_LT(std::abs(without.back() - 0.5f * kTargetDistance), 0.05f * kTargetDistance);
    }

    // The move applies about the anchor; seen from the camera an outward move may not leave the first half of the
    // rasterizer's depth range, an inward one always applies, and the view enters the statistics either way.
    TEST(RayDepthStepMath, OutwardMoveStopsAtHalfTheFarLimit) {
        constexpr float kFarLimit = 100.0f;
        const float camera[3] = {1.0f, 0.0f, 0.0f};
        float row[ray_depth::kStateStride] = {0.0f, 0.0f, 0.0f, 0.01f, 0.0f, 1.0f};
        const auto commit = [&](const float distance, const float factor) {
            float mean[3] = {0.0f, 0.0f, distance};
            float log_scale[3] = {0.0f, -1.0f, -2.0f};
            ray_depth::commit(row, mean, log_scale, camera, std::log(factor), true, kFarLimit);
            return std::pair{mean[2], log_scale[0]};
        };
        const auto [inside, inside_scale] = commit(40.0f, 1.2f);
        EXPECT_NEAR(inside, 48.0f, 1e-4f);
        EXPECT_NEAR(inside_scale, std::log(1.2f), 1e-6f);
        const float anchor_before = row[0];
        const auto [beyond, beyond_scale] = commit(40.0f, 1.3f);
        EXPECT_EQ(beyond, 40.0f);
        EXPECT_EQ(beyond_scale, 0.0f);
        EXPECT_GT(row[0], anchor_before) << "a suppressed move still enters the running anchor";
        const auto [inward, inward_scale] = commit(80.0f, 0.5f);
        EXPECT_NEAR(inward, 40.0f, 1e-4f);
        EXPECT_NEAR(inward_scale, std::log(0.5f), 1e-6f);
    }

    // An outward proposal needs observers whose baseline spread resolves the splat (J = B2 / vperp at least
    // kMinOutwardResolution); an inward one applies either way, and the moments and the spread update as usual.
    TEST(RayDepthStepMath, OutwardMoveNeedsObserversThatResolveTheDepth) {
        const float camera[3] = {0.0f, 0.0f, 0.0f};
        const float x[3] = {0.0f, 0.0f, 10.0f};
        const float log_scale[3] = {0.0f, 0.0f, 0.0f}; // vperp = 1 across the viewing ray
        const float quat[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        const auto propose = [&](const float b2, const float gx) {
            // Anchor one unit beside the camera: b = c - a = (-1, 0, 0), g_t = -gx, bperp2 = 1.
            float row[ray_depth::kStateStride] = {1.0f, 0.0f, 0.0f, b2, 0.0f, 1.0f};
            const float g_x[3] = {gx, 0.0f, 0.0f};
            const auto p = ray_depth::propose(row, x, log_scale, quat, camera, g_x, 0.01f, 1.0f);
            EXPECT_NEAR(row[4], -0.1f * gx, 1e-7f) << "the first moment updates either way";
            EXPECT_NEAR(row[3], b2 + (1.0f - b2) * ray_depth::kObserverRate, 1e-7f) << "the spread updates either way";
            EXPECT_EQ(p.bperp2, 1.0f);
            return p.dt;
        };
        const float below = 0.1f * ray_depth::kMinOutwardResolution;
        const float above = 10.0f * ray_depth::kMinOutwardResolution;
        EXPECT_EQ(propose(below, 1.0f), 0.0f) << "outward, observers cannot resolve it";
        EXPECT_GT(propose(above, 1.0f), 0.0f) << "outward, observers resolve it";
        EXPECT_LT(propose(below, -1.0f), 0.0f) << "inward, unresolved";
        EXPECT_LT(propose(above, -1.0f), 0.0f) << "inward, resolved";
    }

    INSTANTIATE_TEST_SUITE_P(Rasterizers, RayDepthRasterizerTest, ::testing::Values(false, true),
                             [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Gut" : "FastGS"; });

} // namespace
