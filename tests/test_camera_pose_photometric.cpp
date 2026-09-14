/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/camera.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "training/camera_pose/bounded_pose_optimizer.hpp"
#include "training/camera_pose/fastgs_pose_evaluator.hpp"
#include "training/camera_pose/se3.hpp"
#include "training/camera_pose/trainer_pose_integration.hpp"
#include "training/checkpoint.hpp"
#include "training/losses/photometric_loss.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/strategies/mcmc.hpp"
#include "visualizer/scene/camera_pose_view.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;
    using namespace lfs::training::camera_pose;
    constexpr int WIDTH = 64;
    constexpr int HEIGHT = 48;
    constexpr int PIXELS = 3 * WIDTH * HEIGHT;
    constexpr float EPSILON = 1.0e-3f;
    using Vector6 = std::array<double, 6>;
    using Matrix6 = std::array<Vector6, 6>;

    std::vector<float> values(const Tensor& tensor) {
        const auto cpu = tensor.to(Device::CPU).contiguous();
        return {cpu.ptr<float>(), cpu.ptr<float>() + cpu.numel()};
    }

    double squared_loss(const std::vector<float>& image, const std::vector<float>& target) {
        double loss = 0.0;
        for (size_t i = 0; i < image.size(); ++i) {
            const double residual = static_cast<double>(image[i]) - target[i];
            loss += residual * residual;
        }
        return loss / static_cast<double>(image.size());
    }

    double scalar_product(const std::vector<float>& a, const std::vector<float>& b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    }

    Matrix4 offset(const Matrix4& pose, int axis, float amount) {
        Twist increment{};
        increment[axis] = amount;
        return apply_left_increment(increment, pose);
    }

    // Test-only linear solve. The recovery driver uses a numerical image
    // Jacobian for conditioning, but the actual descent gradient comes from
    // CUDA backward. No target pose enters this solve or the update rule.
    Vector6 solve(Matrix6 matrix, Vector6 rhs) {
        for (int col = 0; col < 6; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 6; ++row)
                if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col]))
                    pivot = row;
            std::swap(matrix[col], matrix[pivot]);
            std::swap(rhs[col], rhs[pivot]);
            if (std::abs(matrix[col][col]) < 1.0e-14)
                throw std::runtime_error("Singular photometric test system");
            const double scale = matrix[col][col];
            for (int k = col; k < 6; ++k)
                matrix[col][k] /= scale;
            rhs[col] /= scale;
            for (int row = 0; row < 6; ++row) {
                if (row == col)
                    continue;
                const double multiplier = matrix[row][col];
                for (int k = col; k < 6; ++k)
                    matrix[row][k] -= multiplier * matrix[col][k];
                rhs[row] -= multiplier * rhs[col];
            }
        }
        return rhs;
    }

    double center_error(const Matrix4& a, const Matrix4& b) {
        double error = 0.0;
        for (int j = 0; j < 3; ++j) {
            double difference = 0.0;
            for (int i = 0; i < 3; ++i)
                difference += -a[4 * i + j] * a[4 * i + 3] + b[4 * i + j] * b[4 * i + 3];
            error += difference * difference;
        }
        return std::sqrt(error);
    }

    double rotation_error(const Matrix4& a, const Matrix4& b) {
        // atan2 form is well conditioned near identity, unlike float acos.
        std::array<double, 9> relative{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    relative[3 * i + j] += static_cast<double>(a[4 * i + k]) * b[4 * j + k];
        const double x = relative[7] - relative[5];
        const double y = relative[2] - relative[6];
        const double z = relative[3] - relative[1];
        return std::atan2(0.5 * std::sqrt(x * x + y * y + z * z),
                          0.5 * (relative[0] + relative[4] + relative[8] - 1.0));
    }

    void expect_bytes_equal(const Tensor& actual, const Tensor& before) {
        ASSERT_EQ(actual.is_valid(), before.is_valid());
        if (!actual.is_valid())
            return;
        ASSERT_EQ(actual.shape(), before.shape());
        ASSERT_EQ(actual.dtype(), before.dtype());
        if (actual.numel() == 0)
            return;
        const auto a = actual.to(Device::CPU).contiguous();
        const auto b = before.to(Device::CPU).contiguous();
        EXPECT_EQ(std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()), 0);
    }

    class CameraPosePhotometricTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                GTEST_SKIP() << "CUDA unavailable: this is not a passed pose-refinement gate";
            camera = std::make_unique<Camera>(
                Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3}, Device::CPU),
                Tensor::zeros({3}, Device::CPU),
                55.0f, 55.0f, WIDTH / 2.0f, HEIGHT / 2.0f,
                Tensor{}, Tensor{}, CameraModelType::PINHOLE,
                "synthetic_pose_gate", std::filesystem::path{}, std::filesystem::path{},
                WIDTH, HEIGHT, 7);
            background = Tensor::zeros({3}, Device::CUDA);
            setup_scene(3);
        }

        void setup_scene(int degree, bool on_axis = false) {
            const size_t n = on_axis ? 1 : 9;
            std::vector<float> means, scales, rotations, colors, sh;
            const int rest = (degree + 1) * (degree + 1) - 1;
            for (size_t i = 0; i < n; ++i) {
                const float angle = on_axis ? 0.0f : 0.17f * static_cast<float>(i);
                means.insert(means.end(), {on_axis ? 0.0f : (static_cast<int>(i % 3) - 1) * 0.65f,
                                           on_axis ? 0.0f : (static_cast<int>(i / 3) - 1) * 0.42f,
                                           2.8f + (on_axis ? 0.0f : 0.31f * static_cast<float>(i % 4))});
                scales.insert(scales.end(), {std::log(on_axis ? 0.32f : 0.14f),
                                             std::log(0.055f), std::log(0.09f)});
                rotations.insert(rotations.end(), {std::cos(angle / 2), 0, 0, std::sin(angle / 2)});
                colors.insert(colors.end(), {0.18f + 0.04f * static_cast<float>(i), -0.22f, 0.3f});
                for (int k = 0; k < rest * 3; ++k)
                    sh.push_back(0.16f * std::sin(0.7f * static_cast<float>(k + 1) + angle));
            }
            optimizer.reset();
            scene = std::make_unique<SplatData>(
                degree, Tensor::from_vector(means, {n, 3}, Device::CUDA),
                Tensor::from_vector(colors, {n, 1, 3}, Device::CUDA),
                Tensor::from_vector(sh, {n, static_cast<size_t>(rest), 3}, Device::CUDA),
                Tensor::from_vector(scales, {n, 3}, Device::CUDA),
                Tensor::from_vector(rotations, {n, 4}, Device::CUDA),
                Tensor::full({n, 1}, 0.2f, Device::CUDA), 1.0f);
            scene->set_active_sh_degree(degree);
            scene->_densification_info = Tensor::full({2, n}, 0.125f, Device::CUDA);
            optimizer = std::make_unique<AdamOptimizer>(*scene, AdamConfig{});
            optimizer->allocate_gradients();
            optimizer->zero_grad(0);
        }

        FastGSCameraPoseOverride pose_tensors(const Matrix4& pose) {
            std::vector<float> center(3, 0.0f);
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    center[j] -= pose[4 * i + j] * pose[4 * i + 3];
            return {
                Tensor::from_vector(std::vector<float>(pose.begin(), pose.end()), {1, 4, 4}, Device::CUDA),
                Tensor::from_vector(center, {3}, Device::CUDA)};
        }

        std::pair<RenderOutput, FastRasterizeContext> forward(const Matrix4& pose,
                                                              int x = 0, int width = 0,
                                                              bool mip = false, bool normal = false) {
            auto tensors = pose_tensors(pose);
            auto result = fast_rasterize_forward(
                *camera, *scene, background, x, 0, width, 0, mip, {}, normal, &tensors);
            if (!result)
                throw std::runtime_error(lfs::format_for_developer(result.error()));
            return std::move(*result);
        }

        std::vector<float> render(const Matrix4& pose, bool mip = false) {
            auto result = forward(pose, 0, 0, mip);
            return values(result.first.image);
        }

        Vector6 gradient(const Matrix4& pose, const std::vector<float>& upstream,
                         const Tensor& alpha_extra = {}, int x = 0, int width = 0, bool mip = false) {
            auto result = forward(pose, x, width, mip);
            auto dimage = Tensor::from_vector(upstream, {3, HEIGHT, static_cast<size_t>(width ? width : WIDTH)}, Device::CUDA);
            auto output = Tensor::full({4, 4}, 123.0f, Device::CUDA);
            fast_rasterize_backward(result.second, dimage, *scene, *optimizer, alpha_extra, {},
                                    DensificationType::None, 1, {}, {}, {}, &output,
                                    FastGSBackwardMode::CameraOnly);
            const auto cpu = values(output);
            Matrix4 matrix{};
            std::copy(cpu.begin(), cpu.end(), matrix.begin());
            for (int i = 12; i < 16; ++i)
                EXPECT_FLOAT_EQ(matrix[i], 0.0f);
            const auto tangent = left_increment_gradient(pose, matrix);
            return {tangent[0], tangent[1], tangent[2], tangent[3], tangent[4], tangent[5]};
        }

        std::vector<float> spatial_weights() {
            std::vector<float> weights(PIXELS);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < HEIGHT; ++y)
                    for (int x = 0; x < WIDTH; ++x)
                        weights[(c * HEIGHT + y) * WIDTH + x] =
                            (0.5f + std::sin(0.13f * x + 0.21f * y + 0.3f * c)) / PIXELS;
            return weights;
        }

        std::unique_ptr<Camera> camera;
        std::unique_ptr<SplatData> scene;
        std::unique_ptr<AdamOptimizer> optimizer;
        Tensor background;
    };

    TEST_F(CameraPosePhotometricTest, AnisotropicSH3MatchesSixAxisFiniteDifferences) {
        const auto pose = exp_se3({0.06f, -0.04f, 0.08f, 0.035f, -0.025f, 0.02f});
        const auto weights = spatial_weights();
        const auto analytic = gradient(pose, weights);
        // At 1e-3 this fixture crosses the alpha=1/255 contribution cutoff.
        // Its secant includes discrete visibility jumps, not just the local
        // derivative. Keep it as a diagnostic, not a gradient reference.
        // tools/diagnose_camera_pose_cutoff.py establishes unchanged membership
        // at BOTH predetermined fine scales. Require both to agree with CUDA
        // and each other; never select whichever epsilon happens to pass.
        // The recovery driver retains its original, larger step.
        constexpr std::array<float, 3> steps{EPSILON, 1.0e-5f, 5.0e-6f};
        Vector6 previous{};
        for (size_t scale = 0; scale < steps.size(); ++scale) {
            const float step = steps[scale];
            SCOPED_TRACE(::testing::Message() << "epsilon=" << step);
            double error = 0.0, magnitude = 0.0;
            for (int axis = 0; axis < 6; ++axis) {
                const double numeric = (scalar_product(render(offset(pose, axis, step)), weights) -
                                        scalar_product(render(offset(pose, axis, -step)), weights)) /
                                       (2 * step);
                SCOPED_TRACE(axis);
                std::cout << "pose gradient: epsilon=" << step << " axis=" << axis
                          << " analytic=" << analytic[axis] << " numeric=" << numeric << '\n';
                EXPECT_TRUE(std::isfinite(analytic[axis]));
                EXPECT_TRUE(std::isfinite(numeric));
                if (scale > 0)
                    EXPECT_NEAR(analytic[axis], numeric, 2.0e-5 + 0.06 * std::abs(numeric));
                if (scale > 1)
                    EXPECT_NEAR(previous[axis], numeric, 2.0e-5 + 0.06 * std::abs(numeric));
                previous[axis] = numeric;
                error += std::pow(analytic[axis] - numeric, 2);
                magnitude += numeric * numeric;
            }
            ASSERT_GT(magnitude, 1.0e-10);
            const double relative_error = std::sqrt(error / magnitude);
            RecordProperty("gradient_relative_error_scale_" + std::to_string(scale),
                           std::to_string(relative_error));
            if (scale > 0)
                EXPECT_LT(relative_error, 0.07);
        }
    }

    TEST_F(CameraPosePhotometricTest, OnAxisAnisotropicRollHasNonzeroGradient) {
        setup_scene(0, true);
        const auto pose = identity_transform();
        std::vector<float> weights(PIXELS);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < HEIGHT; ++y)
                for (int x = 0; x < WIDTH; ++x)
                    weights[(c * HEIGHT + y) * WIDTH + x] =
                        static_cast<float>((x - WIDTH / 2) * (y - HEIGHT / 2)) / PIXELS;
        const auto analytic = gradient(pose, weights);
        const double numeric = (scalar_product(render(offset(pose, 5, EPSILON)), weights) -
                                scalar_product(render(offset(pose, 5, -EPSILON)), weights)) /
                               (2 * EPSILON);
        ASSERT_GT(std::abs(numeric), 1.0e-4);
        EXPECT_NEAR(analytic[5], numeric, 0.03 * std::abs(numeric));
    }

    TEST_F(CameraPosePhotometricTest, MipCameraGradientMatchesFiniteDifferences) {
        const auto pose = exp_se3({0.06f, -0.04f, 0.08f, 0.035f, -0.025f, 0.02f});
        const auto weights = spatial_weights();
        const auto before = scene->means().clone();
        const auto analytic = gradient(pose, weights, {}, 0, 0, true);
        for (const float step : {1.0e-5f, 5.0e-6f}) {
            for (int axis = 0; axis < 6; ++axis) {
                SCOPED_TRACE(::testing::Message() << "axis=" << axis << " epsilon=" << step);
                const double numeric = (scalar_product(render(offset(pose, axis, step), true), weights) -
                    scalar_product(render(offset(pose, axis, -step), true), weights)) / (2 * step);
                EXPECT_NEAR(analytic[axis], numeric, 2.0e-5 + 0.06 * std::abs(numeric));
            }
        }
        expect_bytes_equal(before, scene->means());
    }

    TEST_F(CameraPosePhotometricTest, SHViewDirectionSurvivesGeometryCancellation) {
        setup_scene(3, true);
        const auto pose = exp_se3({0.1f, -0.04f, 0.08f, 0.02f, -0.03f, 0.01f});
        auto result = forward(pose);
        const auto image = values(result.first.image);
        const auto alpha = values(result.first.alpha);
        const double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
        const double image_sum = std::accumulate(image.begin(), image.end(), 0.0);
        result.second.release_forward_context();
        ASSERT_GT(alpha_sum, 1.0);
        // One splat on black: sum(RGB)/sum(alpha) cancels footprint changes.
        const std::vector<float> upstream(PIXELS, static_cast<float>(1.0 / alpha_sum));
        const auto dalpha = Tensor::full({1, HEIGHT, WIDTH},
                                         static_cast<float>(-image_sum / (alpha_sum * alpha_sum)), Device::CUDA);
        const auto analytic = gradient(pose, upstream, dalpha);
        auto objective = [&](const Matrix4& p) {
            auto r = forward(p);
            const auto rgb = values(r.first.image);
            const auto a = values(r.first.alpha);
            return std::accumulate(rgb.begin(), rgb.end(), 0.0) /
                   std::accumulate(a.begin(), a.end(), 0.0);
        };
        double translation_signal = 0.0;
        for (int axis = 0; axis < 6; ++axis) {
            const double numeric = (objective(offset(pose, axis, EPSILON)) -
                                    objective(offset(pose, axis, -EPSILON))) /
                                   (2 * EPSILON);
            SCOPED_TRACE(axis);
            EXPECT_NEAR(analytic[axis], numeric, 2.0e-4 + 0.03 * std::abs(numeric));
            if (axis < 3)
                translation_signal += numeric * numeric;
        }
        EXPECT_GT(translation_signal, 1.0e-6);
    }

    TEST_F(CameraPosePhotometricTest, CameraOnlyPreservesModelOptimizerAndSource) {
        const auto before = scene->clone();
        const auto info = scene->_densification_info.clone();
        const auto w2c = camera->world_view_transform().clone();
        const auto center = camera->cam_position().clone();
        struct State {
            ParamType type;
            Tensor moments, bounds, grad;
            int64_t step;
        };
        std::vector<State> saved;
        for (auto type : {ParamType::Means, ParamType::Scaling, ParamType::Rotation,
                          ParamType::Opacity, ParamType::Sh0, ParamType::ShN}) {
            const auto* state = optimizer->get_state(type);
            ASSERT_NE(state, nullptr);
            saved.push_back({type, state->exp_avg.clone(), state->joint_bounds.clone(),
                             state->grad.is_valid() ? state->grad.clone() : Tensor{}, state->step_count});
        }
        const auto pose = exp_se3({0.03f, -0.02f, 0.04f, 0.01f, 0.02f, -0.01f});
        const auto first = gradient(pose, spatial_weights());
        {
            auto rendered = forward(pose);
            const auto rendered_pose = rendered.first.pose_world_view_transform;
            const auto rendered_center = rendered.first.pose_cam_position;
            ASSERT_TRUE(rendered_pose.is_valid());
            ASSERT_TRUE(rendered_center.is_valid());
            const auto expected = pose_tensors(pose);
            expect_bytes_equal(rendered_pose, expected.world_view_transform);
            expect_bytes_equal(rendered_center, expected.cam_position);
            rendered.second.release_forward_context();
            // Geometry consumers may retain the pose with cached pixels across
            // a later render of the same imported camera.
            auto later = forward(identity_transform());
            expect_bytes_equal(rendered_pose, expected.world_view_transform);
            expect_bytes_equal(rendered_center, expected.cam_position);
        }
        const auto second = gradient(pose, spatial_weights());
        for (int axis = 0; axis < 6; ++axis)
            EXPECT_NEAR(first[axis], second[axis], 1.0e-7);
        const auto target = Tensor::from_vector(render(identity_transform()), {3, HEIGHT, WIDTH}, Device::CUDA);
        FastGSPoseEvaluator evaluator(*camera, *scene, *optimizer, background, make_pose_mse_objective(target));
        const auto image = evaluator.evaluate(pose);
        EXPECT_NEAR(image.loss, evaluator.loss(pose), 1.0e-9);
        EXPECT_GT(image.loss, 0.0);
        const auto current = render(pose);
        const auto target_values = values(target);
        std::vector<float> upstream(PIXELS);
        for (int p = 0; p < PIXELS; ++p)
            upstream[p] = 2.0f * (current[p] - target_values[p]) / PIXELS;
        const auto reference = gradient(pose, upstream);
        for (int axis = 0; axis < 6; ++axis)
            EXPECT_NEAR(image.gradient[axis], reference[axis], 1.0e-7 + std::abs(reference[axis]) * 1.0e-5);
        auto invalid_pose = pose;
        invalid_pose[0] = 10;
        EXPECT_THROW((void)evaluator.evaluate(invalid_pose), std::invalid_argument);
        FastGSPoseEvaluator invalid_objective(*camera, *scene, *optimizer, background,
                                              [](const RenderOutput&, bool) { return PoseObjectiveResult{}; });
        EXPECT_THROW((void)invalid_objective.evaluate(pose), std::invalid_argument);
        auto reference_pose = identity_transform();
        reference_pose[3] = 1;
        PoseSessionConfig settings;
        settings.choose_anchors = false;
        settings.optimizer.scene_scale = 3;
        PoseRefinementSession session(1, {{camera->uid(), pose, PoseRole::Train}, {camera->uid() + 1, identity_transform(), PoseRole::Anchor}, {camera->uid() + 2, reference_pose, PoseRole::Anchor}}, settings);
        const auto visit = evaluator.visit(session, settings.warmup_iterations, 1);
        EXPECT_GT(visit.accepted_steps, 0);
        EXPECT_LT(evaluator.loss(session.current_pose(camera->uid())), image.loss);
        expect_bytes_equal(scene->means(), before.means());
        expect_bytes_equal(scene->scaling_raw(), before.scaling_raw());
        expect_bytes_equal(scene->rotation_raw(), before.rotation_raw());
        expect_bytes_equal(scene->opacity_raw(), before.opacity_raw());
        expect_bytes_equal(scene->sh0(), before.sh0());
        expect_bytes_equal(scene->shN(), before.shN());
        expect_bytes_equal(scene->_densification_info, info);
        expect_bytes_equal(camera->world_view_transform(), w2c);
        expect_bytes_equal(camera->cam_position(), center);
        for (const auto& old : saved) {
            const auto* state = optimizer->get_state(old.type);
            ASSERT_NE(state, nullptr);
            expect_bytes_equal(state->exp_avg, old.moments);
            expect_bytes_equal(state->joint_bounds, old.bounds);
            expect_bytes_equal(state->grad, old.grad);
            EXPECT_EQ(state->step_count, old.step);
        }
    }

    TEST_F(CameraPosePhotometricTest, SparseGuardRejectsDriftBeforePhotometricRendering) {
        std::vector<Camera::SfmObservation> observations;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 5; ++col) {
                const float x = (col - 2) * 0.4f;
                const float y = (row - 1.5f) * 0.4f;
                const float z = 3.0f + 0.1f * (row + col);
                observations.push_back({55 * x / z + WIDTH / 2.0f, 55 * y / z + HEIGHT / 2.0f, x, y, z});
            }
        }
        camera->set_sfm_observations(observations);
        ASSERT_TRUE(make_sparse_reprojection_guard(*camera).active());
        const auto source_before = camera->world_view_transform().clone();
        const auto means_before = scene->means().clone();
        int objective_calls = 0;
        FastGSPoseEvaluator evaluator(*camera, *scene, *optimizer, background,
                                      [&](const RenderOutput&, bool) {
                                          ++objective_calls;
                                          return PoseObjectiveResult{0.1, {}, {}};
                                      });
        const auto drift = exp_se3({0.01f, 0, 0, 0, 0, 0});
        EXPECT_FALSE(evaluator.allows(drift));
        EXPECT_TRUE(std::isinf(evaluator.loss(drift)));
        EXPECT_EQ(objective_calls, 0);
        EXPECT_NEAR(evaluator.loss(identity_transform()), 0.1, 1e-12);
        EXPECT_EQ(objective_calls, 1);
        expect_bytes_equal(camera->world_view_transform(), source_before);
        expect_bytes_equal(scene->means(), means_before);

        camera->set_sfm_observations({});
        EXPECT_FALSE(make_sparse_reprojection_guard(*camera).active());
        FastGSPoseEvaluator without_sparse(*camera, *scene, *optimizer, background,
                                           [](const RenderOutput&, bool) { return PoseObjectiveResult{0.2, {}, {}}; });
        EXPECT_TRUE(without_sparse.allows(drift));
        EXPECT_NEAR(without_sparse.loss(drift), 0.2, 1e-12);
    }

    TEST_F(CameraPosePhotometricTest, SparseGuardSupportsUndistortionWithoutMutatingMeasurements) {
        UndistortParams params{};
        params.src_fx = params.src_fy = 55;
        params.src_cx = WIDTH / 2.0f;
        params.src_cy = HEIGHT / 2.0f;
        params.src_width = WIDTH;
        params.src_height = HEIGHT;
        params.dst_fx = params.dst_fy = 55;
        params.dst_cx = params.src_cx - 2;
        params.dst_cy = params.src_cy - 1;
        params.dst_width = WIDTH - 4;
        params.dst_height = HEIGHT - 2;
        params.model_type = CameraModelType::PINHOLE;
        params.num_distortion = 5;
        params.distortion[0] = 0.12f;
        params.distortion[3] = 0.01f;
        params.distortion[4] = -0.005f;
        std::vector<Camera::SfmObservation> observations;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 5; ++col) {
                const float x = (col - 2) * 0.4f, y = (row - 1.5f) * 0.4f;
                const float z = 3.0f + 0.1f * (row + col);
                const float nx = x / z, ny = y / z, r2 = nx * nx + ny * ny;
                const float dx = nx * (1 + 0.12f * r2) + 0.02f * nx * ny - 0.005f * (r2 + 2 * nx * nx);
                const float dy = ny * (1 + 0.12f * r2) + 0.01f * (r2 + 2 * ny * ny) - 0.01f * nx * ny;
                observations.push_back({55 * dx + params.src_cx, 55 * dy + params.src_cy, x, y, z,
                                        static_cast<std::uint64_t>(observations.size())});
            }
        }
        camera->set_sfm_observations(observations);
        camera->adopt_undistortion(params);
        camera->prepare_undistortion();
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto guard = make_sparse_reprojection_guard(*camera);
            ASSERT_TRUE(guard.active());
            EXPECT_LT(guard.source_error(), 1e-6);
            EXPECT_TRUE(guard.allows(identity_transform()));
            EXPECT_FALSE(guard.allows(exp_se3({0.01f, 0, 0, 0, 0, 0})));
            EXPECT_TRUE(make_sparse_track_measurements(*camera, false).empty());
            const auto measurements = make_sparse_track_measurements(*camera, true);
            ASSERT_EQ(measurements.size(), observations.size());
            for (size_t i = 0; i < measurements.size(); ++i) {
                EXPECT_EQ(measurements[i].point_id, observations[i].point3d_id);
                EXPECT_NEAR(measurements[i].u, params.dst_fx * observations[i].x / observations[i].z + params.dst_cx, 0.002);
                EXPECT_NEAR(measurements[i].v, params.dst_fy * observations[i].y / observations[i].z + params.dst_cy, 0.002);
            }
            ASSERT_EQ(camera->sfm_observations().size(), observations.size());
            for (size_t i = 0; i < observations.size(); ++i) {
                EXPECT_EQ(camera->sfm_observations()[i].u, observations[i].u);
                EXPECT_EQ(camera->sfm_observations()[i].v, observations[i].v);
            }
        }
        EXPECT_EQ(camera->focal_x(), params.dst_fx);
        EXPECT_EQ(camera->camera_width(), params.dst_width);
    }

    TEST_F(CameraPosePhotometricTest, UndistortionInverseMappingSupportsModelsCropScaleAndInvalidInputs) {
        for (const auto model : {CameraModelType::PINHOLE, CameraModelType::FISHEYE,
                                 CameraModelType::THIN_PRISM_FISHEYE}) {
            for (const float scale : {1.0f, 0.25f}) {
                UndistortParams p{};
                p.model_type = model;
                p.src_fx = p.src_fy = 800 * scale;
                p.src_cx = 600 * scale;
                p.src_cy = 400 * scale;
                p.src_width = static_cast<int>(1200 * scale);
                p.src_height = static_cast<int>(800 * scale);
                p.dst_fx = 820 * scale;
                p.dst_fy = 810 * scale;
                p.dst_cx = 580 * scale;
                p.dst_cy = 390 * scale;
                p.dst_width = static_cast<int>(1160 * scale);
                p.dst_height = static_cast<int>(780 * scale);
                p.num_distortion = 1;
                p.distortion[0] = 0.1f;
                const float x = 0.2f, y = -0.15f, r = std::hypot(x, y);
                const float theta = std::atan(r);
                const float factor = model == CameraModelType::PINHOLE
                                         ? 1 + 0.1f * r * r
                                         : theta * (1 + 0.1f * theta * theta) / r;
                const float u = p.src_fx * x * factor + p.src_cx;
                const float v = p.src_fy * y * factor + p.src_cy;
                float a = -1, b = -1;
                ASSERT_TRUE(undistort_observation(p, u, v, a, b));
                EXPECT_NEAR(a, p.dst_fx * x + p.dst_cx, 0.002f);
                EXPECT_NEAR(b, p.dst_fy * y + p.dst_cy, 0.002f);
                a = b = -1;
                EXPECT_FALSE(undistort_observation(p, -1, v, a, b));
                EXPECT_FALSE(undistort_observation(p, std::numeric_limits<float>::quiet_NaN(), v, a, b));
                auto invalid = p;
                invalid.dst_cx = -10000;
                EXPECT_FALSE(undistort_observation(invalid, u, v, a, b));
                invalid = p;
                invalid.src_fx = 0;
                EXPECT_FALSE(undistort_observation(invalid, u, v, a, b));
                invalid = p;
                invalid.model_type = CameraModelType::EQUIRECTANGULAR;
                EXPECT_FALSE(undistort_observation(invalid, u, v, a, b));
                invalid = p;
                invalid.num_distortion = 13;
                EXPECT_FALSE(undistort_observation(invalid, u, v, a, b));
                EXPECT_EQ(a, -1);
                EXPECT_EQ(b, -1);
            }
        }
    }

    TEST_F(CameraPosePhotometricTest, TiledGradientMatchesFullImage) {
        const auto pose = identity_transform();
        const auto weights = spatial_weights();
        const auto full = gradient(pose, weights);
        Vector6 sum{};
        for (int tile = 0; tile < 2; ++tile) {
            std::vector<float> tile_weights(PIXELS / 2);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < HEIGHT; ++y)
                    for (int x = 0; x < WIDTH / 2; ++x)
                        tile_weights[(c * HEIGHT + y) * (WIDTH / 2) + x] =
                            weights[(c * HEIGHT + y) * WIDTH + x + tile * WIDTH / 2];
            const auto partial = gradient(pose, tile_weights, {}, tile * WIDTH / 2, WIDTH / 2);
            for (int axis = 0; axis < 6; ++axis)
                sum[axis] += partial[axis];
        }
        for (int axis = 0; axis < 6; ++axis)
            EXPECT_NEAR(sum[axis], full[axis], 1.0e-5 + 0.01 * std::abs(full[axis]));
    }

    TEST_F(CameraPosePhotometricTest, OptionalGradientPreservesJointGaussianUpdate) {
        auto baseline = scene->clone();
        AdamOptimizer baseline_optimizer(baseline, AdamConfig{});
        baseline_optimizer.allocate_gradients();
        baseline_optimizer.zero_grad(0);
        const auto upstream = Tensor::from_vector(spatial_weights(), {3, HEIGHT, WIDTH}, Device::CUDA);
        auto plain = fast_rasterize_forward(*camera, baseline, background);
        ASSERT_TRUE(plain.has_value());
        fast_rasterize_backward(plain->second, upstream, baseline, baseline_optimizer,
                                {}, {}, DensificationType::None, 1);
        auto optional = fast_rasterize_forward(*camera, *scene, background);
        ASSERT_TRUE(optional.has_value());
        auto output = Tensor::zeros({4, 4}, Device::CUDA);
        fast_rasterize_backward(optional->second, upstream, *scene, *optimizer,
                                {}, {}, DensificationType::None, 1, {}, {}, {}, &output);
        EXPECT_GT(output.abs().sum().item<float>(), 0.0f);
        for (const auto pair : {
                 std::pair{&scene->means(), &baseline.means()},
                 std::pair{&scene->scaling_raw(), &baseline.scaling_raw()},
                 std::pair{&scene->rotation_raw(), &baseline.rotation_raw()},
                 std::pair{&scene->opacity_raw(), &baseline.opacity_raw()},
                 std::pair{&scene->sh0(), &baseline.sh0()},
                 std::pair{&scene->shN(), &baseline.shN()}}) {
            const auto actual = values(*pair.first);
            const auto expected = values(*pair.second);
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i)
                EXPECT_NEAR(actual[i], expected[i], 1.0e-6f);
        }
        EXPECT_EQ(optimizer->get_step_count(ParamType::Means),
                  baseline_optimizer.get_step_count(ParamType::Means));
    }

    TEST_F(CameraPosePhotometricTest, RejectsInvalidContractsBeforeBackward) {
        // Pose uploads feed CUDA kernels even when the caller selects Vulkan
        // for generic GPU factories. This needs no Vulkan device allocation.
        {
            const GpuBackendScope backend_scope(GpuBackend::Vulkan);
            const auto pose = make_fastgs_pose_override(camera->uid(), identity_transform());
            EXPECT_EQ(gpu_backend_of(pose.world_view_transform), GpuBackend::CUDA);
            EXPECT_EQ(gpu_backend_of(pose.cam_position), GpuBackend::CUDA);
        }
        auto invalid_pose = pose_tensors(identity_transform());
        invalid_pose.cam_position = Tensor::zeros({3}, Device::CPU);
        EXPECT_FALSE(fast_rasterize_forward(*camera, *scene, background, 0, 0, 0, 0,
                                            false, {}, false, &invalid_pose)
                         .has_value());
        auto result = forward(identity_transform());
        auto upstream = Tensor::ones({3, HEIGHT, WIDTH}, Device::CUDA);
        EXPECT_THROW(fast_rasterize_backward(result.second, upstream, *scene, *optimizer,
                                             {}, {}, DensificationType::None, 1, {}, {}, {}, nullptr,
                                             FastGSBackwardMode::CameraOnly),
                     std::invalid_argument);
        auto wrong_shape = Tensor::zeros({16}, Device::CUDA);
        EXPECT_THROW(fast_rasterize_backward(result.second, upstream, *scene, *optimizer,
                                             {}, {}, DensificationType::None, 1, {}, {}, {}, &wrong_shape),
                     std::invalid_argument);
        auto cpu_output = Tensor::zeros({4, 4}, Device::CPU);
        EXPECT_THROW(fast_rasterize_backward(result.second, upstream, *scene, *optimizer,
                                             {}, {}, DensificationType::None, 1, {}, {}, {}, &cpu_output),
                     std::invalid_argument);
        auto wrong_dtype = Tensor::zeros({4, 4}, Device::CUDA, DataType::Int32);
        EXPECT_THROW(fast_rasterize_backward(result.second, upstream, *scene, *optimizer,
                                             {}, {}, DensificationType::None, 1, {}, {}, {}, &wrong_dtype),
                     std::invalid_argument);
        auto strided = Tensor::zeros({4, 4}, Device::CUDA).transpose(0, 1);
        EXPECT_THROW(fast_rasterize_backward(result.second, upstream, *scene, *optimizer,
                                             {}, {}, DensificationType::None, 1, {}, {}, {}, &strided),
                     std::invalid_argument);
        auto output = Tensor::zeros({4, 4}, Device::CUDA);
        auto pose_alias = result.second.pose_world_view_transform.reshape({4, 4});
        EXPECT_THROW(fast_rasterize_backward(result.second, upstream, *scene, *optimizer,
                                             {}, {}, DensificationType::None, 1, {}, {}, {}, &pose_alias),
                     std::invalid_argument);
        auto normals = Tensor::ones({3, HEIGHT, WIDTH}, Device::CUDA);
        EXPECT_THROW(fast_rasterize_backward(result.second, upstream, *scene, *optimizer,
                                             {}, {}, DensificationType::None, 1, {}, {}, normals, &output),
                     std::invalid_argument);
        result.second.release_forward_context();
        auto mip = forward(identity_transform(), 0, 0, true);
        EXPECT_NO_THROW(fast_rasterize_backward(mip.second, upstream, *scene, *optimizer,
                                               {}, {}, DensificationType::None, 1, {}, {}, {}, &output,
                                               FastGSBackwardMode::CameraOnly));
    }

    TEST_F(CameraPosePhotometricTest, RecoversPerturbedPoseFromImagesWithFixedGeometry) {
        const auto truth = exp_se3({0.03f, -0.02f, 0.04f, 0.025f, -0.018f, 0.012f});
        const auto target = render(truth);
        const std::array<Twist, 3> perturbations{{{0.035f, -0.025f, 0.03f, 0.012f, -0.01f, 0.018f},
                                                  {-0.025f, 0.02f, -0.035f, -0.015f, 0.012f, -0.012f},
                                                  {0.018f, 0.03f, -0.025f, 0.018f, 0.008f, -0.015f}}};
        int trial = 0;
        for (const auto& perturbation : perturbations) {
            SCOPED_TRACE(trial);
            auto pose = apply_left_increment(perturbation, truth);
            const double initial_loss = squared_loss(render(pose), target);
            const double initial_center = center_error(pose, truth);
            const double initial_rotation = rotation_error(pose, truth);
            ASSERT_GT(initial_loss, 1.0e-7);
            int accepted = 0;
            // Bounded reference solver for the gate, NOT the production trainer.
            // All updates and acceptance decisions use only rendered RGB loss.
            for (int iteration = 0; iteration < 24; ++iteration) {
                const auto current = render(pose);
                const double loss = squared_loss(current, target);
                if (loss < initial_loss * 1.0e-4)
                    break;
                std::vector<float> upstream(PIXELS);
                for (int p = 0; p < PIXELS; ++p)
                    upstream[p] = 2.0f * (current[p] - target[p]) / PIXELS;
                const auto g = gradient(pose, upstream);
                std::array<std::vector<float>, 6> jacobian;
                for (int axis = 0; axis < 6; ++axis) {
                    const auto plus = render(offset(pose, axis, EPSILON));
                    const auto minus = render(offset(pose, axis, -EPSILON));
                    jacobian[axis].resize(PIXELS);
                    for (int p = 0; p < PIXELS; ++p)
                        jacobian[axis][p] = (plus[p] - minus[p]) / (2 * EPSILON);
                }
                Matrix6 hessian{};
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j)
                        hessian[i][j] = 2.0 * scalar_product(jacobian[i], jacobian[j]) / PIXELS;
                    hessian[i][i] += 1.0e-5;
                }
                auto step = solve(hessian, g);
                double scale = 1.0;
                for (int axis = 0; axis < 6; ++axis)
                    scale = std::min(scale, (axis < 3 ? 0.04 : 0.02) /
                                                std::max(std::abs(step[axis]), 1.0e-12));
                bool improved = false;
                for (int backtrack = 0; backtrack < 8; ++backtrack) {
                    Twist increment{};
                    for (int axis = 0; axis < 6; ++axis)
                        increment[axis] = static_cast<float>(-scale * step[axis]);
                    const auto candidate = apply_left_increment(increment, pose);
                    const double candidate_loss = squared_loss(render(candidate), target);
                    if (std::isfinite(candidate_loss) && candidate_loss < loss) {
                        pose = candidate;
                        ++accepted;
                        improved = true;
                        break;
                    }
                    scale *= 0.5;
                }
                if (!improved)
                    break;
            }
            const double final_loss = squared_loss(render(pose), target);
            const double final_center = center_error(pose, truth);
            const double final_rotation = rotation_error(pose, truth);
            const auto prefix = "trial_" + std::to_string(trial++) + "_";
            RecordProperty(prefix + "loss_ratio", std::to_string(final_loss / initial_loss));
            RecordProperty(prefix + "center_error_ratio", std::to_string(final_center / initial_center));
            RecordProperty(prefix + "rotation_error_ratio", std::to_string(final_rotation / initial_rotation));
            RecordProperty(prefix + "accepted_steps", accepted);
            EXPECT_GT(accepted, 0);
            EXPECT_LT(final_loss, initial_loss * 0.10);
            EXPECT_LT(final_center, initial_center * 0.30);
            EXPECT_LT(final_rotation, initial_rotation * 0.30);
        }
    }
    TEST_F(CameraPosePhotometricTest, BoundedControllerRecoversPoseFromImages) {
        const auto truth = exp_se3({0.03f, -0.02f, 0.04f, 0.025f, -0.018f, 0.012f});
        const auto target = render(truth);
        const auto target_tensor = Tensor::from_vector(target, {3, HEIGHT, WIDTH}, Device::CUDA);
        FastGSPoseEvaluator evaluator(*camera, *scene, *optimizer, background, make_pose_mse_objective(target_tensor));
        RecordProperty("production_evaluator", 1);
        const std::array<Twist, 3> perturbations{{{0.035f, -0.025f, 0.03f, 0.012f, -0.01f, 0.018f},
                                                  {-0.025f, 0.02f, -0.035f, -0.015f, 0.012f, -0.012f},
                                                  {0.018f, 0.03f, -0.025f, 0.018f, 0.008f, -0.015f}}};
        for (size_t trial = 0; trial < perturbations.size(); ++trial) {
            SCOPED_TRACE(trial);
            const auto initial = apply_left_increment(perturbations[trial], truth);
            BoundedPoseConfig config;
            config.scene_scale = 3.0; // Declared scene units, never derived from the target pose.
            BoundedPoseOptimizer controller(camera->uid(), initial, config);
            const double initial_loss = squared_loss(render(initial), target);
            const double initial_center = center_error(initial, truth);
            const double initial_rotation = rotation_error(initial, truth);
            int renders = 0;
            // This uses the intended bounded controller: no numerical image
            // Jacobian/Hessian, no true-pose gradient, at most 80 steps and
            // 8 candidate renders per step. Same model for all evaluations.
            for (int iteration = 0; iteration < 80; ++iteration) {
                const auto state = controller.snapshot();
                const auto image = evaluator.evaluate(state.current);
                const double loss = image.loss;
                if (loss < initial_loss * 1.0e-4)
                    break;
                PoseEvaluation evaluation{state.uid, 1, state.revision, loss, image.gradient};
                const auto result = controller.step(evaluation, [&](const Matrix4& candidate) {
                    ++renders;
                    return evaluator.loss(candidate);
                });
                ASSERT_LE(result.evaluations, config.max_backtracks);
                if (result.status != PoseStepStatus::Accepted)
                    break;
                EXPECT_LT(result.image_loss, loss);
            }
            const auto state = controller.snapshot();
            const double final_loss = squared_loss(render(state.current), target);
            const auto prefix = "trial_" + std::to_string(trial) + "_";
            RecordProperty(prefix + "loss_ratio", std::to_string(final_loss / initial_loss));
            RecordProperty(prefix + "center_error_ratio", std::to_string(center_error(state.current, truth) / initial_center));
            RecordProperty(prefix + "rotation_error_ratio", std::to_string(rotation_error(state.current, truth) / initial_rotation));
            RecordProperty(prefix + "accepted_steps", static_cast<int>(state.accepted_steps));
            RecordProperty(prefix + "candidate_renders", renders);
            std::cout << "bounded controller trial=" << trial << " accepted=" << state.accepted_steps
                      << " candidate_renders=" << renders << " loss_ratio=" << final_loss / initial_loss << '\n';
            EXPECT_EQ(state.source, initial);
            EXPECT_GT(state.accepted_steps, 0u);
            EXPECT_LE(renders, 80 * config.max_backtracks);
            EXPECT_LE(state.center_displacement, config.scene_scale * config.max_center_fraction);
            EXPECT_LE(state.rotation_displacement, config.max_rotation_radians);
            EXPECT_LT(final_loss, initial_loss * 0.10);
            EXPECT_LT(center_error(state.current, truth), initial_center * 0.30);
            EXPECT_LT(rotation_error(state.current, truth), initial_rotation * 0.30);
        }
    }
    class CameraPoseViewTest : public CameraPosePhotometricTest {};

    TEST_F(CameraPoseViewTest, CurrentPoseUsesUIDAndKeepsSourceImmutable) {
        const auto source = camera->world_view_transform().clone();
        PoseSessionSnapshot snapshot;
        snapshot.generation = 1;
        PoseCameraDisplay display;
        display.pose.uid = camera->uid();
        display.pose.current = exp_se3({0.02f, -0.03f, 0.01f, 0.01f, 0.02f, 0.03f});
        snapshot.cameras.push_back(display);
        ASSERT_EQ(lfs::vis::findCameraPose(&snapshot, camera->uid()), &snapshot.cameras.front());
        EXPECT_EQ(lfs::vis::findCameraPose(&snapshot, camera->uid() + 1), nullptr);
        EXPECT_EQ(lfs::vis::findCameraPose(nullptr, camera->uid()), nullptr);
        const auto current = lfs::vis::cameraWorldToCamera(*camera, &snapshot);
        ASSERT_TRUE(current.has_value());
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                EXPECT_FLOAT_EQ((*current)[col][row], display.pose.current[row * 4 + col]);
        snapshot.cameras.clear();
        const auto fallback = lfs::vis::cameraWorldToCamera(*camera, &snapshot);
        ASSERT_TRUE(fallback.has_value());
        EXPECT_EQ(*fallback, glm::mat4(1.0f));
        expect_bytes_equal(camera->world_view_transform(), source);
    }

    TEST_F(CameraPoseViewTest, FrustumAndFocusSharePoseAndSceneAxes) {
        const auto pose = exp_se3({0.02f, -0.03f, 0.01f, 0.01f, 0.02f, 0.03f});
        const auto w2c = lfs::vis::cameraPoseMatrix(pose);
        const auto scene_transform = glm::translate(glm::mat4(1.0f), glm::vec3(1, 2, 3)) *
                                     glm::rotate(glm::mat4(1.0f), 0.4f, glm::vec3(0, 0, 1));
        const auto frustum = scene_transform * glm::inverse(w2c) * lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES_4;
        const auto focus = lfs::rendering::visualizerCameraPoseFromDataWorldToCamera(
            glm::mat3(w2c), glm::vec3(w2c[3]), scene_transform);
        for (int row = 0; row < 3; ++row) {
            EXPECT_NEAR(frustum[3][row], focus.translation[row], 1e-6f);
            for (int col = 0; col < 3; ++col)
                EXPECT_NEAR(frustum[col][row], focus.rotation[col][row], 1e-6f);
        }
    }

    TEST_F(CameraPoseViewTest, DisplacementLabelClearsAndUsesNetMovement) {
        auto& localization = lfs::event::LocalizationManager::getInstance();
        struct LocalizationScope {
            ~LocalizationScope() { lfs::event::LocalizationManager::getInstance().reset(); }
        } localization_scope;
        localization.reset();
        const auto locales = std::filesystem::path(__FILE__).parent_path().parent_path() /
                             "src/visualizer/gui/resources/locales";
        ASSERT_TRUE(localization.initialize(locales.string()));
        for (const auto* key : {"training.pose.paused", "training.pose.unchanged",
                                "training.pose.tooltip", "training.pose.finished", "training.pose.corrected"})
            ASSERT_TRUE(localization.hasKey(key)) << key;
        PoseCameraDisplay display;
        display.pose.center_displacement = 0.125;
        display.pose.rotation_displacement = 1.5707963267948966;
        display.pose.accepted_steps = 999;
        EXPECT_EQ(lfs::vis::cameraPoseDisplacementLabel(&display), "\u0394 0.125 / 90.00\u00b0");
        EXPECT_TRUE(lfs::vis::cameraPoseDisplacementLabel(nullptr).empty());
        display.pose.center_displacement = 0;
        display.pose.rotation_displacement = 0;
        EXPECT_EQ(lfs::vis::cameraPoseDisplacementLabel(&display), "\u0394 0 / 0.00\u00b0");
        PoseSessionSnapshot snapshot;
        snapshot.paused = true;
        display.state = PoseDisplayState::Updated;
        display.eligible_visits = 1;
        EXPECT_EQ(lfs::vis::cameraPoseVisualState(display, snapshot), "unchanged");
        EXPECT_FALSE(lfs::vis::cameraPoseTooltip(display, snapshot).empty());
        display.state = PoseDisplayState::Anchor;
        EXPECT_EQ(lfs::vis::cameraPoseVisualState(display, snapshot), "anchor");
        display.state = PoseDisplayState::Evaluation;
        EXPECT_EQ(lfs::vis::cameraPoseVisualState(display, snapshot), "evaluation");
        display.state = PoseDisplayState::Frozen;
        snapshot.refinement_finished = true;
        display.pose.center_displacement = 0.02;
        EXPECT_EQ(lfs::vis::cameraPoseVisualState(display, snapshot), "corrected");
        EXPECT_FALSE(lfs::vis::cameraPoseTooltip(display, snapshot).empty());
        display.pose.center_displacement = 0;
        EXPECT_EQ(lfs::vis::cameraPoseVisualState(display, snapshot), "unchanged");
        display.eligible_visits = 0;
        EXPECT_EQ(lfs::vis::cameraPoseVisualState(display, snapshot), "not_refined");
        snapshot.refinement_finished = false;
        snapshot.paused = false;
        display.state = PoseDisplayState::Rejected;
        const auto rejected = lfs::vis::cameraPoseIndicator(lfs::vis::cameraPoseVisualState(display, snapshot));
        EXPECT_EQ(rejected.marker, lfs::vis::CameraPoseMarker::Cross);
        EXPECT_FLOAT_EQ(lfs::vis::cameraPoseIndicatorColor(rejected, 0.25f).a, 0.25f);
    }

    TEST(CameraPoseActivationTest, ConfigurationPreservesOptInAndRejectsUnsupportedTraining) {
        using namespace lfs::core::param;
        auto params = OptimizationParameters::mcmc_defaults();
        EXPECT_FALSE(params.refine_camera_poses);
        auto legacy = params.to_json();
        legacy.erase("refine_camera_poses");
        EXPECT_FALSE(OptimizationParameters::from_json(legacy).refine_camera_poses);
        params.refine_camera_poses = true;
        const auto restored = OptimizationParameters::from_json(params.to_json());
        EXPECT_TRUE(restored.refine_camera_poses);
        EXPECT_TRUE(restored.validate().empty());
        TrainingParameters target;
        ExplicitTrainingOverrides overrides;
        overrides.optimization_json = R"({"refine_camera_poses":true})";
        apply_explicit_training_overrides(target, overrides);
        EXPECT_TRUE(target.optimization.refine_camera_poses);
        auto invalid = params.to_json();
        invalid["refine_camera_poses"] = "true";
        EXPECT_THROW((void)OptimizationParameters::from_json(invalid), nlohmann::json::exception);
        for (const int option : {0}) {
            auto incompatible = restored;
            switch (option) {
            case 0: incompatible.gut = true; break;
            case 1: incompatible.mip_filter = true; break;
            case 2: incompatible.use_depth_loss = true; break;
            case 3: incompatible.use_normal_loss = true; break;
            case 4: incompatible.use_ppisp = true; break;
            case 5: incompatible.use_bilateral_grid = true; break;
            case 6: incompatible.use_exposure_correction = true; break;
            case 7: incompatible.enable_sparsity = true; break;
            case 8: incompatible.mask_mode = MaskMode::Ignore; break;
            case 9: incompatible.ppisp_use_controller = true; break;
            }
            EXPECT_FALSE(incompatible.validate().empty()) << option;
        }
        params.mip_filter = true;
        params.refine_camera_poses = false;
        EXPECT_TRUE(params.validate().empty());
    }

    TEST(CameraPoseActivationTest, CommandLineCapturesOptInAndRejectsConflict) {
        const char* argv[] = {"LichtFeld-Studio", "--refine-camera-poses", "--enable-mip"};
        auto ordinary = lfs::core::args::parse_args_and_params(1, argv);
        ASSERT_TRUE(ordinary.has_value()) << ordinary.error();
        EXPECT_FALSE((*ordinary)->optimization.refine_camera_poses);
        auto enabled = lfs::core::args::parse_args_and_params(2, argv);
        ASSERT_TRUE(enabled.has_value()) << enabled.error();
        EXPECT_TRUE((*enabled)->optimization.refine_camera_poses);
        EXPECT_TRUE((*enabled)->overrides.has_optimization_key("refine_camera_poses"));
        const auto combined = lfs::core::args::parse_args_and_params(3, argv);
        ASSERT_TRUE(combined.has_value()) << combined.error();
        EXPECT_TRUE((*combined)->optimization.mip_filter);
    }

    TEST(CameraPoseActivationTest, ScheduleRoundTripValidationAndCliOverrides) {
        using namespace lfs::core::param;
        auto params = OptimizationParameters::mcmc_defaults();
        params.refine_camera_poses = true;
        params.camera_pose_start_step = 750;
        params.camera_pose_end_percent = 60;
        auto restored = OptimizationParameters::from_json(params.to_json());
        EXPECT_EQ(restored.camera_pose_start_step, 750);
        EXPECT_EQ(restored.camera_pose_end_percent, 60);
        EXPECT_EQ(restored.resolved_camera_pose_stop_step(), 18000);
        EXPECT_TRUE(restored.validate().empty());
        auto controller = restored;
        controller.camera_pose_end_percent = 100;
        controller.ppisp_use_controller = true;
        controller.ppisp_controller_activation_step = 12000;
        EXPECT_EQ(controller.resolved_camera_pose_stop_step(), 12000);
        controller.camera_pose_start_step = 12000;
        EXPECT_FALSE(controller.validate().empty());
        auto rounding = restored;
        rounding.iterations = 30001;
        rounding.camera_pose_end_percent = 29;
        EXPECT_EQ(rounding.resolved_camera_pose_stop_step(), 8700);
        restored.camera_pose_end_percent = 0;
        EXPECT_FALSE(restored.validate().empty());
        restored.refine_camera_poses = false;
        EXPECT_TRUE(restored.validate().empty());
        restored = params;
        restored.camera_pose_start_step = static_cast<int>(restored.iterations);
        EXPECT_FALSE(restored.validate().empty());
        const char* argv[] = {"LichtFeld-Studio", "--refine-camera-poses", "--camera-pose-start-step", "750", "--camera-pose-end-percent", "60"};
        const auto parsed = lfs::core::args::parse_args_and_params(6, argv);
        ASSERT_TRUE(parsed.has_value()) << parsed.error();
        EXPECT_EQ((*parsed)->optimization.camera_pose_start_step, 750);
        EXPECT_EQ((*parsed)->optimization.camera_pose_end_percent, 60);
        EXPECT_TRUE((*parsed)->overrides.has_optimization_key("camera_pose_start_step"));
        EXPECT_TRUE((*parsed)->overrides.has_optimization_key("camera_pose_end_percent"));
    }

    TEST(CameraPoseActivationTest, SharedThreeDgsFeaturesRemainCompatible) {
        using namespace lfs::core::param;
        auto params = OptimizationParameters::mcmc_defaults();
        params.refine_camera_poses = true;
        params.mip_filter = true;
        params.use_depth_loss = true;
        params.use_normal_loss = true;
        params.mask_mode = MaskMode::SegmentAndIgnore;
        params.use_ppisp = true;
        params.use_bilateral_grid = true;
        params.enable_sparsity = true;
        EXPECT_TRUE(params.camera_pose_incompatibility().empty());
        EXPECT_TRUE(params.validate().empty());
    }

    class CameraPoseJointIntegrationTest : public CameraPosePhotometricTest {
    protected:
        std::unique_ptr<PoseRefinementSession> joint_session(bool combined = false) {
            auto far = identity_transform();
            far[3] = 1;
            PoseSessionConfig config;
            config.total_iterations = 100;
            config.warmup_iterations = 0;
            config.choose_anchors = false;
            if (combined)
                config.joint_reprojection_weight = PoseSessionConfig::DEFAULT_JOINT_REPROJECTION_WEIGHT;
            config.optimizer.scene_scale = scene->get_scene_scale();
            const std::vector<PoseCameraInput> inputs{
                {camera->uid(), identity_transform(), PoseRole::Train},
                {camera->uid() + 1, identity_transform(), PoseRole::Anchor},
                {camera->uid() + 2, far, PoseRole::Anchor}};
            std::vector<SparseTrackMeasurement> measurements;
            for (int i = 0; i < 20; ++i) {
                const SparsePointPosition truth{-0.8 + (i % 5) * 0.4, -0.6 + (i / 5) * 0.4, 4.0 + (i % 3) * 0.2};
                const SparsePointPosition source{truth[0] + 0.002, truth[1] - 0.001, truth[2] + 0.003};
                for (const auto& input : inputs) {
                    measurements.push_back({static_cast<std::uint64_t>(i), source, input.uid, true, input.source, {55, 55, WIDTH / 2.0, HEIGHT / 2.0, WIDTH, HEIGHT}, 55 * (truth[0] + input.source[3]) / truth[2] + WIDTH / 2.0, 55 * truth[1] / truth[2] + HEIGHT / 2.0});
                }
            }
            auto session = std::make_unique<PoseRefinementSession>(1, inputs, config);
            std::vector<Camera::SfmObservation> observations;
            for (const auto& m : measurements)
                if (m.camera_uid == camera->uid())
                    observations.push_back({static_cast<float>(m.u), static_cast<float>(m.v),
                                            static_cast<float>(m.source[0]), static_cast<float>(m.source[1]),
                                            static_cast<float>(m.source[2]), m.point_id});
            camera->set_sfm_observations(std::move(observations));
            session->configure_sparse_points(std::move(measurements));
            return session;
        }
    };

    TEST_F(CameraPoseJointIntegrationTest, ProductionEvaluatorCommitsSharedPointsWithoutChangingGaussians) {
        auto session = joint_session();
        ASSERT_TRUE(session->joint_geometry_enabled(camera->uid()));
        const auto before = session->save_state();
        auto means = scene->means().clone();
        const auto camera_source = camera->world_view_transform().clone();
        const PoseObjective objective = [](const RenderOutput& output, bool gradients) {
            return PoseObjectiveResult{1.0, gradients ? Tensor::zeros_like(output.image) : Tensor{}, {}};
        };
        FastGSPoseEvaluator evaluator(*camera, *scene, *optimizer, background,
                                      objective, {}, false, session.get());
        auto candidate = identity_transform();
        candidate[3] = 0.02f;
        FastGSPoseEvaluator fixed(*camera, *scene, *optimizer, background, objective);
        EXPECT_FALSE(fixed.allows(candidate));
        EXPECT_FALSE(std::isfinite(fixed.loss(candidate)));
        EXPECT_TRUE(evaluator.allows(candidate));
        EXPECT_TRUE(std::isfinite(evaluator.loss(candidate)));
        const auto result = evaluator.visit(*session, 5, 1);
        EXPECT_TRUE(result.scheduled);
        EXPECT_EQ(result.accepted_steps, 0);
        EXPECT_NE(session->save_state().at("points"), before.at("points"));
        expect_bytes_equal(scene->means(), means);
        expect_bytes_equal(camera->world_view_transform(), camera_source);
    }

    TEST_F(CameraPoseJointIntegrationTest, SharedGeometrySurvivesCheckpointEnvelopeAndModelLoad) {
        auto session = joint_session();
        (void)session->visit(camera->uid(), 5, 1, [](const Matrix4&) { return PoseImageEvaluation{1.0, {}}; }, [](const Matrix4&) { return 1.0; });
        lfs::core::param::TrainingParameters params;
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 100;
        params.optimization.max_cap = 64;
        params.camera_pose_state_json = session->save_state().dump();
        auto source_model = scene->clone();
        MCMC strategy(source_model);
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        const auto written = serialize_checkpoint(stream, 5, strategy, params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(written.has_value()) << lfs::format_for_developer(written.error());
        stream.seekg(0);
        const auto metadata = load_checkpoint_params(stream, written->bytes);
        ASSERT_TRUE(metadata.has_value()) << metadata.error();
        EXPECT_EQ(nlohmann::json::parse(metadata->camera_pose_state_json), session->save_state());
        auto loaded_model = scene->clone();
        MCMC loaded_strategy(loaded_model);
        lfs::core::param::TrainingParameters loaded_params;
        stream.clear();
        stream.seekg(0);
        const auto loaded = load_checkpoint(stream, written->bytes, loaded_strategy, loaded_params,
                                            nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        auto resumed = joint_session();
        resumed->restore_state(nlohmann::json::parse(loaded_params.camera_pose_state_json));
        EXPECT_EQ(resumed->save_state().at("points"), session->save_state().at("points"));
        EXPECT_EQ(resumed->current_pose(camera->uid()), session->current_pose(camera->uid()));
        expect_bytes_equal(loaded_model.means(), source_model.means());
        auto bad = nlohmann::json::parse(loaded_params.camera_pose_state_json);
        bad.erase("points");
        loaded_params.camera_pose_state_json = bad.dump();
        EXPECT_FALSE(validate_checkpoint_pose_state(written->header, loaded_params).has_value());
    }

    class CameraPoseCombinedIntegrationTest : public CameraPoseJointIntegrationTest {};

    TEST_F(CameraPoseCombinedIntegrationTest, ProductionObjectiveAndVersionThreeCheckpointRoundTrip) {
        auto session = joint_session(true);
        const auto target = Tensor::from_vector(render(identity_transform()), {3, HEIGHT, WIDTH}, Device::CUDA);
        auto means = scene->means().clone();
        // Record the unrectified value so an identical-image SSIM roundoff
        // failure is distinguishable from NaN or a genuinely invalid objective.
        auto rendered = forward(identity_transform());
        losses::PhotometricLoss raw_loss;
        const auto raw = raw_loss.forward(rendered.first.image, target, {0.2f});
        ASSERT_TRUE(raw.has_value()) << raw.error();
        const float raw_value = raw->first.item<float>();
        std::ostringstream raw_text;
        raw_text << std::scientific << raw_value;
        RecordProperty("identical_image_raw_loss", raw_text.str());
        ASSERT_TRUE(std::isfinite(raw_value));
        ASSERT_GE(raw_value, -32 * std::numeric_limits<float>::epsilon() * 0.2f);
        auto objective = make_pose_photometric_objective(target, 0.2f);
        const auto checked = objective(rendered.first, true);
        EXPECT_GE(checked.loss, 0);
        EXPECT_DOUBLE_EQ(checked.loss, objective(rendered.first, false).loss);
        if (raw_value < 0) {
            EXPECT_DOUBLE_EQ(checked.loss, 0);
            for (const float value : values(checked.grad_image))
                EXPECT_EQ(value, 0);
        }
        rendered.second.release_forward_context();
        FastGSPoseEvaluator evaluator(*camera, *scene, *optimizer, background,
                                      objective, {}, false, session.get());
        const auto result = evaluator.visit(*session, 5, 1);
        EXPECT_TRUE(result.scheduled);
        EXPECT_EQ(session->diagnostics().point_solves, session->shared_point_count());
        expect_bytes_equal(scene->means(), means);
        ASSERT_EQ(session->save_state()["version"], 3);
        lfs::core::param::TrainingParameters params;
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 100;
        params.optimization.max_cap = 64;
        params.camera_pose_state_json = session->save_state().dump();
        auto source_model = scene->clone();
        MCMC strategy(source_model);
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        const auto written = serialize_checkpoint(stream, 5, strategy, params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(written.has_value()) << lfs::format_for_developer(written.error());
        auto loaded_model = scene->clone();
        MCMC loaded_strategy(loaded_model);
        lfs::core::param::TrainingParameters loaded_params;
        stream.seekg(0);
        const auto loaded = load_checkpoint(stream, written->bytes, loaded_strategy, loaded_params,
                                            nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        auto resumed = joint_session(true);
        const auto state = nlohmann::json::parse(loaded_params.camera_pose_state_json);
        resumed->restore_state(state);
        EXPECT_EQ(resumed->save_state()["points"], session->save_state()["points"]);
        EXPECT_EQ(resumed->current_pose(camera->uid()), session->current_pose(camera->uid()));
        expect_bytes_equal(loaded_model.means(), source_model.means());
        for (const auto& invalid_weight : {nlohmann::json(nullptr), nlohmann::json(0), nlohmann::json(-0.1), nlohmann::json("0.0001")}) {
            auto bad = state;
            bad["settings"]["joint_reprojection_weight"] = invalid_weight;
            loaded_params.camera_pose_state_json = bad.dump();
            EXPECT_FALSE(validate_checkpoint_pose_state(written->header, loaded_params).has_value());
        }
    }

    class CameraPoseTrainerIntegrationTest : public CameraPosePhotometricTest {};

    TEST_F(CameraPoseTrainerIntegrationTest, PhotometricObjectiveMatchesTrainerLoss) {
        const auto pose = exp_se3({0.03f, -0.02f, 0.04f, 0.01f, 0.02f, -0.01f});
        const auto target = Tensor::from_vector(render(identity_transform()), {3, HEIGHT, WIDTH}, Device::CUDA);
        for (const float weight : {0.0f, 0.2f, 1.0f}) {
            SCOPED_TRACE(weight);
            losses::PhotometricLoss trainer_loss;
            auto rendered = forward(pose);
            auto reference = trainer_loss.forward(rendered.first.image, target, {weight});
            ASSERT_TRUE(reference.has_value()) << reference.error();
            const double expected_loss = reference->first.item<float>();
            const auto upstream = values(reference->second.grad_image);
            rendered.second.release_forward_context();
            const auto expected_gradient = gradient(pose, upstream);
            FastGSPoseEvaluator evaluator(*camera, *scene, *optimizer, background,
                                          make_pose_photometric_objective(target, weight));
            const auto image = evaluator.evaluate(pose);
            EXPECT_NEAR(image.loss, expected_loss, 1e-7);
            EXPECT_NEAR(evaluator.loss(pose), expected_loss, 1e-7);
            for (int axis = 0; axis < 6; ++axis)
                EXPECT_NEAR(image.gradient[axis], expected_gradient[axis], 1e-6 + std::abs(expected_gradient[axis]) * 1e-4);
        }
        EXPECT_THROW((void)make_pose_photometric_objective(target, -0.1f), std::invalid_argument);
    }

    TEST_F(CameraPoseTrainerIntegrationTest, CheckpointRestoresPoseStateWithModel) {
        auto far = identity_transform();
        far[3] = 1;
        PoseSessionConfig config;
        config.total_iterations = 100;
        config.warmup_iterations = 0;
        config.choose_anchors = false;
        config.optimizer.scene_scale = scene->get_scene_scale();
        const std::vector<PoseCameraInput> inputs{
            {camera->uid(), identity_transform(), PoseRole::Train},
            {camera->uid() + 1, identity_transform(), PoseRole::Anchor},
            {camera->uid() + 2, far, PoseRole::Anchor}};
        PoseRefinementSession session(1, inputs, config);
        ASSERT_GT(session.visit(camera->uid(), 5, 1, [](const Matrix4& pose) {
            const float residual = pose[3] - 0.02f;
            return PoseImageEvaluation{residual * residual, {2*residual,0,0,0,0,0}}; }, [](const Matrix4& pose) { const double residual = pose[3]-0.02; return residual*residual; }).accepted_steps, 0);
        lfs::core::param::TrainingParameters params;
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 100;
        params.optimization.max_cap = 64;
        params.camera_pose_state_json = session.save_state().dump();
        auto source_model = scene->clone();
        ASSERT_EQ(source_model.opacity_raw().dtype(), DataType::Float32);
        ASSERT_EQ(source_model.opacity_raw().ndim(), 2);
        ASSERT_EQ(source_model.opacity_raw().shape()[0], source_model.size());
        ASSERT_EQ(source_model.opacity_raw().shape()[1], 1);
        MCMC source_strategy(source_model);
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        const auto written = serialize_checkpoint(stream, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(written.has_value()) << lfs::format_for_developer(written.error());
        EXPECT_TRUE(has_flag(written->header.flags, CheckpointFlags::HAS_CAMERA_POSES));
        stream.seekg(0);
        const auto metadata = load_checkpoint_params(stream, written->bytes);
        ASSERT_TRUE(metadata.has_value()) << metadata.error();
        EXPECT_EQ(nlohmann::json::parse(metadata->camera_pose_state_json), session.save_state());
        auto loaded_model = scene->clone();
        MCMC loaded_strategy(loaded_model);
        lfs::core::param::TrainingParameters loaded_params;
        stream.clear();
        stream.seekg(0);
        const auto restored = load_checkpoint(stream, written->bytes, loaded_strategy, loaded_params,
                                              nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(restored.has_value()) << restored.error();
        EXPECT_EQ(*restored, 5);
        const auto state = nlohmann::json::parse(loaded_params.camera_pose_state_json);
        PoseRefinementSession resumed(2, inputs, pose_session_config_from_state(state));
        resumed.restore_state(state);
        EXPECT_EQ(resumed.current_pose(camera->uid()), session.current_pose(camera->uid()));
        expect_bytes_equal(loaded_model.means(), source_model.means());
        expect_bytes_equal(loaded_model.opacity_raw(), source_model.opacity_raw());
        loaded_params.optimization.max_cap = 17;
        bool context_checked = false;
        stream.clear();
        stream.seekg(0);
        const auto rejected = load_checkpoint(stream, written->bytes, loaded_strategy, loaded_params,
                                              nullptr, nullptr, nullptr, nullptr, {}, "pose context rejection", nullptr,
                                              [&](const lfs::core::param::TrainingParameters&, const SplatData&) -> lfs::Status {
                                                  context_checked = true;
                                                  return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                                                      .code = lfs::ErrorCode::FailedPrecondition,
                                                      .domain = lfs::ErrorDomain::Training,
                                                      .user_message = "Dataset source poses do not match",
                                                      .detail = "Dataset source poses do not match",
                                                      .detection = LFS_SOURCE_SITE_CURRENT(),
                                                  }));
                                              });
        EXPECT_TRUE(context_checked);
        EXPECT_FALSE(rejected.has_value());
        EXPECT_EQ(loaded_params.optimization.max_cap, 17);
        expect_bytes_equal(loaded_model.means(), source_model.means());
        expect_bytes_equal(loaded_model.opacity_raw(), source_model.opacity_raw());
        auto header = written->header;
        header.flags = CheckpointFlags::NONE;
        EXPECT_FALSE(validate_checkpoint_pose_state(header, loaded_params).has_value());
        header = written->header;
        ++header.iteration;
        EXPECT_FALSE(validate_checkpoint_pose_state(header, loaded_params).has_value());
        loaded_params.camera_pose_state_json.clear();
        EXPECT_FALSE(validate_checkpoint_pose_state(written->header, loaded_params).has_value());
        loaded_params.camera_pose_state_json = params.camera_pose_state_json;
        // Pose-free checkpoints still require valid optimization parameters.
        // Cover both the current envelope and the legacy flat parameter form.
        const auto optimization_json = params.optimization.to_json();
        const nlohmann::json pose_free_checkpoint{{"optimization", optimization_json},
                                                  {"dataset", params.dataset.to_json()}};
        for (const auto& payload : {pose_free_checkpoint, optimization_json}) {
            SCOPED_TRACE(payload.contains("optimization") ? "pose-free checkpoint" : "legacy flat parameters");
            const auto plain = parse_checkpoint_params_json(payload.dump(), loaded_params);
            EXPECT_TRUE(plain.camera_pose_state_json.empty());
            EXPECT_EQ(plain.optimization.iterations, params.optimization.iterations);
            EXPECT_EQ(plain.optimization.max_cap, params.optimization.max_cap);
        }
    }

    TEST_F(CameraPoseTrainerIntegrationTest, UnsupportedTrainingCombinationsAreExplicit) {
        lfs::core::param::OptimizationParameters params;
        EXPECT_TRUE(trainer_pose_incompatibility(params).empty());
        for (const int option : {0}) {
            auto unsupported = params;
            switch (option) {
            case 0: unsupported.gut = true; break;
            case 1: unsupported.mip_filter = true; break;
            case 2: unsupported.use_depth_loss = true; break;
            case 3: unsupported.use_normal_loss = true; break;
            case 4: unsupported.use_ppisp = true; break;
            case 5: unsupported.use_bilateral_grid = true; break;
            case 6: unsupported.use_exposure_correction = true; break;
            case 7: unsupported.enable_sparsity = true; break;
            case 8: unsupported.mask_mode = lfs::core::param::MaskMode::Ignore; break;
            case 9: unsupported.ppisp_use_controller = true; break;
            }
            EXPECT_FALSE(trainer_pose_incompatibility(unsupported).empty());
        }
    }
} // namespace
