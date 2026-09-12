/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/camera_pose/bounded_pose_optimizer.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
    using namespace lfs::training::camera_pose;

    PoseEvaluation baseline(const BoundedPoseOptimizer& optimizer, double target = 0.02,
                            std::uint64_t model_revision = 1) {
        const auto state = optimizer.snapshot();
        const double residual = state.current[3] - target;
        PoseEvaluation evaluation{state.uid, model_revision, state.revision, residual * residual, {}};
        evaluation.image_gradient[0] = static_cast<float>(2 * residual);
        return evaluation;
    }

    TEST(CameraPoseControllerTest, RejectsInvalidSourcesAndConfiguration) {
        auto config = BoundedPoseConfig{};
        config.scene_scale = 0;
        EXPECT_THROW(BoundedPoseOptimizer(1, identity_transform(), config), std::invalid_argument);
        config = {};
        auto source = identity_transform();
        source[0] = -1; // Reflection, not a camera rotation.
        EXPECT_THROW(BoundedPoseOptimizer(1, source, config), std::invalid_argument);
        source = identity_transform();
        source[3] = std::numeric_limits<float>::infinity();
        EXPECT_THROW(BoundedPoseOptimizer(1, source, config), std::invalid_argument);
    }

    TEST(CameraPoseControllerTest, FrozenAnchorAndEvaluationNeverRenderOrMove) {
        for (const auto role : {PoseRole::Anchor, PoseRole::Evaluation, PoseRole::Train}) {
            BoundedPoseOptimizer optimizer(7, identity_transform(), {}, role);
            if (role == PoseRole::Train)
                optimizer.set_frozen(true);
            int calls = 0;
            const auto result = optimizer.step(baseline(optimizer), [&](const Matrix4&) { ++calls; return 0.0; });
            EXPECT_EQ(result.status, PoseStepStatus::Frozen);
            EXPECT_EQ(calls, 0);
            EXPECT_EQ(optimizer.snapshot().current, identity_transform());
            EXPECT_EQ(optimizer.snapshot().accepted_steps, 0u);
        }
    }

    TEST(CameraPoseControllerTest, StaleAndInvalidGradientsAreRejectedBeforeRendering) {
        BoundedPoseOptimizer optimizer(7, identity_transform(), {});
        auto old = baseline(optimizer);
        optimizer.reset();
        const auto never = [](const Matrix4&) { ADD_FAILURE() << "Unexpected candidate render"; return 0.0; };
        EXPECT_EQ(optimizer.step(old, never).status, PoseStepStatus::Stale);
        auto wrong_camera = baseline(optimizer);
        wrong_camera.uid = 123;
        EXPECT_EQ(optimizer.step(wrong_camera, never).status, PoseStepStatus::Stale);
        auto invalid = baseline(optimizer);
        invalid.image_gradient[4] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_EQ(optimizer.step(invalid, never).status, PoseStepStatus::InvalidInput);
        EXPECT_EQ(optimizer.snapshot().current, identity_transform());
    }

    TEST(CameraPoseControllerTest, RejectsWorseAndNonfiniteLossWithoutMoving) {
        BoundedPoseOptimizer optimizer(7, identity_transform(), {});
        for (double loss : {1.0, std::numeric_limits<double>::quiet_NaN()}) {
            int calls = 0;
            const auto before = optimizer.snapshot();
            const auto result = optimizer.step(baseline(optimizer), [&](const Matrix4&) { ++calls; return loss; });
            EXPECT_EQ(result.status, PoseStepStatus::Rejected);
            EXPECT_LE(calls, BoundedPoseConfig{}.max_backtracks);
            EXPECT_EQ(optimizer.snapshot().current, before.current);
            EXPECT_EQ(optimizer.snapshot().revision, before.revision);
            EXPECT_EQ(optimizer.snapshot().accepted_steps, 0u);
        }
    }

    TEST(CameraPoseControllerTest, ExceptionRollsBackControllerAndSnapshotsAreValues) {
        BoundedPoseOptimizer optimizer(7, identity_transform(), {});
        const auto before = optimizer.snapshot();
        EXPECT_THROW((void)optimizer.step(baseline(optimizer), [](const Matrix4&) -> double {
            throw std::runtime_error("Renderer failed");
        }),
                     std::runtime_error);
        EXPECT_EQ(optimizer.snapshot().revision, before.revision);
        EXPECT_EQ(optimizer.snapshot().rejected_steps, before.rejected_steps);
        const auto result = optimizer.step(baseline(optimizer), [](const Matrix4& pose) {
            return (pose[3] - 0.02) * (pose[3] - 0.02);
        });
        EXPECT_EQ(result.status, PoseStepStatus::Accepted);
        EXPECT_EQ(before.current, identity_transform());
        EXPECT_EQ(optimizer.snapshot().source, before.source);
        EXPECT_NE(optimizer.snapshot().current, before.current);
    }

    TEST(CameraPoseControllerTest, CumulativeBoundsSurviveRepeatedStepsAndReset) {
        BoundedPoseConfig config;
        config.scene_scale = 2;
        config.max_center_fraction = 0.01;
        BoundedPoseOptimizer optimizer(7, identity_transform(), config);
        for (int iteration = 0; iteration < 100; ++iteration) {
            const auto before = optimizer.snapshot();
            // New model revision discards stale curvature but preserves bounds.
            const auto result = optimizer.step(baseline(optimizer, 1.0, iteration), [](const Matrix4& pose) {
                return (pose[3] - 1.0) * (pose[3] - 1.0);
            });
            EXPECT_LE(result.evaluations, config.max_backtracks);
            const auto after = optimizer.snapshot();
            EXPECT_LE(after.center_displacement, config.scene_scale * config.max_center_fraction);
            EXPECT_LE(after.center_displacement - before.center_displacement,
                      config.scene_scale * config.step_center_fraction + 1e-7);
        }
        EXPECT_GT(optimizer.snapshot().accepted_steps, 1u);
        const auto revision = optimizer.snapshot().revision;
        optimizer.reset();
        EXPECT_GT(optimizer.snapshot().revision, revision);
        EXPECT_EQ(optimizer.snapshot().current, identity_transform());
        EXPECT_EQ(optimizer.snapshot().accepted_steps, 0u);
    }

    TEST(CameraPoseControllerTest, ChangingModelInvalidatesCurvatureHistory) {
        BoundedPoseOptimizer optimizer(7, identity_transform(), {});
        const auto result = optimizer.step(baseline(optimizer), [](const Matrix4& pose) {
            return (pose[3] - 0.02) * (pose[3] - 0.02);
        });
        ASSERT_EQ(result.status, PoseStepStatus::Accepted);
        auto fresh_history = optimizer;
        fresh_history.set_frozen(false); // Explicitly discard curvature.
        const auto loss = [](const Matrix4& pose) { return (pose[3] + 0.02) * (pose[3] + 0.02); };
        const auto a = optimizer.step(baseline(optimizer, -0.02, 2), loss);
        const auto b = fresh_history.step(baseline(fresh_history, -0.02, 2), loss);
        EXPECT_EQ(a.status, b.status);
        EXPECT_EQ(optimizer.snapshot().current, fresh_history.snapshot().current);
        const auto stale = baseline(optimizer, 0.02, 1);
        EXPECT_EQ(optimizer.step(stale, loss).status, PoseStepStatus::Stale);
        optimizer.reset();
        const auto after_reset = optimizer.step(baseline(optimizer, -0.02, 1), loss);
        EXPECT_EQ(after_reset.status, PoseStepStatus::Accepted);
    }
    TEST(CameraPoseControllerTest, CumulativeRotationBoundIsIndependentOfTranslation) {
        BoundedPoseConfig config;
        config.max_rotation_radians = 0.01;
        const auto target = exp_se3({0, 0, 0, 0, 0, 0.3f});
        const auto loss = [&](const Matrix4& pose) {
            double value = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    const double delta = static_cast<double>(pose[4 * i + j]) - target[4 * i + j];
                    value += 0.5 * delta * delta;
                }
            return value;
        };
        BoundedPoseOptimizer optimizer(7, identity_transform(), config);
        for (int iteration = 0; iteration < 100; ++iteration) {
            const auto state = optimizer.snapshot();
            Matrix4 matrix_gradient{};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    matrix_gradient[4 * i + j] = state.current[4 * i + j] - target[4 * i + j];
            const PoseEvaluation evaluation{state.uid, 1, state.revision, loss(state.current),
                                            left_increment_gradient(state.current, matrix_gradient)};
            (void)optimizer.step(evaluation, loss);
            EXPECT_LE(optimizer.snapshot().rotation_displacement, config.max_rotation_radians);
            EXPECT_NEAR(optimizer.snapshot().center_displacement, 0.0, 1e-12);
        }
        EXPECT_GT(optimizer.snapshot().accepted_steps, 1u);
    }
} // namespace
