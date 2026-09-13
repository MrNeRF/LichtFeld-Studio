/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "training/camera_pose/pose_refinement_session.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
    using namespace lfs::training::camera_pose;

    std::vector<PoseCameraInput> cameras() {
        auto far = identity_transform();
        far[3] = 1;
        return {{30, identity_transform(), PoseRole::Train},
                {90, far, PoseRole::Evaluation},
                {20, far, PoseRole::Train},
                {10, identity_transform(), PoseRole::Train}};
    }
    PoseSessionConfig config() {
        PoseSessionConfig result;
        result.total_iterations = 100;
        result.warmup_iterations = 10;
        result.visits_between_updates = 3;
        return result;
    }
    double loss(const Matrix4& pose) {
        const double residual = pose[3] - 0.02;
        return residual * residual;
    }
    PoseImageEvaluation evaluate(const Matrix4& pose) {
        PoseImageEvaluation result{loss(pose), {}};
        result.gradient[0] = 2 * (pose[3] - 0.02f);
        return result;
    }

    TEST(CameraPoseSessionTest, DeterministicAnchorsExcludeEvaluationAndSortByUid) {
        PoseRefinementSession session(7, cameras(), config());
        const auto snapshot = session.published_snapshot();
        ASSERT_EQ(snapshot->cameras.size(), 4u);
        EXPECT_EQ(snapshot->generation, 7u);
        EXPECT_EQ(snapshot->cameras[0].pose.uid, 10);
        EXPECT_EQ(snapshot->cameras[1].pose.uid, 20);
        EXPECT_EQ(snapshot->cameras[2].pose.uid, 30);
        EXPECT_EQ(snapshot->cameras[3].pose.uid, 90);
        EXPECT_EQ(snapshot->cameras[0].state, PoseDisplayState::Anchor);
        EXPECT_EQ(snapshot->cameras[1].state, PoseDisplayState::Anchor);
        EXPECT_EQ(snapshot->cameras[2].state, PoseDisplayState::Waiting);
        EXPECT_EQ(snapshot->cameras[3].state, PoseDisplayState::Evaluation);
        for (const int uid : {10, 20, 90}) {
            const auto before = session.current_pose(uid);
            EXPECT_FALSE(session.visit(uid, 10, 1, {}, {}).scheduled);
            EXPECT_EQ(session.current_pose(uid), before);
        }
    }

    TEST(CameraPoseSessionTest, RejectsAmbiguousMembershipAndDegenerateGauge) {
        auto inputs = cameras();
        inputs.push_back(inputs.front());
        EXPECT_THROW(PoseRefinementSession(1, inputs, config()), std::invalid_argument);
        inputs = cameras();
        for (auto& input : inputs)
            input.source = identity_transform();
        EXPECT_THROW(PoseRefinementSession(1, inputs, config()), std::invalid_argument);
        inputs = cameras();
        inputs[0].role = PoseRole::Evaluation;
        EXPECT_THROW(PoseRefinementSession(1, inputs, config()), std::invalid_argument);
        auto settings = config();
        settings.choose_anchors = false;
        EXPECT_THROW(PoseRefinementSession(1, cameras(), settings), std::invalid_argument);
        EXPECT_THROW(PoseRefinementSession(0, cameras(), config()), std::invalid_argument);
    }

    TEST(CameraPoseSessionTest, ExplicitAnchorsAndIndependentCameraCadence) {
        auto inputs = cameras();
        inputs[2].role = inputs[3].role = PoseRole::Anchor;
        inputs.push_back({40, identity_transform(), PoseRole::Train});
        auto settings = config();
        settings.choose_anchors = false;
        PoseRefinementSession session(1, inputs, settings);
        for (int visit = 0; visit < 4; ++visit) {
            for (const int uid : {30, 40}) {
                const auto result = session.visit(uid, 10 + visit, 1, evaluate, loss);
                EXPECT_EQ(result.scheduled, visit == 0 || visit == 3);
                if (result.scheduled)
                    EXPECT_GT(result.accepted_steps, 0);
                EXPECT_LE(result.candidate_renders, settings.steps_per_visit * settings.optimizer.max_backtracks);
            }
        }
    }

    TEST(CameraPoseSessionTest, WarmupPauseAndFinalFreezeDoNotEvaluate) {
        PoseRefinementSession session(1, cameras(), config());
        EXPECT_EQ(session.published_snapshot()->stop_iteration, 80);
        EXPECT_FALSE(session.visit(30, 9, 1, {}, {}).scheduled);
        session.set_paused(true);
        EXPECT_FALSE(session.visit(30, 10, 1, {}, {}).scheduled);
        EXPECT_TRUE(session.published_snapshot()->paused);
        EXPECT_FALSE(session.published_snapshot()->refinement_finished);
        session.set_paused(false);
        EXPECT_TRUE(session.visit(30, 10, 1, evaluate, loss).scheduled);
        const auto before = session.current_pose(30);
        EXPECT_FALSE(session.visit(30, 80, 2, {}, {}).scheduled);
        // The freeze transition must be visible immediately, even within the
        // normal snapshot throttle interval.
        EXPECT_EQ(session.published_snapshot()->stop_iteration, 80);
        EXPECT_EQ(session.published_snapshot()->cameras[2].state, PoseDisplayState::Frozen);
        EXPECT_TRUE(session.published_snapshot()->refinement_finished);
        EXPECT_EQ(session.current_pose(30), before);
        EXPECT_THROW((void)session.visit(30, 79, 2, {}, {}), std::invalid_argument);
    }

    TEST(CameraPoseSessionTest, PublishedSnapshotsRemainImmutableAcrossUpdateAndReset) {
        PoseRefinementSession session(3, cameras(), config());
        const auto original = session.published_snapshot();
        ASSERT_GT(session.visit(30, 10, 1, evaluate, loss).accepted_steps, 0);
        session.publish(true);
        const auto updated = session.published_snapshot();
        EXPECT_NE(updated->cameras[2].pose.current, original->cameras[2].pose.current);
        EXPECT_EQ(original->cameras[2].pose.current, identity_transform());
        EXPECT_EQ(updated->cameras[2].state, PoseDisplayState::Updated);
        EXPECT_GT(updated->sequence, original->sequence);
        session.reset();
        const auto reset = session.published_snapshot();
        EXPECT_EQ(reset->cameras[2].pose.current, identity_transform());
        EXPECT_EQ(reset->cameras[2].eligible_visits, 0u);
        EXPECT_GT(reset->sequence, updated->sequence);
        EXPECT_GT(reset->cameras[2].pose.revision, updated->cameras[2].pose.revision);
        EXPECT_NE(updated->cameras[2].pose.current, identity_transform());
    }

    TEST(CameraPoseSessionTest, CancellationRollsBackWholeBurstAndPreservesRetryCadence) {
        PoseRefinementSession session(1, cameras(), config());
        std::stop_source stop;
        int evaluations = 0;
        const auto result = session.visit(30, 10, 1, [&](const Matrix4& pose) {
            if (++evaluations == 2) stop.request_stop();
            return evaluate(pose); }, loss, stop.get_token());
        EXPECT_TRUE(result.cancelled);
        EXPECT_EQ(result.accepted_steps, 0);
        EXPECT_GT(result.candidate_renders, 0);
        EXPECT_EQ(session.current_pose(30), identity_transform());
        session.publish(true);
        EXPECT_EQ(session.published_snapshot()->cameras[2].eligible_visits, 0u);
        EXPECT_TRUE(session.visit(30, 10, 1, evaluate, loss).scheduled);
    }

    TEST(CameraPoseSessionTest, EvaluatorExceptionRollsBackPoseHistoryAndCadence) {
        PoseRefinementSession session(1, cameras(), config());
        int evaluations = 0;
        EXPECT_THROW((void)session.visit(30, 10, 1, [&](const Matrix4& pose) {
            if (++evaluations == 2) throw std::runtime_error("renderer failure");
            return evaluate(pose); }, loss), std::runtime_error);
        EXPECT_EQ(session.current_pose(30), identity_transform());
        session.publish(true);
        EXPECT_EQ(session.published_snapshot()->cameras[2].pose.accepted_steps, 0u);
        EXPECT_EQ(session.published_snapshot()->cameras[2].eligible_visits, 0u);
        EXPECT_TRUE(session.visit(30, 10, 1, evaluate, loss).scheduled);
    }

    TEST(CameraPoseSessionTest, RejectionAndZeroGradientDoNotClaimConvergence) {
        PoseRefinementSession session(1, cameras(), config());
        const auto result = session.visit(30, 10, 1, evaluate, [](const Matrix4&) { return 1.0; });
        EXPECT_EQ(result.accepted_steps, 0);
        session.publish(true);
        EXPECT_EQ(session.published_snapshot()->cameras[2].state, PoseDisplayState::Rejected);
        session.reset();
        (void)session.visit(30, 10, 1, [](const Matrix4&) { return PoseImageEvaluation{}; }, loss);
        session.publish(true);
        EXPECT_EQ(session.published_snapshot()->cameras[2].state, PoseDisplayState::Ready);
        EXPECT_EQ(pose_display_state_name(PoseDisplayState::Updated), "updated");
    }
    TEST(CameraPoseSessionTest, DurableStateRoundTripPreservesPosesPauseAndCadence) {
        PoseRefinementSession original(1, cameras(), config());
        ASSERT_GT(original.visit(30, 10, 1, evaluate, loss).accepted_steps, 0);
        original.set_paused(true);
        // Save must use live state, not depend on the 250 ms display publisher.
        auto serialized = nlohmann::json::parse(original.save_state().dump());
        std::reverse(serialized["cameras"].begin(), serialized["cameras"].end());
        PoseRefinementSession restored(99, cameras(), config());
        const auto before = restored.published_snapshot();
        restored.restore_state(serialized);
        const auto after = restored.published_snapshot();
        EXPECT_EQ(after->generation, 99u);
        EXPECT_GT(after->sequence, before->sequence);
        EXPECT_TRUE(after->paused);
        EXPECT_EQ(after->iteration, 10);
        EXPECT_EQ(restored.current_pose(30), original.current_pose(30));
        EXPECT_GT(after->cameras[2].pose.revision, original.published_snapshot()->cameras[2].pose.revision);
        EXPECT_EQ(after->cameras[2].eligible_visits, 1u);
        EXPECT_EQ(before->cameras[2].pose.current, identity_transform());
        EXPECT_FALSE(restored.visit(30, 10, 2, {}, {}).scheduled);
        restored.set_paused(false);
        EXPECT_FALSE(restored.visit(30, 11, 2, evaluate, loss).scheduled);
        EXPECT_FALSE(restored.visit(30, 12, 2, evaluate, loss).scheduled);
        EXPECT_GT(restored.visit(30, 13, 2, evaluate, loss).accepted_steps, 0);
        restored.reset();
        EXPECT_EQ(restored.current_pose(30), identity_transform());
    }

    TEST(CameraPoseSessionTest, CorruptDurableStateNeverPartiallyChangesLiveSession) {
        PoseRefinementSession session(1, cameras(), config());
        ASSERT_GT(session.visit(30, 10, 1, evaluate, loss).accepted_steps, 0);
        const auto saved = session.save_state();
        const auto snapshot = session.published_snapshot();
        using Json = nlohmann::json;
        const std::vector<std::function<void(Json&)>> corruptions{
            [](Json& j) { j["version"] = 2; },
            [](Json& j) { j["settings"]["steps_per_visit"] = 8; },
            [](Json& j) { j["iteration"] = 101; },
            [](Json& j) { j["paused"] = 1; },
            [](Json& j) { j["cameras"][2]["uid"] = 10; },
            [](Json& j) { j["cameras"][2]["role"] = static_cast<int>(PoseRole::Evaluation); },
            [](Json& j) { j["cameras"][2]["source"][3] = 0.01; },
            [](Json& j) { j["cameras"][2]["current"][3] = 10.0; },
            [](Json& j) { j["cameras"][2]["current"][15] = 0; },
            [](Json& j) { j["cameras"][2]["current"][0] = nullptr; },
            [](Json& j) { j["cameras"][2]["eligible_visits"] = -1; },
            [](Json& j) { j["cameras"][2]["eligible_visits"] = 1.5; },
            [](Json& j) { j["cameras"][2]["candidate_renders"] = 0; },
            [](Json& j) { j["cameras"][2]["revision"] = 0; },
            [](Json& j) { j["cameras"][2]["display_state"] = 99; },
            [](Json& j) { j["cameras"][2].erase("current"); },
            [](Json& j) { j["cameras"].erase(0); },
        };
        for (size_t i = 0; i < corruptions.size(); ++i) {
            SCOPED_TRACE(i);
            auto bad = saved;
            corruptions[i](bad);
            EXPECT_ANY_THROW(session.restore_state(bad));
            EXPECT_EQ(session.save_state(), saved);
            EXPECT_EQ(session.published_snapshot(), snapshot);
        }
    }

    TEST(CameraPoseSessionTest, DurableStateCannotMoveAnchorsOrEvaluationCameras) {
        PoseRefinementSession session(1, cameras(), config());
        const auto saved = session.save_state();
        for (const size_t index : {0u, 1u, 3u}) {
            auto bad = saved;
            auto& camera = bad["cameras"][index];
            camera["current"][3] = camera["current"][3].get<float>() + 0.001f;
            camera["accepted_steps"] = camera["revision"] = camera["candidate_renders"] = 1;
            EXPECT_THROW(session.restore_state(bad), std::invalid_argument);
            EXPECT_EQ(session.save_state(), saved);
        }
    }
} // namespace
