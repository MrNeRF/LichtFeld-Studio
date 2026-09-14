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

    TEST(CameraPoseSessionTest, GeometricProposalReachesControllerWithoutChangingImageObjective) {
        PoseRefinementSession session(7, cameras(), config());
        const auto result = session.visit(30, 10, 1, [](const Matrix4& pose) {
            const double residual = pose[3] - 0.002;
            return PoseImageEvaluation{residual * residual,
                                       {static_cast<float>(2 * residual), 0, 0, 0, 0, 0},
                                       Twist{static_cast<float>(-residual), 0, 0, 0, 0, 0}}; }, [](const Matrix4& pose) {
            const double residual = pose[3] - 0.002;
            return residual * residual; });
        EXPECT_GE(result.accepted_steps, 1);
        EXPECT_NEAR(session.current_pose(30)[3], 0.002, 1e-8);
        EXPECT_EQ(session.current_pose(10), identity_transform());
    }

    TEST(CameraPoseSessionTest, GeometricRejectionDoesNotRenderOrMoveAndExceptionsRollBack) {
        PoseRefinementSession session(7, cameras(), config());
        int renders = 0;
        int checks = 0;
        const auto result = session.visit(30, 10, 1, evaluate, [&](const Matrix4& pose) { ++renders; return loss(pose); }, {}, [&](const Matrix4&) { ++checks; return false; });
        EXPECT_TRUE(result.scheduled);
        EXPECT_GT(checks, 0);
        EXPECT_EQ(renders, 0);
        EXPECT_EQ(result.candidate_renders, 0);
        EXPECT_EQ(result.accepted_steps, 0);
        EXPECT_EQ(session.current_pose(30), identity_transform());
        const auto saved = session.save_state();
        PoseRefinementSession restored(8, cameras(), config());
        EXPECT_NO_THROW(restored.restore_state(saved));
        EXPECT_EQ(restored.current_pose(30), identity_transform());

        PoseRefinementSession throwing(9, cameras(), config());
        EXPECT_THROW((void)throwing.visit(30, 10, 1, evaluate, loss, {},
                                          [](const Matrix4&) -> bool { throw std::runtime_error("Constraint failed"); }),
                     std::runtime_error);
        EXPECT_EQ(throwing.current_pose(30), identity_transform());
        // The failed visit has not consumed cadence or changed optimizer state.
        const auto retried = throwing.visit(30, 10, 1, evaluate, loss);
        EXPECT_TRUE(retried.scheduled);
        EXPECT_GT(retried.accepted_steps, 0);
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
    std::vector<SparseTrackMeasurement> shared_measurements() {
        std::vector<SparseTrackMeasurement> result;
        const ReprojectionCalibration k{800, 800, 800, 600, 1600, 1200};
        for (int i = 0; i < 20; ++i) {
            const SparsePointPosition truth{-0.8 + (i % 5) * 0.4, -0.6 + (i / 5) * 0.4, 4.0 + (i % 3) * 0.2};
            const SparsePointPosition source{truth[0] + 0.002, truth[1] - 0.001, truth[2] + 0.003};
            for (const auto& camera : cameras()) {
                if (camera.role == PoseRole::Evaluation)
                    continue;
                result.push_back({static_cast<std::uint64_t>(i), source, camera.uid, true, camera.source, k,
                                  k.fx * (truth[0] + camera.source[3]) / truth[2] + k.cx,
                                  k.fy * truth[1] / truth[2] + k.cy});
            }
        }
        return result;
    }

    TEST(CameraPoseJointGeometryTest, PointOnlyRefinementPersistsAndResetRestoresSource) {
        PoseRefinementSession session(1, cameras(), config());
        const auto measurements = shared_measurements();
        session.configure_sparse_points(measurements);
        ASSERT_EQ(session.shared_point_count(), 20u);
        ASSERT_TRUE(session.joint_geometry_enabled(30));
        const auto before = session.save_state();
        const auto result = session.visit(30, 10, 1, [](const Matrix4&) { return PoseImageEvaluation{1, {}}; }, [](const Matrix4&) { return 1.0; });
        EXPECT_EQ(result.accepted_steps, 0);
        EXPECT_EQ(session.current_pose(30), identity_transform());
        const auto after = session.save_state();
        EXPECT_NE(after.at("points"), before.at("points"));
        const auto tracks = build_sparse_point_tracks(measurements);
        for (size_t i = 0; i < tracks.size(); ++i) {
            const auto position = after["points"][i]["current"].get<SparsePointPosition>();
            EXPECT_LT(sparse_point_cost(tracks[i], position), sparse_point_cost(tracks[i], tracks[i].source));
        }
        PoseRefinementSession restored(2, cameras(), pose_session_config_from_state(after));
        restored.configure_sparse_points(measurements);
        restored.restore_state(after);
        // Restore invalidates stale pose evaluations even when coordinates do
        // not change. Everything else, including shared points, is preserved.
        auto expected_restored = after;
        for (auto& camera : expected_restored["cameras"])
            camera["revision"] = camera["revision"].get<std::uint64_t>() + 1;
        EXPECT_EQ(restored.save_state(), expected_restored);
        restored.reset();
        auto expected_reset = before;
        for (size_t i = 0; i < expected_reset["cameras"].size(); ++i)
            expected_reset["cameras"][i]["revision"] = expected_restored["cameras"][i]["revision"].get<std::uint64_t>() + 1;
        EXPECT_EQ(restored.save_state(), expected_reset);
    }

    TEST(CameraPoseJointGeometryTest, CancellationAndExceptionsDoNotCommitPointsOrPoses) {
        for (const bool cancel : {false, true}) {
            PoseRefinementSession session(1, cameras(), config());
            session.configure_sparse_points(shared_measurements());
            const auto before = session.save_state();
            std::stop_source stop;
            auto callback = [&](const Matrix4&) -> PoseImageEvaluation {
                if (!cancel)
                    throw std::runtime_error("Image evaluation failed");
                stop.request_stop();
                return {1, {}};
            };
            if (cancel) {
                const auto result = session.visit(30, 10, 1, callback, loss, stop.get_token());
                EXPECT_TRUE(result.cancelled);
            } else {
                EXPECT_THROW((void)session.visit(30, 10, 1, callback, loss), std::runtime_error);
            }
            EXPECT_EQ(session.save_state().at("points"), before.at("points"));
            EXPECT_EQ(session.save_state().at("cameras"), before.at("cameras"));
        }
    }

    TEST(CameraPoseJointGeometryTest, AcceptedPoseAndSharedPointsRoundTripTogether) {
        auto measurements = shared_measurements();
        for (auto& m : measurements)
            if (m.camera_uid == 30)
                m.u += m.calibration.fx * 0.002 / (m.source[2] - 0.003);
        PoseRefinementSession session(1, cameras(), config());
        session.configure_sparse_points(measurements);
        const auto before = session.save_state();
        auto objective = [](const Matrix4& pose) {
            const double residual = pose[3] - 0.002;
            return residual * residual;
        };
        const auto result = session.visit(30, 10, 1, [&](const Matrix4& pose) {
            PoseImageEvaluation image{objective(pose), {}};
            image.gradient[0] = static_cast<float>(2 * (pose[3] - 0.002));
            return image; }, objective);
        ASSERT_GT(result.accepted_steps, 0);
        EXPECT_LT(objective(session.current_pose(30)), objective(identity_transform()));
        const auto after = session.save_state();
        EXPECT_NE(after.at("points"), before.at("points"));
        PoseRefinementSession restored(2, cameras(), config());
        restored.configure_sparse_points(measurements);
        restored.restore_state(after);
        auto expected_restored = after;
        for (auto& camera : expected_restored["cameras"])
            camera["revision"] = camera["revision"].get<std::uint64_t>() + 1;
        EXPECT_EQ(restored.save_state(), expected_restored);
        for (const int uid : {10, 20, 90})
            EXPECT_EQ(session.current_pose(uid), restored.current_pose(uid));
    }

    TEST(CameraPoseJointGeometryTest, ChangedMeasurementsRejectRestoreWithoutMutation) {
        PoseRefinementSession source(1, cameras(), config());
        source.configure_sparse_points(shared_measurements());
        auto measurements = shared_measurements();
        measurements.back().u += 0.01;
        PoseRefinementSession target(2, cameras(), config());
        target.configure_sparse_points(measurements);
        const auto before = target.save_state();
        EXPECT_THROW(target.restore_state(source.save_state()), std::invalid_argument);
        EXPECT_EQ(target.save_state(), before);
    }

    TEST(CameraPoseJointGeometryTest, CorruptPointStateCannotPartiallyRestoreSession) {
        PoseRefinementSession session(1, cameras(), config());
        session.configure_sparse_points(shared_measurements());
        const auto before = session.save_state();
        for (int defect = 0; defect < 5; ++defect) {
            auto bad = before;
            bad["paused"] = true;
            auto& point = bad["points"].back();
            if (defect == 0)
                point["current"][0] = 100.0;
            if (defect == 1)
                point["current"][1] = nullptr;
            if (defect == 2)
                point["fingerprint"] = 0;
            if (defect == 3)
                point["source"][2] = 0.0;
            if (defect == 4)
                point["id"] = 0;
            EXPECT_THROW(session.restore_state(bad), std::invalid_argument);
            EXPECT_EQ(session.save_state(), before);
        }
    }

    TEST(CameraPoseJointGeometryTest, LegacyStateCannotSilentlySwitchGeometryModel) {
        PoseRefinementSession legacy(1, cameras(), config());
        PoseRefinementSession joint(2, cameras(), config());
        joint.configure_sparse_points(shared_measurements());
        const auto old_state = legacy.save_state(), joint_state = joint.save_state();
        EXPECT_EQ(old_state.at("version"), 1);
        EXPECT_EQ(joint_state.at("version"), 2);
        EXPECT_THROW(joint.restore_state(old_state), std::invalid_argument);
        EXPECT_THROW(legacy.restore_state(joint_state), std::invalid_argument);
        EXPECT_EQ(legacy.save_state(), old_state);
        EXPECT_EQ(joint.save_state(), joint_state);
    }

    TEST(CameraPoseJointGeometryTest, EvaluationAndDisabledMeasurementsNeverEnterGraph) {
        auto measurements = shared_measurements();
        for (auto& m : measurements)
            if (m.camera_uid == 30)
                m.camera_uid = 90;
        PoseRefinementSession session(1, cameras(), config());
        session.configure_sparse_points(measurements);
        EXPECT_EQ(session.shared_point_count(), 0u);
        EXPECT_FALSE(session.joint_geometry_enabled(30));
        measurements = shared_measurements();
        for (auto& m : measurements)
            if (m.camera_uid == 30)
                m.training = false;
        session.configure_sparse_points(measurements);
        EXPECT_EQ(session.shared_point_count(), 0u);
    }
    TEST(CameraPoseDiagnosticsTest, SeparatesConstraintAndImageRejectionsWithoutChangingCadence) {
        PoseRefinementSession geometric(1, cameras(), config());
        const auto rejected = geometric.visit(30, 10, 1, evaluate, loss, {},
                                              [](const Matrix4&) { return false; });
        EXPECT_EQ(rejected.accepted_steps, 0);
        auto d = geometric.diagnostics();
        EXPECT_EQ(d.visits, 1u);
        EXPECT_GT(d.candidate_checks, 0u);
        EXPECT_EQ(d.fixed_rejections, d.candidate_checks);
        EXPECT_EQ(d.candidate_renders, 0u);
        EXPECT_EQ(d.image_rejections, 0u);
        EXPECT_FALSE(geometric.visit(30, 11, 2, evaluate, loss).scheduled);
        EXPECT_EQ(geometric.diagnostics().visits, 1u);

        PoseRefinementSession image(2, cameras(), config());
        (void)image.visit(30, 10, 1, evaluate, [](const Matrix4&) { return 1.0; });
        d = image.diagnostics();
        EXPECT_GT(d.candidate_renders, 0u);
        EXPECT_EQ(d.image_rejections, d.candidate_renders);
        EXPECT_EQ(d.fixed_rejections + d.joint_rejections + d.invalid_losses + d.objective_rejections, 0u);
        EXPECT_EQ(d.committed_steps, 0u);
        EXPECT_GE(d.visit_ms, d.baseline_ms + d.candidate_ms);
        PoseRefinementSession joint(3, cameras(), config());
        joint.configure_sparse_points(shared_measurements());
        (void)joint.visit(30, 10, 1, evaluate, loss);
        d = joint.diagnostics();
        EXPECT_GT(d.joint_rejections, 0u);
        EXPECT_EQ(d.candidate_checks, d.fixed_rejections + d.joint_rejections + d.candidate_renders);
        EXPECT_EQ(d.candidate_renders, d.invalid_losses + d.image_rejections + d.objective_rejections + d.accepted_candidates);
        EXPECT_EQ(d.committed_steps, d.accepted_candidates);
    }

    TEST(CameraPoseDiagnosticsTest, MeasuresPointWorkAndDoesNotPersistDiagnostics) {
        PoseRefinementSession session(1, cameras(), config());
        session.configure_sparse_points(shared_measurements());
        (void)session.visit(30, 10, 1, [](const Matrix4&) { return PoseImageEvaluation{1.0, {}}; }, [](const Matrix4&) { return 1.0; });
        const auto d = session.diagnostics();
        EXPECT_GE(d.point_solves, session.shared_point_count());
        EXPECT_GT(d.point_proposals, 0u);
        EXPECT_EQ(d.baseline_evaluations, 1u);
        EXPECT_EQ(d.committed_steps, 0u);
        EXPECT_GE(d.visit_ms, d.point_ms);
        const auto saved = session.save_state();
        EXPECT_FALSE(saved.contains("diagnostics"));
        auto corrupt = saved;
        corrupt["points"][0]["current"][0] = 1000;
        EXPECT_THROW(session.restore_state(corrupt), std::invalid_argument);
        EXPECT_EQ(session.diagnostics().point_solves, d.point_solves);
        session.restore_state(saved);
        EXPECT_EQ(session.diagnostics().visits, 0u);
        EXPECT_EQ(session.diagnostics().point_ms, 0.0);
        session.reset();
        EXPECT_EQ(session.diagnostics().point_solves, 0u);
    }

    TEST(CameraPoseDiagnosticsTest, ExceptionsAndCancellationCountWorkWithoutCommitting) {
        for (const bool cancel : {false, true}) {
            PoseRefinementSession session(1, cameras(), config());
            std::stop_source stop;
            auto callback = [&](const Matrix4&) -> PoseImageEvaluation {
                if (!cancel)
                    throw std::runtime_error("Diagnostic exception");
                stop.request_stop();
                return {1, {}};
            };
            if (cancel) {
                EXPECT_TRUE(session.visit(30, 10, 1, callback, loss, stop.get_token()).cancelled);
            } else {
                EXPECT_THROW((void)session.visit(30, 10, 1, callback, loss), std::runtime_error);
            }
            const auto d = session.diagnostics();
            EXPECT_EQ(d.visits, 1u);
            EXPECT_EQ(d.baseline_evaluations, 1u);
            EXPECT_EQ(d.exceptions, cancel ? 0u : 1u);
            EXPECT_EQ(d.cancellations, cancel ? 1u : 0u);
            EXPECT_EQ(d.committed_steps, 0u);
            EXPECT_EQ(session.current_pose(30), identity_transform());
        }
    }
} // namespace
