/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/camera_pose/bounded_pose_optimizer.hpp"
#include "training/camera_pose/joint_pose_proposal.hpp"
#include "training/camera_pose/sparse_point_refinement.hpp"
#include "training/camera_pose/sparse_reprojection_guard.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
    using namespace lfs::training::camera_pose;

    constexpr ReprojectionCalibration calibration{800, 780, 800, 600, 1600, 1200};

    TEST(CameraPoseAdaptiveTest, BiasCorrectionPersistsAcrossModelRevisionsAndRestore) {
        BoundedPoseConfig config;
        config.center_prior = config.rotation_prior = 0;
        BoundedPoseOptimizer optimizer(30, identity_transform(), config);
        auto advance = [](BoundedPoseOptimizer& controller, float gradient, std::uint64_t model) {
            const auto pose = controller.snapshot();
            const auto loss = [gradient](const Matrix4& p) { return 1.0 + gradient * p[3]; };
            PoseEvaluation evaluation{30, model, pose.revision, loss(pose.current), {gradient, 0, 0, 0, 0, 0}};
            evaluation.adaptive = true;
            return controller.step(evaluation, loss);
        };
        ASSERT_EQ(advance(optimizer, 1, 1).status, PoseStepStatus::Accepted);
        auto first = optimizer.snapshot();
        EXPECT_NEAR(first.current[3], -1e-5, 1e-11);
        EXPECT_DOUBLE_EQ(first.first_moment[0], 0.1);
        EXPECT_DOUBLE_EQ(first.second_moment[0], 0.001);
        EXPECT_EQ(first.adaptive_steps, 1u);
        ASSERT_EQ(advance(optimizer, 2, 2).status, PoseStepStatus::Accepted);
        const auto second = optimizer.snapshot();
        const double expected = -1e-5 * (0.29 / (1 - 0.9 * 0.9)) /
                                (std::sqrt(0.004999 / (1 - 0.999 * 0.999)) + 1e-8);
        EXPECT_NEAR(second.current[3] - first.current[3], expected, 1e-11);
        EXPECT_NEAR(second.first_moment[0], 0.29, 1e-15);
        EXPECT_NEAR(second.second_moment[0], 0.004999, 1e-15);
        EXPECT_EQ(second.adaptive_steps, 2u);
        BoundedPoseOptimizer resumed(30, identity_transform(), config);
        resumed.restore(second);
        ASSERT_EQ(advance(optimizer, 3, 3).status, PoseStepStatus::Accepted);
        ASSERT_EQ(advance(resumed, 3, 3).status, PoseStepStatus::Accepted);
        EXPECT_EQ(resumed.snapshot().current, optimizer.snapshot().current);
        EXPECT_EQ(resumed.snapshot().first_moment, optimizer.snapshot().first_moment);
        EXPECT_EQ(resumed.snapshot().second_moment, optimizer.snapshot().second_moment);
        EXPECT_EQ(resumed.snapshot().adaptive_steps, optimizer.snapshot().adaptive_steps);
    }

    TEST(CameraPoseAdaptiveTest, ExceptionsAndInvalidRestoreAreAtomicAndResetClearsMoments) {
        BoundedPoseOptimizer optimizer(30, identity_transform(), {});
        PoseEvaluation evaluation{30, 1, 0, 1, {1, 0, 0, 0, 0, 0}};
        evaluation.adaptive = true;
        EXPECT_THROW((void)optimizer.step(evaluation, [](const Matrix4&) -> double {
            throw std::runtime_error("cancelled evaluation");
        }),
                     std::runtime_error);
        EXPECT_EQ(optimizer.snapshot().adaptive_steps, 0u);
        EXPECT_EQ(optimizer.snapshot().current, identity_transform());
        ASSERT_EQ(optimizer.step(evaluation, [](const Matrix4& p) { return 1.0 + p[3]; }).status,
                  PoseStepStatus::Accepted);
        const auto valid = optimizer.snapshot();
        auto invalid = valid;
        invalid.second_moment[0] = -1;
        EXPECT_THROW(optimizer.restore(invalid), std::invalid_argument);
        EXPECT_EQ(optimizer.snapshot().second_moment, valid.second_moment);
        EXPECT_EQ(optimizer.snapshot().revision, valid.revision);
        BoundedPoseOptimizer anchor(30, identity_transform(), {}, PoseRole::Anchor);
        EXPECT_THROW(anchor.restore(valid), std::invalid_argument);
        optimizer.reset();
        EXPECT_EQ(optimizer.snapshot().adaptive_steps, 0u);
        EXPECT_EQ(optimizer.snapshot().first_moment, (std::array<double, 6>{}));
        EXPECT_EQ(optimizer.snapshot().second_moment, (std::array<double, 6>{}));
        EXPECT_EQ(optimizer.snapshot().current, identity_transform());
    }

    TEST(CameraPoseAdaptiveTest, OrdinaryControllerKeepsOriginalStepAndNoAdaptiveState) {
        BoundedPoseOptimizer optimizer(30, identity_transform(), {});
        const PoseEvaluation evaluation{30, 1, 0, 1, {1, 0, 0, 0, 0, 0}};
        ASSERT_EQ(optimizer.step(evaluation, [](const Matrix4& p) { return 1.0 + p[3]; }).status,
                  PoseStepStatus::Accepted);
        EXPECT_NEAR(optimizer.snapshot().current[3], -0.003, 1e-9);
        EXPECT_EQ(optimizer.snapshot().adaptive_steps, 0u);
        EXPECT_EQ(optimizer.snapshot().first_moment, (std::array<double, 6>{}));
        EXPECT_EQ(optimizer.snapshot().second_moment, (std::array<double, 6>{}));
    }

    std::vector<SparsePointTrack> joint_tracks(double scale = 1.0) {
        std::vector<SparsePointTrack> result;
        for (int i = 0; i < 20; ++i) {
            SparsePointPosition truth{(i % 5 - 2) * 0.4 * scale,
                                      (i / 5 - 1.5) * 0.4 * scale, (3.5 + (i % 3) * 0.5) * scale};
            SparsePointTrack track{static_cast<std::uint64_t>(i), truth, {}};
            track.source[2] += 0.01 * scale;
            for (int camera = 0; camera < 3; ++camera) {
                auto p = identity_transform();
                p[3] = static_cast<float>((camera - 1) * scale);
                p[7] = static_cast<float>((camera % 2) * 0.3 * scale);
                track.measurements.push_back({track.point_id, track.source, camera, true, p, calibration,
                                              calibration.fx * (truth[0] + p[3]) / truth[2] + calibration.cx,
                                              calibration.fy * (truth[1] + p[7]) / truth[2] + calibration.cy});
            }
            result.push_back(track);
        }
        return result;
    }

    TEST(CameraPoseSchurTest, RecoversPoseWithMovableStructureWithoutMutatingSources) {
        auto tracks = joint_tracks();
        auto pose = apply_left_increment({0.008f, -0.004f, 0.003f, 0.001f, -0.001f, 0.002f},
                                         tracks.front().measurements[1].pose);
        std::vector<SparsePointPosition> points;
        for (const auto& track : tracks)
            points.push_back(track.source);
        const auto original_points = points;
        const auto fingerprint = sparse_track_fingerprint(tracks.front());
        auto cost = [&](const Matrix4& p) {
            double value = 0;
            for (size_t i = 0; i < tracks.size(); ++i) {
                auto track = tracks[i];
                track.measurements[1].pose = p;
                value += sparse_point_cost(track, points[i]);
            }
            return value;
        };
        const double before = cost(pose);
        const auto step = propose_joint_pose(1, pose, tracks, points, 1.0);
        ASSERT_TRUE(step);
        EXPECT_EQ(points, original_points);
        EXPECT_EQ(sparse_track_fingerprint(tracks.front()), fingerprint);
        const auto candidate = apply_left_increment(*step, pose);
        for (size_t i = 0; i < tracks.size(); ++i) {
            auto track = tracks[i];
            track.measurements[1].pose = candidate;
            if (auto update = propose_sparse_point(track, 0.1, 2.0, points[i]))
                points[i] = update->position;
        }
        EXPECT_LT(cost(candidate), before * 0.05);
        EXPECT_NEAR(candidate[3], 0, 0.001);
        EXPECT_NEAR(candidate[7], 0.3, 0.001);
        EXPECT_NEAR(candidate[11], 0, 0.001);
    }

    TEST(CameraPoseSchurTest, ProposalIsInvariantToWorldUnits) {
        std::optional<Twist> reference;
        std::optional<Twist> combined_reference;
        for (const double scale : {1.0, 100.0}) {
            auto tracks = joint_tracks(scale);
            std::vector<SparsePointPosition> points;
            for (const auto& track : tracks)
                points.push_back(track.source);
            const auto pose = apply_left_increment({static_cast<float>(0.005 * scale), 0, 0, 0, 0.001f, 0},
                                                   tracks.front().measurements[1].pose);
            auto step = propose_joint_pose(1, pose, tracks, points, scale);
            ASSERT_TRUE(step);
            for (int i = 0; i < 3; ++i)
                (*step)[i] = static_cast<float>((*step)[i] / scale);
            if (reference)
                for (int i = 0; i < 6; ++i)
                    EXPECT_NEAR((*step)[i], (*reference)[i], 1e-6);
            reference = step;
            const Twist image_gradient{static_cast<float>(0.02 / scale), 0, 0, 0, 0.03f, 0};
            auto combined = propose_joint_pose(1, pose, tracks, points, scale, image_gradient);
            ASSERT_TRUE(combined);
            for (int i = 0; i < 3; ++i)
                (*combined)[i] = static_cast<float>((*combined)[i] / scale);
            if (combined_reference)
                for (int i = 0; i < 6; ++i)
                    EXPECT_NEAR((*combined)[i], (*combined_reference)[i], 1e-6);
            combined_reference = combined;
            auto invalid_gradient = image_gradient;
            invalid_gradient[0] = std::numeric_limits<float>::quiet_NaN();
            EXPECT_FALSE(propose_joint_pose(1, pose, tracks, points, scale, invalid_gradient));
            EXPECT_FALSE(propose_joint_pose(1, pose, tracks, points, scale, image_gradient, -1));
        }
    }

    TEST(CameraPoseSchurTest, InvalidOrUnobservableGeometryNeverProducesProposal) {
        for (int defect = 0; defect < 7; ++defect) {
            auto tracks = joint_tracks();
            std::vector<SparsePointPosition> points;
            for (const auto& track : tracks)
                points.push_back(track.source);
            if (defect == 0)
                points.pop_back();
            if (defect == 1)
                tracks[0].measurements[0].training = false;
            if (defect == 2)
                tracks[0].measurements[0].camera_uid = 1;
            if (defect == 3)
                points[0][2] = std::numeric_limits<double>::quiet_NaN();
            if (defect == 4)
                tracks[0].measurements[0].pose[0] = 2;
            if (defect == 5)
                for (auto& track : tracks)
                    for (auto& m : track.measurements)
                        m.pose = identity_transform();
            if (defect == 6)
                tracks[0].measurements[1].camera_uid = 5;
            EXPECT_FALSE(propose_joint_pose(1, identity_transform(), tracks, points, 1)) << defect;
        }
    }

    TEST(CameraPoseSchurTest, NormalizedFactorSolvesCoupledSystem) {
        const joint_detail::Matrix<3> h{{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}}};
        joint_detail::Factor<3> factor;
        ASSERT_TRUE(factor.compute(h));
        const auto x = factor.solve({0, -5, 7}); // h * (1, -2, 3)
        EXPECT_NEAR(x[0], 1, 1e-12);
        EXPECT_NEAR(x[1], -2, 1e-12);
        EXPECT_NEAR(x[2], 3, 1e-12);
    }

    TEST(CameraPoseSchurTest, ReducedStepMatchesFullNumericalNormalEquations) {
        const auto tracks = joint_tracks();
        const auto pose = apply_left_increment({0.006f, -0.003f, 0.002f, 0.001f, -0.001f, 0.002f},
                                               tracks.front().measurements[1].pose);
        std::vector<SparsePointPosition> points;
        for (const auto& track : tracks)
            points.push_back(track.source);
        // Independent full 66-variable system, using central differences rather
        // than the production analytic Jacobians or point elimination.
        constexpr size_t dimensions = 6 + 3 * 20;
        joint_detail::Matrix<dimensions> h{};
        joint_detail::Vector<dimensions> rhs{};
        constexpr double epsilon = 1e-3;
        for (size_t t = 0; t < tracks.size(); ++t) {
            for (const auto& m : tracks[t].measurements) {
                const auto p = m.camera_uid == 1 ? pose : m.pose;
                auto residual = [&](const Matrix4& camera, const SparsePointPosition& point) {
                    const double x = camera[0] * point[0] + camera[1] * point[1] + camera[2] * point[2] + camera[3];
                    const double y = camera[4] * point[0] + camera[5] * point[1] + camera[6] * point[2] + camera[7];
                    const double z = camera[8] * point[0] + camera[9] * point[1] + camera[10] * point[2] + camera[11];
                    return std::array<double, 2>{(calibration.fx * x / z + calibration.cx - m.u) / 1600,
                                                 (calibration.fy * y / z + calibration.cy - m.v) / 1600};
                };
                const auto r = residual(p, points[t]);
                const double norm = std::hypot(r[0], r[1]);
                const double weight = norm <= 2.0 / 1600 ? 1.0 : 2.0 / (1600 * norm);
                std::array<joint_detail::Vector<dimensions>, 2> jacobian{};
                if (m.camera_uid == 1)
                    for (size_t i = 0; i < 6; ++i) {
                        Twist plus{}, minus{};
                        plus[i] = static_cast<float>(epsilon);
                        minus[i] = -plus[i];
                        const auto a = residual(apply_left_increment(plus, p), points[t]);
                        const auto b = residual(apply_left_increment(minus, p), points[t]);
                        for (size_t row = 0; row < 2; ++row)
                            jacobian[row][i] = (a[row] - b[row]) / (2 * epsilon);
                    }
                for (size_t i = 0; i < 3; ++i) {
                    auto plus = points[t], minus = points[t];
                    plus[i] += epsilon;
                    minus[i] -= epsilon;
                    const auto a = residual(p, plus), b = residual(p, minus);
                    for (size_t row = 0; row < 2; ++row)
                        jacobian[row][6 + 3 * t + i] = (a[row] - b[row]) / (2 * epsilon);
                }
                for (size_t i = 0; i < dimensions; ++i)
                    for (size_t row = 0; row < 2; ++row) {
                        rhs[i] -= weight * jacobian[row][i] * r[row];
                        for (size_t j = 0; j < dimensions; ++j)
                            h[i][j] += weight * jacobian[row][i] * jacobian[row][j];
                    }
            }
        }
        joint_detail::Factor<dimensions> full;
        ASSERT_TRUE(full.compute(h));
        const auto expected = full.solve(rhs);
        const auto actual = propose_joint_pose(1, pose, tracks, points, 1);
        ASSERT_TRUE(actual);
        for (size_t i = 0; i < 6; ++i)
            EXPECT_NEAR((*actual)[i], expected[i], 2e-5) << i;
    }

    SparsePointTrack point_track(const double scale = 1.0) {
        const SparsePointPosition truth{0.3 * scale, 0.2 * scale, 4.0 * scale};
        SparsePointTrack track{123456789012ULL, {0.35 * scale, 0.17 * scale, 4.15 * scale}, {}};
        for (int i = 0; i < 3; ++i) {
            auto pose = identity_transform();
            pose[3] = static_cast<float>((i - 1) * 0.8 * scale);
            pose[7] = static_cast<float>((i % 2) * 0.3 * scale);
            track.measurements.push_back({track.point_id, track.source, i, true, pose, calibration,
                                          calibration.fx * (truth[0] + pose[3]) / truth[2] + calibration.cx,
                                          calibration.fy * (truth[1] + pose[7]) / truth[2] + calibration.cy});
        }
        return track;
    }

    TEST(CameraPoseSparsePointTest, RecoversPointFromMultipleViewsWithoutMutatingInputs) {
        for (const double scale : {1.0, 100.0}) {
            const auto track = point_track(scale);
            const auto source = track.source;
            const auto proposal = propose_sparse_point(track, scale);
            ASSERT_TRUE(proposal.has_value());
            EXPECT_LT(proposal->candidate_cost, proposal->source_cost * 1e-6);
            EXPECT_NEAR(proposal->position[0] / scale, 0.3, 1e-5);
            EXPECT_NEAR(proposal->position[1] / scale, 0.2, 1e-5);
            EXPECT_NEAR(proposal->position[2] / scale, 4.0, 1e-5);
            EXPECT_EQ(track.source, source);
            EXPECT_EQ(track.measurements[0].source, source);
        }
    }

    TEST(CameraPoseSparsePointTest, KeepsTrackIdentityAndExcludesEvaluationMeasurements) {
        auto input = point_track().measurements;
        const auto duplicate_position = input;
        for (auto m : duplicate_position) {
            m.point_id = 0; // Zero is a valid COLMAP point ID, not a missing ID.
            input.push_back(m);
        }
        auto evaluation = input.front();
        evaluation.training = false;
        evaluation.source[2] = -100;
        input.push_back(evaluation);
        const auto tracks = build_sparse_point_tracks(input);
        ASSERT_EQ(tracks.size(), 2u);
        EXPECT_EQ(tracks[0].point_id, 0u);
        EXPECT_EQ(tracks[1].point_id, 123456789012ULL);
        EXPECT_EQ(tracks[0].source, tracks[1].source);
        EXPECT_EQ(tracks[1].measurements.size(), 3u);
        auto insufficient = point_track().measurements;
        insufficient.back().training = false;
        EXPECT_TRUE(build_sparse_point_tracks(insufficient).empty());
    }

    TEST(CameraPoseSparsePointTest, RejectsAmbiguousTracksAndDegenerateGeometry) {
        auto duplicate = point_track().measurements;
        duplicate.push_back(duplicate.front());
        EXPECT_TRUE(build_sparse_point_tracks(duplicate).empty());
        auto inconsistent = point_track().measurements;
        inconsistent.back().source[0] += 0.01;
        EXPECT_TRUE(build_sparse_point_tracks(inconsistent).empty());
        auto missing = point_track().measurements;
        for (auto& m : missing)
            m.point_id = std::numeric_limits<std::uint64_t>::max();
        EXPECT_TRUE(build_sparse_point_tracks(missing).empty());
        auto degenerate = point_track();
        for (auto& m : degenerate.measurements)
            m.pose = identity_transform();
        EXPECT_FALSE(propose_sparse_point(degenerate, 1));
        auto evaluation = point_track();
        evaluation.measurements.back().training = false;
        EXPECT_FALSE(propose_sparse_point(evaluation, 1));
        auto invalid = point_track();
        invalid.measurements.front().pose[0] = 2;
        EXPECT_FALSE(propose_sparse_point(invalid, 1));
    }

    TEST(CameraPoseSparsePointTest, BoundsCumulativePointMovementAndRejectsInvalidInputs) {
        const auto track = point_track();
        const auto proposal = propose_sparse_point(track, 0.005);
        ASSERT_TRUE(proposal.has_value());
        double distance2 = 0;
        for (int i = 0; i < 3; ++i)
            distance2 += std::pow(proposal->position[i] - track.source[i], 2);
        EXPECT_LE(std::sqrt(distance2), 0.005);
        EXPECT_LT(proposal->candidate_cost, proposal->source_cost);
        EXPECT_FALSE(propose_sparse_point(track, 0));
        EXPECT_FALSE(propose_sparse_point(track, std::numeric_limits<double>::infinity()));
        EXPECT_FALSE(propose_sparse_point(track, 1, 0));
        auto invalid = track;
        invalid.measurements.front().u = std::numeric_limits<double>::quiet_NaN();
        EXPECT_FALSE(propose_sparse_point(invalid, 1));
    }

    std::vector<ReprojectionObservation> sparse_points() {
        std::vector<ReprojectionObservation> points;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 5; ++col) {
                const double x = (col - 2) * 0.5;
                const double y = (row - 1.5) * 0.5;
                const double z = 3.0 + 0.2 * (row + col);
                points.push_back({calibration.fx * x / z + calibration.cx,
                                  calibration.fy * y / z + calibration.cy, x, y, z});
            }
        }
        return points;
    }

    TEST(CameraPoseReprojectionTest, AccurateCalibrationRejectsPhotometricDrift) {
        const auto source = identity_transform();
        const SparseReprojectionGuard guard(source, calibration, sparse_points());
        ASSERT_TRUE(guard.active());
        EXPECT_TRUE(guard.allows(source));
        EXPECT_FALSE(guard.allows(exp_se3({0.01f, 0, 0, 0, 0, 0})));
        EXPECT_FALSE(guard.allows(exp_se3({0, 0, 0, 0, 0.002f, 0})));
        BoundedPoseOptimizer optimizer(7, source, {});
        const PoseEvaluation evaluation{7, 1, 0, 0.0004, {-0.04f, 0, 0, 0, 0, 0}};
        const auto update = optimizer.step(evaluation, [&](const Matrix4& candidate) {
            return guard.allows(candidate) ? 0.0 : std::numeric_limits<double>::infinity();
        });
        EXPECT_EQ(update.status, PoseStepStatus::Rejected);
        EXPECT_EQ(optimizer.snapshot().current, source);
    }

    TEST(CameraPoseReprojectionTest, GeometricProposalRecoversCoupledRotationAndTranslation) {
        const auto source = exp_se3({0.02f, -0.015f, 0.01f, 0.001f, -0.002f, 0.0015f});
        const SparseReprojectionGuard guard(source, calibration, sparse_points());
        const auto proposal = guard.proposal(source, 1.0);
        ASSERT_TRUE(proposal.has_value());
        const auto corrected = apply_left_increment(*proposal, source);
        EXPECT_TRUE(guard.allows(corrected));
        EXPECT_LT(guard.error(corrected), guard.error(source) * 0.01);
        // Changing units used by the solver must not change the world-unit step.
        const auto other_scale = guard.proposal(source, 100.0);
        ASSERT_TRUE(other_scale.has_value());
        for (size_t i = 0; i < proposal->size(); ++i)
            EXPECT_NEAR((*proposal)[i], (*other_scale)[i], 1e-7);
        auto reduced = calibration;
        reduced.fx /= 4;
        reduced.fy /= 4;
        reduced.cx /= 4;
        reduced.cy /= 4;
        reduced.width /= 4;
        reduced.height /= 4;
        auto points = sparse_points();
        for (auto& point : points) {
            point.u /= 4;
            point.v /= 4;
        }
        const auto resized = SparseReprojectionGuard(source, reduced, points).proposal(source, 1.0);
        ASSERT_TRUE(resized.has_value());
        for (size_t i = 0; i < proposal->size(); ++i)
            EXPECT_NEAR((*proposal)[i], (*resized)[i], 1e-7);
    }

    TEST(CameraPoseReprojectionTest, GeometricProposalRejectsMissingAndUnobservableGeometry) {
        const auto source = identity_transform();
        EXPECT_FALSE(SparseReprojectionGuard{}.proposal(source, 1.0));
        std::vector<ReprojectionObservation> points;
        for (int i = 0; i < 20; ++i) {
            const double x = (i - 9.5) * 0.2, y = x, z = 4 + 0.2 * x;
            points.push_back({calibration.fx * x / z + calibration.cx,
                              calibration.fy * y / z + calibration.cy, x, y, z});
        }
        const SparseReprojectionGuard line(source, calibration, points);
        ASSERT_TRUE(line.active());
        EXPECT_FALSE(line.proposal(source, 1.0));
        const SparseReprojectionGuard good(source, calibration, sparse_points());
        EXPECT_FALSE(good.proposal(source, 0));
        EXPECT_FALSE(good.proposal(source, std::numeric_limits<double>::quiet_NaN()));
        auto invalid = source;
        invalid[0] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(good.proposal(invalid, 1.0));
    }

    TEST(CameraPoseControllerTest, GeometricProposalStillRequiresPhotometricDescentAndBoundedBudget) {
        const auto source = identity_transform();
        const PoseEvaluation baseline{7, 1, 0, 4e-6, {-0.004f, 0, 0, 0, 0, 0}, Twist{0.002f, 0, 0, 0, 0, 0}};
        const auto loss = [](const Matrix4& pose) { return std::pow(pose[3] - 0.002, 2); };
        BoundedPoseOptimizer guided(7, source, {});
        EXPECT_EQ(guided.step(baseline, loss).status, PoseStepStatus::Accepted);
        EXPECT_NEAR(guided.snapshot().current[3], 0.002, 1e-8);

        BoundedPoseOptimizer rejected(7, source, {});
        int evaluations = 0;
        const auto result = rejected.step(baseline, [&](const Matrix4&) { ++evaluations; return 1.0; });
        EXPECT_EQ(result.status, PoseStepStatus::Rejected);
        EXPECT_LE(evaluations, BoundedPoseConfig{}.max_backtracks);
        EXPECT_EQ(rejected.snapshot().current, source);

        BoundedPoseOptimizer retry(7, source, {});
        int retry_evaluations = 0;
        const auto retried = retry.step(baseline, [&](const Matrix4& pose) {
            ++retry_evaluations;
            return pose[3] <= 0.0021f ? 1.0 : loss(pose);
        });
        EXPECT_EQ(retried.status, PoseStepStatus::Accepted);
        EXPECT_GT(retry_evaluations, 1);
        EXPECT_LE(retry_evaluations, BoundedPoseConfig{}.max_backtracks);
        EXPECT_NEAR(retry.snapshot().current[3], 0.003, 1e-8);

        auto wrong = baseline;
        wrong.geometric_proposal = Twist{-0.002f, 0, 0, 0, 0, 0};
        auto ordinary = baseline;
        ordinary.geometric_proposal.reset();
        BoundedPoseOptimizer fallback(7, source, {}), control(7, source, {});
        EXPECT_EQ(fallback.step(wrong, loss).status, control.step(ordinary, loss).status);
        EXPECT_EQ(fallback.snapshot().current, control.snapshot().current);
    }

    TEST(CameraPoseReprojectionTest, PermitsCorrectionButKeepsImmutableSourceCeiling) {
        const auto source = exp_se3({0.04f, 0, 0, 0, 0, 0});
        const SparseReprojectionGuard guard(source, calibration, sparse_points());
        ASSERT_TRUE(guard.active());
        EXPECT_TRUE(guard.allows(identity_transform()));
        EXPECT_TRUE(guard.allows(exp_se3({0.02f, 0, 0, 0, 0, 0})));
        EXPECT_FALSE(guard.allows(exp_se3({0.06f, 0, 0, 0, 0, 0})));
        for (int i = 0; i < 100; ++i)
            EXPECT_FALSE(guard.allows(exp_se3({0.041f, 0, 0, 0, 0, 0})));
    }

    TEST(CameraPoseReprojectionTest, ResizeAndWorldTranslationPreserveDecisions) {
        const auto source = exp_se3({0.02f, 0, 0, 0, 0, 0});
        auto points = sparse_points();
        const SparseReprojectionGuard original(source, calibration, points);
        auto reduced = calibration;
        reduced.fx /= 8;
        reduced.fy /= 8;
        reduced.cx /= 8;
        reduced.cy /= 8;
        reduced.width /= 8;
        reduced.height /= 8;
        for (auto& point : points) {
            point.u /= 8;
            point.v /= 8;
        }
        const SparseReprojectionGuard resized(source, reduced, points);
        ASSERT_TRUE(resized.active());
        EXPECT_NEAR(original.source_error(), resized.source_error(), 1e-12);
        for (const float shift : {0.0f, 0.01f, 0.03f}) {
            const auto pose = exp_se3({shift, 0, 0, 0, 0, 0});
            EXPECT_EQ(original.allows(pose), resized.allows(pose));
        }
        for (auto& point : points) {
            point.x += 10;
            point.y -= 3;
            point.z += 8;
        }
        auto translated_source = source;
        translated_source[3] -= 10;
        translated_source[7] += 3;
        translated_source[11] -= 8;
        const SparseReprojectionGuard translated(translated_source, reduced, points);
        ASSERT_TRUE(translated.active());
        EXPECT_NEAR(translated.source_error(), original.source_error(), 1e-7);
        auto corrected = identity_transform();
        corrected[3] = -10;
        corrected[7] = 3;
        corrected[11] = -8;
        EXPECT_TRUE(translated.allows(corrected));
    }

    TEST(CameraPoseReprojectionTest, FixedSupportCannotDisappearOrHideBehindOutliers) {
        auto points = sparse_points();
        points.push_back({100, 100, 0, 0, 3});  // Gross source mismatch.
        points.push_back({800, 600, 0, 0, -3}); // Not visible in the source.
        const SparseReprojectionGuard guard(identity_transform(), calibration, points);
        ASSERT_TRUE(guard.active());
        EXPECT_EQ(guard.observation_count(), 20u);
        EXPECT_FALSE(guard.allows(exp_se3({0, 0, -4, 0, 0, 0})));
        auto invalid = identity_transform();
        invalid[3] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(guard.allows(invalid));
        EXPECT_FALSE(guard.allows(exp_se3({0.02f, 0, 0, 0, 0, 0})));
    }

    TEST(CameraPoseReprojectionTest, MissingOrConcentratedEvidenceDoesNotClaimProtection) {
        EXPECT_FALSE(SparseReprojectionGuard{}.active());
        auto points = sparse_points();
        points.resize(5);
        EXPECT_FALSE(SparseReprojectionGuard(identity_transform(), calibration, points).active());
        points.assign(20, ReprojectionObservation{800, 600, 0, 0, 3});
        const SparseReprojectionGuard concentrated(identity_transform(), calibration, points);
        EXPECT_FALSE(concentrated.active());
        EXPECT_TRUE(concentrated.allows(exp_se3({0.01f, 0, 0, 0, 0, 0})));
    }

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
