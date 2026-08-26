/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_temporal_coordinator.hpp"

#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {
        SceneTemporalRequest requestFor(const TemporalViewId view = TemporalViewId::Main) {
            SceneTemporalRequest request{
                .view = view,
                .requirements = {
                    .depth = true,
                    .motion = true,
                    .jitter = true,
                    .history_color = true,
                    .history_depth = true,
                },
                .render_extent = {1280, 720},
                .output_extent = {1920, 1080},
            };
            request.frame.view.size = request.render_extent;
            request.frame.output_extent = request.output_extent;
            request.frame.render_scale = 2.0f / 3.0f;
            request.frame.scene_generation = 7;
            request.frame.backend_key = 42;
            return request;
        }

        bool commitComplete(SceneTemporalCoordinator& coordinator,
                            const PreparedSceneTemporalFrame& prepared) {
            return coordinator.commit(prepared,
                                      SceneHistoryStorage::VulkanImage,
                                      SceneHistoryStorage::VulkanImage);
        }
    } // namespace

    TEST(SceneTemporalCoordinator, DisabledPlanDoesNotIssueTicketOrAdvanceState) {
        SceneTemporalCoordinator coordinator;
        auto disabled = requestFor();
        disabled.requirements = {};
        const auto prepared = coordinator.prepare(disabled);
        EXPECT_TRUE(prepared.plan.zeroCost());
        EXPECT_EQ(prepared.ticket, 0u);
        EXPECT_FALSE(coordinator.commit(prepared));

        const auto first = coordinator.prepare(requestFor());
        EXPECT_EQ(first.frame.sequence, 0u);
        EXPECT_FALSE(first.frame.history_valid);
    }

    TEST(SceneTemporalCoordinator, CommitsFrameAndHistoryAsOneTransaction) {
        SceneTemporalCoordinator coordinator;
        const auto first = coordinator.prepare(requestFor());
        ASSERT_TRUE(commitComplete(coordinator, first));

        const auto second = coordinator.prepare(requestFor());
        EXPECT_EQ(second.frame.sequence, 1u);
        EXPECT_TRUE(second.frame.history_valid);
        EXPECT_TRUE(second.history.available());
        EXPECT_EQ(second.history.sequence, second.frame.sequence);
    }

    TEST(SceneTemporalCoordinator, RejectsReplayedAndSupersededPreparedFrames) {
        SceneTemporalCoordinator coordinator;
        const auto first = coordinator.prepare(requestFor());
        ASSERT_TRUE(commitComplete(coordinator, first));
        EXPECT_FALSE(commitComplete(coordinator, first));

        const auto stale = coordinator.prepare(requestFor());
        const auto current = coordinator.prepare(requestFor());
        EXPECT_FALSE(commitComplete(coordinator, stale));
        EXPECT_TRUE(commitComplete(coordinator, current));
    }

    TEST(SceneTemporalCoordinator, FailedHistoryCommitCannotAdvanceFrameState) {
        SceneTemporalCoordinator coordinator;
        const auto first = coordinator.prepare(requestFor());
        EXPECT_FALSE(coordinator.commit(first,
                                        SceneHistoryStorage::VulkanImage,
                                        SceneHistoryStorage::None));

        const auto retry = coordinator.prepare(requestFor());
        EXPECT_TRUE(retry.active());
        EXPECT_EQ(retry.frame.sequence, 0u);
        EXPECT_FALSE(retry.frame.history_valid);
        EXPECT_TRUE(hasTemporalResetReason(retry.frame.reset_reasons,
                                           TemporalResetReason::InvalidInput));
        EXPECT_TRUE(commitComplete(coordinator, retry));
    }

    TEST(SceneTemporalCoordinator, ValidFrameRecoversFromPendingInvalidInputReset) {
        SceneTemporalCoordinator coordinator;
        coordinator.resetAll(TemporalResetReason::InvalidInput);

        const auto recovered = coordinator.prepare(requestFor());
        EXPECT_TRUE(recovered.active());
        EXPECT_FALSE(recovered.frame.history_valid);
        EXPECT_TRUE(hasTemporalResetReason(recovered.frame.reset_reasons,
                                           TemporalResetReason::InvalidInput));
        EXPECT_TRUE(commitComplete(coordinator, recovered));

        const auto next = coordinator.prepare(requestFor());
        EXPECT_TRUE(next.active());
        EXPECT_TRUE(next.frame.history_valid);
        EXPECT_EQ(next.frame.reset_reasons, TemporalResetReason::None);
    }

    TEST(SceneTemporalCoordinator, DiscardInvalidatesOnlyMatchingPendingFrame) {
        SceneTemporalCoordinator coordinator;
        const auto left = coordinator.prepare(requestFor(TemporalViewId::SplitLeft));
        const auto right = coordinator.prepare(requestFor(TemporalViewId::SplitRight));
        coordinator.discard(left);
        EXPECT_FALSE(commitComplete(coordinator, left));
        EXPECT_TRUE(commitComplete(coordinator, right));
    }

    TEST(SceneTemporalCoordinator, KeepsEveryViewIndependent) {
        SceneTemporalCoordinator coordinator;
        for (const auto view : {TemporalViewId::Main,
                                TemporalViewId::SplitLeft,
                                TemporalViewId::SplitRight}) {
            ASSERT_TRUE(commitComplete(coordinator, coordinator.prepare(requestFor(view))));
        }
        for (const auto view : {TemporalViewId::Main,
                                TemporalViewId::SplitLeft,
                                TemporalViewId::SplitRight}) {
            const auto next = coordinator.prepare(requestFor(view));
            EXPECT_TRUE(next.frame.history_valid);
            EXPECT_TRUE(next.history.available());
        }
    }

    TEST(SceneTemporalCoordinator, ResetAllInvalidatesFramesHistoryAndTickets) {
        SceneTemporalCoordinator coordinator;
        const auto pending = coordinator.prepare(requestFor());
        coordinator.resetAll(TemporalResetReason::Scene);
        EXPECT_FALSE(commitComplete(coordinator, pending));

        const auto reset = coordinator.prepare(requestFor());
        EXPECT_FALSE(reset.frame.history_valid);
        EXPECT_FALSE(reset.history.available());
        EXPECT_TRUE(hasTemporalResetReason(reset.frame.reset_reasons,
                                           TemporalResetReason::Scene));
        ASSERT_TRUE(commitComplete(coordinator, reset));
        const auto restarted = coordinator.prepare(requestFor());
        EXPECT_EQ(restarted.frame.sequence, 1u);
        EXPECT_TRUE(restarted.frame.history_valid);
    }

} // namespace lfs::vis
