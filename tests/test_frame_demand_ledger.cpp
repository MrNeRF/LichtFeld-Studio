/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU-only contract tests for the frame-demand ledger: idle means no plan
// and no deadline; every frame carries a reason; continuous demand is finite
// and paced by its own cadence, never by the loop.

#include "visualizer/rendering/frame_demand.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <optional>

namespace {

    using namespace std::chrono_literals;
    using lfs::vis::Cadence;
    namespace DirtyFlag = lfs::vis::DirtyFlag;
    using lfs::vis::Finite;
    using lfs::vis::FrameClock;
    using lfs::vis::FrameDemandLedger;
    using lfs::vis::FrameReason;
    using lfs::vis::FrameRequest;
    using lfs::vis::FrameScope;
    using lfs::vis::trainingPreviewStepAdvanced;

    FrameClock::time_point t0() { return FrameClock::time_point{} + 1h; }

    TEST(FrameDemandLedger, IdleHasNoPlanAndNoDeadline) {
        FrameDemandLedger ledger;
        const auto plan = ledger.plan(t0());
        EXPECT_TRUE(plan.empty());
        EXPECT_FALSE(ledger.nextDeadline(t0()).has_value());
        const auto s = ledger.snapshot(t0());
        EXPECT_EQ(s.frames_presented, 0u);
        EXPECT_EQ(s.wakes_without_frame, 1u);
        EXPECT_TRUE(s.live_holders.empty());
    }

    TEST(FrameDemandLedger, OneShotIsConsumedExactlyOnce) {
        FrameDemandLedger ledger;
        int wakes = 0;
        ledger.setWakeCallback([&] { ++wakes; });
        ledger.request(FrameRequest{.reason = FrameReason::SceneChange,
                                    .scope = FrameScope::View,
                                    .views = 1,
                                    .flags = DirtyFlag::ALL,
                                    .detail = "test"});
        EXPECT_EQ(wakes, 1);
        ASSERT_TRUE(ledger.nextDeadline(t0()).has_value());

        auto plan = ledger.plan(t0());
        EXPECT_TRUE(plan.present);
        EXPECT_EQ(plan.render_views, 1u);
        EXPECT_EQ(plan.view_flags[0], DirtyFlag::ALL);
        EXPECT_TRUE(plan.reasons.test(static_cast<std::size_t>(FrameReason::SceneChange)));
        ASSERT_EQ(plan.details.size(), 1u);
        EXPECT_EQ(plan.details[0], "test");
        ledger.noteViewRendered(plan.render_views, plan);
        ledger.notePresented(plan);

        EXPECT_TRUE(ledger.plan(t0() + 1ms).empty());
        EXPECT_FALSE(ledger.nextDeadline(t0() + 1ms).has_value());
        const auto s = ledger.snapshot(t0());
        EXPECT_EQ(s.frames_presented, 1u);
        EXPECT_EQ(s.views_rendered[0], 1u);
        EXPECT_EQ(s.presents_by_reason[static_cast<std::size_t>(FrameReason::SceneChange)], 1u);
        EXPECT_EQ(s.frames_without_reason, 0u);
    }

    TEST(FrameDemandLedger, GuiReasonNeverRendersAView) {
        FrameDemandLedger ledger;
        ledger.request(FrameRequest{.reason = FrameReason::Input, .scope = FrameScope::Gui});
        const auto plan = ledger.plan(t0());
        EXPECT_TRUE(plan.present);
        EXPECT_EQ(plan.render_views, 0u);
    }

    TEST(FrameDemandLedger, ViewRequestWithoutViewsIsDroppedAndCounted) {
        FrameDemandLedger ledger;
#ifndef NDEBUG
        EXPECT_DEATH(
            ledger.request(FrameRequest{.reason = FrameReason::Selection,
                                        .scope = FrameScope::View,
                                        .views = 0}),
            "non-empty view mask");
#else
        ledger.request(FrameRequest{.reason = FrameReason::Selection, .scope = FrameScope::View, .views = 0});
        EXPECT_TRUE(ledger.plan(t0()).empty());
        EXPECT_EQ(ledger.snapshot(t0()).requests_dropped, 1u);
#endif
    }

    TEST(FrameDemandLedger, DeadlineOneShotWaitsUntilDue) {
        FrameDemandLedger ledger;
        ledger.request(FrameRequest{.reason = FrameReason::Tooltip,
                                    .scope = FrameScope::Gui,
                                    .not_before = t0() + 300ms});
        EXPECT_TRUE(ledger.plan(t0()).empty());
        ASSERT_TRUE(ledger.nextDeadline(t0()).has_value());
        EXPECT_EQ(*ledger.nextDeadline(t0()), t0() + 300ms);
        EXPECT_FALSE(ledger.plan(t0() + 300ms).empty());
    }

    TEST(FrameDemandLedger, HolderEndsWhenAliveTurnsFalse) {
        FrameDemandLedger ledger;
        ledger.setDisplayInterval(10ms);
        bool playing = true;
        auto token = ledger.hold(FrameReason::Playback, FrameScope::View, 1, DirtyFlag::CAMERA,
                                 Cadence::display(), Finite::alive([&] { return playing; }), "sequencer");
        EXPECT_FALSE(ledger.plan(t0()).empty());
        EXPECT_EQ(ledger.snapshot(t0()).live_holders.size(), 1u);
        playing = false;
        EXPECT_TRUE(ledger.plan(t0() + 10ms).empty());
        EXPECT_EQ(ledger.snapshot(t0()).live_holders.size(), 0u);
        EXPECT_EQ(ledger.snapshot(t0()).holders_expired, 1u);
        EXPECT_FALSE(ledger.nextDeadline(t0() + 10ms).has_value());
    }

    TEST(FrameDemandLedger, HolderEndsAtItsDeadline) {
        FrameDemandLedger ledger;
        ledger.setDisplayInterval(10ms);
        auto token = ledger.hold(FrameReason::StatusMessage, FrameScope::Gui, 0, 0, Cadence::display(),
                                 Finite::until(t0() + 25ms));
        EXPECT_FALSE(ledger.plan(t0()).empty());
        EXPECT_FALSE(ledger.plan(t0() + 10ms).empty());
        EXPECT_FALSE(ledger.plan(t0() + 20ms).empty());
        EXPECT_TRUE(ledger.plan(t0() + 25ms).empty());
        EXPECT_FALSE(ledger.nextDeadline(t0() + 25ms).has_value());
    }

    TEST(FrameDemandLedger, ReleasingTheTokenEndsTheHolder) {
        FrameDemandLedger ledger;
        {
            auto token = ledger.hold(FrameReason::GuiAnimation, FrameScope::Gui, 0, 0, Cadence::display(),
                                     Finite::alive([] { return true; }));
            EXPECT_EQ(ledger.liveHolderCount(), 1u);
        }
        EXPECT_EQ(ledger.liveHolderCount(), 0u);
        EXPECT_TRUE(ledger.plan(t0()).empty());
    }

    TEST(FrameDemandLedger, DisplayCadenceNeverFiresEarlierThanTheDisplayInterval) {
        FrameDemandLedger ledger;
        ledger.setDisplayInterval(16ms);
        auto token = ledger.hold(FrameReason::CameraMotion, FrameScope::View, 1, DirtyFlag::CAMERA,
                                 Cadence::display(), Finite::alive([] { return true; }));
        EXPECT_FALSE(ledger.plan(t0()).empty());
        EXPECT_TRUE(ledger.plan(t0() + 5ms).empty());
        ASSERT_TRUE(ledger.nextDeadline(t0() + 5ms).has_value());
        EXPECT_EQ(*ledger.nextDeadline(t0() + 5ms), t0() + 16ms);
        EXPECT_FALSE(ledger.plan(t0() + 16ms).empty());
    }

    TEST(FrameDemandLedger, IntervalCadenceFiresOnItsOwnPeriod) {
        FrameDemandLedger ledger;
        auto token = ledger.hold(FrameReason::HudSample, FrameScope::Gui, 0, 0, Cadence::every(250ms),
                                 Finite::alive([] { return true; }), "vram_hud");
        EXPECT_FALSE(ledger.plan(t0()).empty());
        EXPECT_TRUE(ledger.plan(t0() + 100ms).empty());
        EXPECT_EQ(*ledger.nextDeadline(t0() + 100ms), t0() + 250ms);
        EXPECT_FALSE(ledger.plan(t0() + 250ms).empty());
    }

    // The training preview is paced by its own scheduler (the stall fix's
    // interval); the ledger neither adds nor removes frames.
    TEST(FrameDemandLedger, ExternalCadenceIsOwnedByItsScheduler) {
        FrameDemandLedger ledger;
        std::optional<FrameClock::time_point> next_due = t0() + 440ms;
        auto token = ledger.hold(FrameReason::TrainingPreview, FrameScope::View, 1, DirtyFlag::SPLATS,
                                 Cadence::external([&](FrameClock::time_point) { return next_due; }),
                                 Finite::alive([] { return true; }), "training");
        EXPECT_TRUE(ledger.plan(t0()).empty());
        EXPECT_TRUE(ledger.plan(t0() + 250ms).empty());
        EXPECT_EQ(*ledger.nextDeadline(t0()), t0() + 440ms);
        const auto plan = ledger.plan(t0() + 440ms);
        EXPECT_EQ(plan.render_views, 1u);
        EXPECT_EQ(plan.view_flags[0], DirtyFlag::SPLATS);
        next_due.reset(); // scheduler says: nothing due (e.g. no step completed)
        EXPECT_TRUE(ledger.plan(t0() + 441ms).empty());
        EXPECT_FALSE(ledger.nextDeadline(t0() + 441ms).has_value());
    }

    TEST(FrameDemandLedger, TrainingPreviewRequiresACompletedStep) {
        EXPECT_FALSE(trainingPreviewStepAdvanced(17, 17));
        EXPECT_FALSE(trainingPreviewStepAdvanced(18, 17));
        EXPECT_TRUE(trainingPreviewStepAdvanced(17, 18));
    }

    TEST(FrameDemandLedger, WakeWithoutFrameBoundsTheWaitOnly) {
        FrameDemandLedger ledger;
        ledger.wakeAt(t0() + 3s, "scratch_release");
        EXPECT_EQ(*ledger.nextDeadline(t0()), t0() + 3s);
        EXPECT_TRUE(ledger.plan(t0() + 3s).empty());
        EXPECT_FALSE(ledger.nextDeadline(t0() + 3s).has_value());
    }

    TEST(FrameDemandLedger, ExplicitViewFlagsStaySeparate) {
        FrameDemandLedger ledger;
        ledger.request(FrameRequest{.reason = FrameReason::CameraMotion,
                                    .scope = FrameScope::View,
                                    .views = 0b10,
                                    .flags = DirtyFlag::CAMERA});
        ledger.request(FrameRequest{.reason = FrameReason::Selection,
                                    .scope = FrameScope::View,
                                    .views = 0b01,
                                    .flags = DirtyFlag::SELECTION});
        const auto plan = ledger.plan(t0());
        EXPECT_EQ(plan.render_views, 0b11);
        EXPECT_EQ(plan.view_flags[0], DirtyFlag::SELECTION);
        EXPECT_EQ(plan.view_flags[1], DirtyFlag::CAMERA);
    }

    TEST(FrameDemandLedger, PresentWithoutReasonIsCounted) {
        FrameDemandLedger ledger;
        lfs::vis::FramePlan empty;
        ledger.notePresented(empty);
        EXPECT_EQ(ledger.snapshot(t0()).frames_without_reason, 1u);
    }

} // namespace
