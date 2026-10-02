/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/frame_demand.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <optional>

namespace {

    using namespace std::chrono_literals;
    namespace DirtyFlag = lfs::vis::DirtyFlag;
    using lfs::vis::FrameClock;
    using lfs::vis::FrameDemandLedger;
    using lfs::vis::FrameReason;
    using lfs::vis::FrameRequest;
    using lfs::vis::FrameScope;

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

    TEST(FrameDemandLedger, DeadlineElapsedDuringPresentWakesImmediately) {
        const auto deadline = t0() + 16ms;
        EXPECT_DOUBLE_EQ(lfs::vis::secondsUntilFrameDeadline(deadline, t0()), 0.016);
        EXPECT_DOUBLE_EQ(lfs::vis::secondsUntilFrameDeadline(deadline, deadline), 0.0);
        EXPECT_DOUBLE_EQ(lfs::vis::secondsUntilFrameDeadline(deadline, t0() + 100ms), 0.0);
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
        ledger.countViewRendered(plan.render_views, plan);
        ledger.countPresented(plan);

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

    TEST(FrameDemandLedger, HolderEndsWhenAliveTurnsFalse) {
        FrameDemandLedger ledger;
        ledger.setDisplayInterval(10ms);
        bool playing = true;
        auto token = ledger.hold(FrameReason::Playback, FrameScope::View, 1, DirtyFlag::CAMERA, [&] { return playing; }, "sequencer");
        EXPECT_FALSE(ledger.plan(t0()).empty());
        EXPECT_EQ(ledger.snapshot(t0()).live_holders.size(), 1u);
        playing = false;
        EXPECT_TRUE(ledger.plan(t0() + 10ms).empty());
        EXPECT_EQ(ledger.snapshot(t0()).live_holders.size(), 0u);
        EXPECT_EQ(ledger.snapshot(t0()).holders_expired, 1u);
        EXPECT_FALSE(ledger.nextDeadline(t0() + 10ms).has_value());
    }

    TEST(FrameDemandLedger, ThumbnailCompletionCanRequestAViewFromAnAnimationPredicate) {
        FrameDemandLedger ledger;
        bool thumbnail_ready = true;
        auto token = ledger.hold(FrameReason::GuiAnimation, FrameScope::Gui, 0, 0,
                                 [&] {
                                     // Camera thumbnail readiness is checked by the visible GUI holder.
                                     // It invalidates the viewport overlay while the plan is evaluated.
                                     if (thumbnail_ready) {
                                         ledger.request(FrameRequest{.reason = FrameReason::Overlay,
                                                                     .scope = FrameScope::View,
                                                                     .views = 1,
                                                                     .flags = DirtyFlag::OVERLAY});
                                     }
                                     return thumbnail_ready;
                                 });
        const auto plan = ledger.plan(t0());
        EXPECT_EQ(plan.render_views, 1u);
        EXPECT_EQ(plan.view_flags[0], DirtyFlag::OVERLAY);
        EXPECT_TRUE(plan.reasons.test(static_cast<std::size_t>(FrameReason::Overlay)));
        ledger.countViewRendered(plan.render_views, plan);
        ledger.countPresented(plan);
        thumbnail_ready = false;
        EXPECT_TRUE(ledger.plan(t0() + 20ms).empty());
        EXPECT_FALSE(ledger.nextDeadline(t0() + 20ms).has_value());
        EXPECT_EQ(ledger.snapshot().views_rendered[0], 1u);
    }

    TEST(FrameDemandLedger, PredicateCanReleaseItsOwnHolder) {
        FrameDemandLedger ledger;
        lfs::vis::DemandToken token;
        token = ledger.hold(FrameReason::GuiAnimation, FrameScope::Gui, 0, 0,
                            [&] {
                                token.release();
                                return true;
                            });
        EXPECT_TRUE(ledger.plan(t0()).empty());
        EXPECT_FALSE(token.active());
        EXPECT_EQ(ledger.snapshot().live_holders.size(), 0u);
        EXPECT_FALSE(ledger.nextDeadline(t0()).has_value());
    }

    TEST(FrameDemandLedger, ReleasingTheTokenEndsTheHolder) {
        FrameDemandLedger ledger;
        {
            auto token = ledger.hold(FrameReason::GuiAnimation, FrameScope::Gui, 0, 0,
                                     [] { return true; });
            EXPECT_EQ(ledger.snapshot().live_holders.size(), 1u);
        }
        EXPECT_EQ(ledger.snapshot().live_holders.size(), 0u);
        EXPECT_TRUE(ledger.plan(t0()).empty());
    }

    TEST(FrameDemandLedger, DisplayCadenceNeverFiresEarlierThanTheDisplayInterval) {
        FrameDemandLedger ledger;
        ledger.setDisplayInterval(16ms);
        auto token = ledger.hold(FrameReason::CameraMotion, FrameScope::View, 1, DirtyFlag::CAMERA,
                                 [] { return true; });
        EXPECT_FALSE(ledger.plan(t0()).empty());
        EXPECT_TRUE(ledger.plan(t0() + 5ms).empty());
        ASSERT_TRUE(ledger.nextDeadline(t0() + 5ms).has_value());
        EXPECT_EQ(*ledger.nextDeadline(t0() + 5ms), t0() + 16ms);
        EXPECT_FALSE(ledger.plan(t0() + 16ms).empty());
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
        ledger.countPresented(empty);
        EXPECT_EQ(ledger.snapshot(t0()).frames_without_reason, 1u);
    }

} // namespace
