/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/temporal_frame_tracker.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {
    namespace {
        TemporalFrameInput frameInput() {
            TemporalFrameInput input;
            input.view.size = {1280, 720};
            input.scene_generation = 4;
            input.backend_key = 2;
            return input;
        }
    } // namespace

    TEST(TemporalFrameTracker, FirstFrameHasNoHistoryAndCommitEnablesNextFrame) {
        TemporalFrameTracker tracker;
        const auto input = frameInput();
        EXPECT_TRUE(hasTemporalResetReason(tracker.prepare(TemporalViewId::Main, input).reset_reasons,
                                           TemporalResetReason::FirstFrame));
        tracker.commit(TemporalViewId::Main, input);
        const auto next = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_TRUE(next.history_valid);
        EXPECT_EQ(next.sequence, 1u);
    }

    TEST(TemporalFrameTracker, CameraMotionPreservesHistoryButExplicitCutResetsIt) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        input.view.translation.x = 1.0f;
        EXPECT_TRUE(tracker.prepare(TemporalViewId::Main, input).history_valid);
        input.camera_cut = true;
        const auto cut = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_FALSE(cut.history_valid);
        EXPECT_TRUE(hasTemporalResetReason(cut.reset_reasons, TemporalResetReason::CameraCut));
    }

    TEST(TemporalFrameTracker, RenderAndProjectionChangesResetHistory) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        input.view.size = {960, 540};
        input.render_scale = 0.75f;
        input.view.focal_length_mm = 50.0f;
        const auto changed = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::RenderSize));
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::RenderScale));
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::Projection));
    }

    TEST(TemporalFrameTracker, SceneAndBackendChangesResetHistory) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        ++input.scene_generation;
        ++input.backend_key;
        const auto changed = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::Scene));
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::Backend));
    }

    TEST(TemporalFrameTracker, PrepareDoesNotAdvanceAndViewsRemainIndependent) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        EXPECT_EQ(tracker.prepare(TemporalViewId::Main, input).sequence, 1u);
        EXPECT_EQ(tracker.prepare(TemporalViewId::Main, input).sequence, 1u);
        EXPECT_TRUE(hasTemporalResetReason(
            tracker.prepare(TemporalViewId::SplitRight, input).reset_reasons,
            TemporalResetReason::FirstFrame));
    }

    TEST(TemporalFrameTracker, InvalidInputCannotBecomeHistory) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        input.view.translation.x = std::numeric_limits<float>::quiet_NaN();
        tracker.commit(TemporalViewId::Main, input);
        const auto state = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_FALSE(state.history_valid);
        EXPECT_TRUE(hasTemporalResetReason(state.reset_reasons, TemporalResetReason::InvalidInput));
    }
} // namespace lfs::vis
