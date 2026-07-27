/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/viewport_interop_service.hpp"

#include <gtest/gtest.h>

using lfs::vis::decideViewportInteropEarly;
using lfs::vis::ViewportInteropAction;
using lfs::vis::ViewportInteropDecision;
using lfs::vis::ViewportInteropSlotInputs;

namespace {

    ViewportInteropSlotInputs baseValidInputs() {
        ViewportInteropSlotInputs in{};
        in.source_ok = true;
        in.frame_slot_in_range = true;
        in.target_present = true;
        in.target_size_matches = true;
        in.target_interop_valid = true;
        in.target_layout_read_only = true;
        in.source_generation = 7;
        in.uploaded_source_generation = 7;
        return in;
    }

} // namespace

TEST(ViewportInteropDecision, DisabledReturnsDisabled) {
    auto in = baseValidInputs();
    in.disabled = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::Disabled);
    EXPECT_FALSE(d.clear_published);
    EXPECT_FALSE(d.publish_from_target);
}

TEST(ViewportInteropDecision, SceneExternalHandleEarlyOut) {
    auto in = baseValidInputs();
    in.external_handle_early_out = true;
    in.has_external_scene_image = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::ExternalSkip);
}

TEST(ViewportInteropDecision, NonSceneIgnoresExternalHandle) {
    auto in = baseValidInputs();
    in.external_handle_early_out = false;
    in.has_external_scene_image = true;
    // Valid cache hit should still win for split/depth policy.
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::CacheHit);
    EXPECT_TRUE(d.publish_from_target);
}

TEST(ViewportInteropDecision, InvalidSourceSceneDoesNotClearPublished) {
    auto in = baseValidInputs();
    in.source_ok = false;
    in.publishes_published = false;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::InvalidReset);
    EXPECT_FALSE(d.clear_published);
    EXPECT_TRUE(d.reset_targets_if_nonempty);
}

TEST(ViewportInteropDecision, InvalidSourceSplitClearsPublished) {
    auto in = baseValidInputs();
    in.source_ok = false;
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::InvalidReset);
    EXPECT_TRUE(d.clear_published);
    EXPECT_TRUE(d.reset_targets_if_nonempty);
}

TEST(ViewportInteropDecision, CacheHitSceneDoesNotPublish) {
    auto in = baseValidInputs();
    in.publishes_published = false;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::CacheHit);
    EXPECT_FALSE(d.publish_from_target);
}

TEST(ViewportInteropDecision, CacheHitSplitPublishes) {
    auto in = baseValidInputs();
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::CacheHit);
    EXPECT_TRUE(d.publish_from_target);
}

TEST(ViewportInteropDecision, CacheHitRequiresGenerationAndLayout) {
    {
        auto in = baseValidInputs();
        in.source_generation = 0;
        EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
    }
    {
        auto in = baseValidInputs();
        in.uploaded_source_generation = 3;
        EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
    }
    {
        auto in = baseValidInputs();
        in.target_layout_read_only = false;
        EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
    }
}

TEST(ViewportInteropDecision, DeferBailClearsPublishedForSplit) {
    auto in = baseValidInputs();
    in.publishes_published = true;
    in.target_size_matches = false; // recreate needed
    in.resize_deferring = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::DeferBail);
    EXPECT_TRUE(d.clear_published);
}

TEST(ViewportInteropDecision, DeferBailDoesNotClearForScene) {
    auto in = baseValidInputs();
    in.publishes_published = false;
    in.target_size_matches = false;
    in.resize_deferring = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::DeferBail);
    EXPECT_FALSE(d.clear_published);
}

TEST(ViewportInteropDecision, DeferBailWhenSlotArrayResizeNeeded) {
    auto in = baseValidInputs();
    in.slot_array_resize_needed = true;
    in.resize_deferring = true;
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::DeferBail);
    EXPECT_TRUE(d.clear_published);
}

TEST(ViewportInteropDecision, SlowPathWhenContentChangedAndNotDeferring) {
    auto in = baseValidInputs();
    in.uploaded_source_generation = 1;
    in.source_generation = 2;
    in.resize_deferring = false;
    EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
}

TEST(ViewportInteropDecision, SlowPathWhenRecreateNeededAndNotDeferring) {
    auto in = baseValidInputs();
    in.target_present = false;
    in.resize_deferring = false;
    EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
}
