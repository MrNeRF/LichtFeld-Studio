/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/pane_interaction.hpp"

#include <gtest/gtest.h>

namespace {

    using lfs::vis::PaneGestureKind;
    using lfs::vis::PaneInteraction;
    using lfs::vis::PaneSnapshot;
    using lfs::vis::SplitAxis;
    using lfs::vis::SplitterRect;
    using lfs::vis::ViewId;
    using lfs::vis::ViewRect;
    using lfs::vis::WorkspaceFrameSnapshot;

    PaneSnapshot pane(const ViewId id, const ViewRect rect, const bool focused = false) {
        PaneSnapshot result;
        result.id = id;
        result.rect = rect;
        result.focused = focused;
        result.projection.focal_length_mm = static_cast<float>(id);
        result.rotation[0][0] = static_cast<float>(id);
        result.translation.x = static_cast<float>(id * 10);
        result.window_size = {rect.width, rect.height};
        result.framebuffer_size = {rect.width * 2, rect.height * 2};
        result.state_generation = id * 100;
        return result;
    }

    WorkspaceFrameSnapshot fourPanes() {
        WorkspaceFrameSnapshot snapshot;
        snapshot.outer = {0, 0, 100, 100};
        snapshot.panes = {
            pane(1, {0, 0, 48, 48}, true),
            pane(2, {52, 0, 48, 48}),
            pane(3, {0, 52, 48, 48}),
            pane(4, {52, 52, 48, 48}),
        };
        snapshot.splitters = {
            SplitterRect{11, SplitAxis::Vertical, {48, 0, 4, 100}},
            SplitterRect{12, SplitAxis::Horizontal, {0, 48, 100, 4}},
        };
        snapshot.focused = 1;
        snapshot.primary = 1;
        return snapshot;
    }

} // namespace

TEST(PaneInteraction, FourPaneHitTestUsesHalfOpenRectsAndMissesDividers) {
    const auto snapshot = fourPanes();

    const auto top_left = PaneInteraction::hitTest(snapshot, {0.0f, 0.0f});
    ASSERT_TRUE(top_left);
    EXPECT_EQ(top_left->view, 1u);
    EXPECT_EQ(top_left->local, glm::vec2(0.0f));
    EXPECT_EQ(top_left->rect, (ViewRect{0, 0, 48, 48}));
    EXPECT_FLOAT_EQ(top_left->camera.projection.focal_length_mm, 1.0f);
    EXPECT_EQ(top_left->camera.state_generation, 100u);

    EXPECT_EQ(PaneInteraction::hitTest(snapshot, {50.0f, 25.0f}), std::nullopt);
    EXPECT_EQ(PaneInteraction::hitTest(snapshot, {25.0f, 50.0f}), std::nullopt);
    EXPECT_EQ(PaneInteraction::hitTest(snapshot, {100.0f, 99.0f}), std::nullopt);

    const auto bottom_right = PaneInteraction::hitTest(snapshot, {99.0f, 99.0f});
    ASSERT_TRUE(bottom_right);
    EXPECT_EQ(bottom_right->view, 4u);
    EXPECT_EQ(bottom_right->local, glm::vec2(47.0f, 47.0f));
}

TEST(PaneInteraction, CaptureRetainsInitiatingViewAcrossPanesAndDividers) {
    const auto snapshot = fourPanes();
    PaneInteraction interaction;

    const auto start = interaction.beginCapture(snapshot, {10.0f, 10.0f},
                                                PaneGestureKind::Selection);
    ASSERT_TRUE(start);
    ASSERT_TRUE(interaction.activeCapture());
    EXPECT_EQ(interaction.activeCapture()->view, 1u);
    EXPECT_EQ(interaction.activeCapture()->kind, PaneGestureKind::Selection);

    const auto moved = interaction.updateCapture(snapshot, {75.0f, 75.0f});
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->view, 1u);
    EXPECT_EQ(moved->local, glm::vec2(75.0f, 75.0f));
    EXPECT_TRUE(interaction.releaseCapture());
    EXPECT_FALSE(interaction.hasCapture());
    EXPECT_FALSE(interaction.releaseCapture());
}

TEST(PaneInteraction, CaptureRecomputesCoordinatesAfterPaneResize) {
    auto original = fourPanes();
    PaneInteraction interaction;
    ASSERT_TRUE(interaction.beginCapture(original, {10.0f, 10.0f}, PaneGestureKind::Camera));

    auto resized = original;
    resized.panes[0].rect = {20, 30, 60, 55};
    const auto moved = interaction.updateCapture(resized, {55.0f, 50.0f});
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->view, 1u);
    EXPECT_EQ(moved->rect, (ViewRect{20, 30, 60, 55}));
    EXPECT_EQ(moved->local, glm::vec2(35.0f, 20.0f));
}

TEST(PaneInteraction, ReconcileCancelsClosedOrHiddenCapturedView) {
    auto snapshot = fourPanes();
    PaneInteraction interaction;
    ASSERT_TRUE(interaction.beginCapture(snapshot, {10.0f, 10.0f}, PaneGestureKind::Camera));

    snapshot.panes.erase(snapshot.panes.begin());
    EXPECT_TRUE(interaction.reconcile(snapshot));
    EXPECT_FALSE(interaction.hasCapture());
    EXPECT_FALSE(interaction.reconcile(snapshot));
    EXPECT_EQ(interaction.updateCapture(snapshot, {20.0f, 20.0f}), std::nullopt);

    auto hidden = fourPanes();
    ASSERT_TRUE(interaction.beginCapture(hidden, {10.0f, 10.0f}, PaneGestureKind::Camera));
    hidden.panes[0].rect = {};
    EXPECT_TRUE(interaction.reconcile(hidden));
    EXPECT_FALSE(interaction.hasCapture());
}

TEST(PaneInteraction, KeyboardUsesFocusedViewIndependentlyOfPointerHit) {
    const auto snapshot = fourPanes();
    const auto pointer = PaneInteraction::hitTest(snapshot, {60.0f, 10.0f});
    ASSERT_TRUE(pointer);
    EXPECT_EQ(pointer->view, 2u);
    ASSERT_EQ(PaneInteraction::keyboardTarget(snapshot), std::optional<ViewId>(1));

    auto unfocused = snapshot;
    unfocused.focused.reset();
    EXPECT_EQ(PaneInteraction::keyboardTarget(unfocused), std::nullopt);
}

TEST(PaneInteraction, StaleSnapshotsRemainIndependentAndDoNotShareHitCache) {
    auto first = fourPanes();
    auto second = first;
    second.outer = {100, 200, 200, 100};
    second.panes[0].rect = {100, 200, 100, 100};
    second.panes[1].rect = {200, 200, 100, 100};
    second.panes[2].rect = {100, 300, 100, 100};
    second.panes[3].rect = {200, 300, 100, 100};

    const auto old_hit = PaneInteraction::hitTest(first, {10.0f, 10.0f});
    ASSERT_TRUE(old_hit);
    EXPECT_EQ(old_hit->view, 1u);
    EXPECT_EQ(PaneInteraction::hitTest(first, {150.0f, 250.0f}), std::nullopt);

    const auto new_hit = PaneInteraction::hitTest(second, {150.0f, 250.0f});
    ASSERT_TRUE(new_hit);
    EXPECT_EQ(new_hit->view, 1u);
    EXPECT_EQ(new_hit->local, glm::vec2(50.0f, 50.0f));
}
