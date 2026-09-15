/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/corner_split_interaction.hpp"

#include <gtest/gtest.h>

namespace {

    using lfs::vis::CornerSplitInteraction;
    using lfs::vis::CornerSplitInteractionConfig;
    using lfs::vis::SplitAxis;
    using lfs::vis::ViewId;
    using lfs::vis::ViewRect;
    using lfs::vis::WorkspaceFrameSnapshot;

    WorkspaceFrameSnapshot onePane(const ViewRect rect = {0, 0, 400, 300}) {
        WorkspaceFrameSnapshot snapshot;
        snapshot.outer = rect;
        snapshot.primary = 1;
        snapshot.focused = 1;
        snapshot.live_ids = {1};
        snapshot.panes.push_back(lfs::vis::PaneSnapshot{
            .id = 1,
            .rect = rect,
            .focused = true,
        });
        return snapshot;
    }

    void expectPartition(const lfs::vis::CornerSplitPreview& preview,
                         const SplitAxis axis, const int min_pane) {
        ASSERT_EQ(preview.axis, axis);
        ASSERT_FALSE(preview.first.empty());
        ASSERT_FALSE(preview.second.empty());
        if (axis == SplitAxis::Horizontal) {
            EXPECT_GE(preview.first.width, min_pane);
            EXPECT_GE(preview.second.width, min_pane);
            EXPECT_EQ(preview.first.x, 0);
            EXPECT_EQ(preview.second.right(), 400);
            EXPECT_EQ(preview.first.right(), preview.splitter.x);
            EXPECT_EQ(preview.splitter.right(), preview.second.x);
            EXPECT_EQ(preview.first.height, 300);
            EXPECT_EQ(preview.second.height, 300);
        } else {
            EXPECT_GE(preview.first.height, min_pane);
            EXPECT_GE(preview.second.height, min_pane);
            EXPECT_EQ(preview.first.y, 0);
            EXPECT_EQ(preview.second.bottom(), 300);
            EXPECT_EQ(preview.first.bottom(), preview.splitter.y);
            EXPECT_EQ(preview.splitter.bottom(), preview.second.y);
            EXPECT_EQ(preview.first.width, 400);
            EXPECT_EQ(preview.second.width, 400);
        }
    }

} // namespace

TEST(CornerSplitInteraction, HandleUsesLowerRightAndReleaseDoesNotMutateSnapshot) {
    const auto snapshot = onePane();
    CornerSplitInteraction interaction;

    EXPECT_EQ(interaction.handleHit(snapshot, {390.0f, 290.0f}), std::optional<ViewId>(1));
    EXPECT_EQ(interaction.handleHit(snapshot, {350.0f, 250.0f}), std::nullopt);
    ASSERT_TRUE(interaction.begin(snapshot, {390.0f, 290.0f}));
    EXPECT_EQ(interaction.update(snapshot, {290.0f, 275.0f})->axis, SplitAxis::Horizontal);
    const auto request = interaction.release(snapshot, {290.0f, 275.0f});
    ASSERT_TRUE(request);
    EXPECT_EQ(request->source, 1u);
    EXPECT_EQ(request->axis, SplitAxis::Horizontal);
    EXPECT_NEAR(request->ratio, 287.0f / 396.0f, 1e-6f);
    EXPECT_EQ(snapshot.panes.size(), 1u);
}

TEST(CornerSplitInteraction, HandleRemainsHitAtBorderlessWindowCorner) {
    // The workspace work region reaches the client border. WindowManager's
    // native resize affordance must defer this exact overlap to the area grip.
    const auto snapshot = onePane({0, 30, 1280, 668});
    CornerSplitInteraction interaction;
    EXPECT_EQ(interaction.handleHit(snapshot, {1275.0f, 693.0f}),
              std::optional<ViewId>(1));
}

TEST(CornerSplitInteraction, InwardXAndYDragsChooseMatchingAxisAndHonorMinimums) {
    const auto snapshot = onePane();
    const CornerSplitInteractionConfig config{.handle_radius_pixels = 18.0f,
                                              .activation_threshold_pixels = 10.0f,
                                              .min_pane_pixels = 64,
                                              .divider_pixels = 4};

    CornerSplitInteraction x(config);
    ASSERT_TRUE(x.begin(snapshot, {390.0f, 290.0f}));
    const auto x_preview = x.update(snapshot, {290.0f, 275.0f});
    ASSERT_TRUE(x_preview);
    expectPartition(*x_preview, SplitAxis::Horizontal, 64);

    CornerSplitInteraction y(config);
    ASSERT_TRUE(y.begin(snapshot, {390.0f, 290.0f}));
    const auto y_preview = y.update(snapshot, {380.0f, 180.0f});
    ASSERT_TRUE(y_preview);
    expectPartition(*y_preview, SplitAxis::Vertical, 64);
}

TEST(CornerSplitInteraction, DiagonalDragLocksDominantAxisUntilRelease) {
    const auto snapshot = onePane();
    CornerSplitInteraction interaction;
    ASSERT_TRUE(interaction.begin(snapshot, {390.0f, 290.0f}));

    const auto first = interaction.update(snapshot, {300.0f, 270.0f});
    ASSERT_TRUE(first);
    EXPECT_EQ(first->axis, SplitAxis::Horizontal);

    const auto second = interaction.update(snapshot, {370.0f, 190.0f});
    ASSERT_TRUE(second);
    EXPECT_EQ(second->axis, SplitAxis::Horizontal);
    EXPECT_TRUE(interaction.release(snapshot, {370.0f, 190.0f}));
}

TEST(CornerSplitInteraction, ReleaseBeforeThresholdCancelsWithoutRequest) {
    const auto snapshot = onePane();
    CornerSplitInteraction interaction;
    ASSERT_TRUE(interaction.begin(snapshot, {390.0f, 290.0f}));
    EXPECT_EQ(interaction.update(snapshot, {385.0f, 288.0f}), std::nullopt);
    EXPECT_EQ(interaction.release(snapshot, {385.0f, 288.0f}), std::nullopt);
    EXPECT_FALSE(interaction.active());
}

TEST(CornerSplitInteraction, ClosedOrHiddenSourceCancelsCapture) {
    auto snapshot = onePane();
    CornerSplitInteraction interaction;
    ASSERT_TRUE(interaction.begin(snapshot, {390.0f, 290.0f}));

    snapshot.panes.clear();
    snapshot.live_ids = {1}; // Retained registry identity may be hidden.
    EXPECT_TRUE(interaction.reconcile(snapshot));
    EXPECT_FALSE(interaction.active());

    snapshot = onePane();
    ASSERT_TRUE(interaction.begin(snapshot, {390.0f, 290.0f}));
    EXPECT_TRUE(interaction.cancel());
    EXPECT_FALSE(interaction.active());
}

TEST(CornerSplitInteraction, GestureHasNoFourViewLimit) {
    auto snapshot = onePane();
    for (ViewId id = 2; id <= 5; ++id) {
        auto pane = snapshot.panes.front();
        pane.id = id;
        pane.rect.x = static_cast<int>((id - 1) * 80);
        snapshot.panes.push_back(pane);
        snapshot.live_ids.push_back(id);
    }
    // The source is still the first visible pane; the interaction itself does
    // not consult a view count and leaves allocation/commit to the workspace.
    CornerSplitInteraction interaction;
    ASSERT_TRUE(interaction.begin(snapshot, {390.0f, 290.0f}));
    const auto request = interaction.release(snapshot, {250.0f, 290.0f});
    ASSERT_TRUE(request);
    EXPECT_EQ(request->source, 1u);
    EXPECT_EQ(request->axis, SplitAxis::Horizontal);
}
