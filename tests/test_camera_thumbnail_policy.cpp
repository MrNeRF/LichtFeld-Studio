/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/camera_thumbnail_batch.hpp"
#include "gui/camera_thumbnail_policy.hpp"
#include "gui/frustum_overlay_key.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

TEST(CameraThumbnailPolicyTest, InvertedVisibleRowRangeIsEmpty) {
    const std::optional<std::pair<std::size_t, std::size_t>> range =
        lfs::vis::gui::cameraThumbnailVisibleRowRange(8, 3, 12);

    EXPECT_FALSE(range.has_value());
}

TEST(CameraThumbnailPolicyTest, RequestsAllCamerasSelectedFirst) {
    const std::vector<int> all = {10, 11, 12, 13};
    const std::vector<int> visible = {10, 11, 12, 12};
    const std::vector<int> selected = {12, 13, 99};

    const auto requested = lfs::vis::gui::cameraThumbnailRequestOrder(all, visible, selected);

    EXPECT_EQ(requested, (std::vector<int>{12, 13, 10, 11}));
}

TEST(CameraThumbnailPolicyTest, NullCameraPlaceholderIsNotRequested) {
    const std::vector<int> all = {10, -1, 11};
    const std::vector<int> visible;
    const std::vector<int> selected;

    const auto requested = lfs::vis::gui::cameraThumbnailRequestOrder(all, visible, selected);

    EXPECT_EQ(requested, (std::vector<int>{10, 11}));
    EXPECT_EQ(std::count(requested.begin(), requested.end(), -1), 0);
}

TEST(CameraThumbnailPolicyTest, ScrollingReprioritizesWithoutDroppingCameras) {
    const std::vector<int> all = {10, 11, 12, 40, 41};
    const std::vector<int> first_window = {10, 11, 12};
    const std::vector<int> second_window = {40, 41};
    const std::vector<int> no_selection;

    const auto first = lfs::vis::gui::cameraThumbnailRequestOrder(all, first_window, no_selection);
    const auto second = lfs::vis::gui::cameraThumbnailRequestOrder(all, second_window, no_selection);

    EXPECT_EQ(first, (std::vector<int>{10, 11, 12, 40, 41}));
    EXPECT_EQ(second, (std::vector<int>{40, 41, 10, 11, 12}));
}

TEST(FrustumOverlayInputKeyTest, RebuildsIffAKeyComponentChanged) {
    const lfs::vis::gui::FrustumOverlayInputKey baseline{
        .scene_identity = 1,
        .camera_list_generation = 2,
        .scene_render_generation = 3,
        .view_projection_hash = 4,
        .hovered_camera_id = 5,
        .selected_set_generation = 6,
        .training_loss_color_generation = 7,
        .thumbnail_atlas_generation = 8,
        .overlay_settings_hash = 9,
    };

    EXPECT_FALSE(lfs::vis::gui::frustumOverlayNeedsRebuild(baseline, baseline));

    auto changed = baseline;
    const auto check = [&baseline, &changed](auto&& mutate) {
        changed = baseline;
        mutate(changed);
        EXPECT_TRUE(lfs::vis::gui::frustumOverlayNeedsRebuild(baseline, changed));
    };
    check([](auto& key) { ++key.scene_identity; });
    check([](auto& key) { ++key.camera_list_generation; });
    check([](auto& key) { ++key.scene_render_generation; });
    check([](auto& key) { ++key.view_projection_hash; });
    check([](auto& key) { ++key.hovered_camera_id; });
    check([](auto& key) { ++key.selected_set_generation; });
    check([](auto& key) { ++key.training_loss_color_generation; });
    check([](auto& key) { ++key.thumbnail_atlas_generation; });
    check([](auto& key) { ++key.overlay_settings_hash; });
}

TEST(CameraThumbnailBatchTest, GroupsPagesWithinByteBudgetAndAdvancesGenerationPerBatch) {
    const std::vector<lfs::vis::gui::CameraThumbnailUploadCandidate> candidates = {
        {.page_index = 0, .byte_size = 2},
        {.page_index = 1, .byte_size = 2},
        {.page_index = 0, .byte_size = 2},
        {.page_index = 1, .byte_size = 2},
    };

    const auto batches = lfs::vis::gui::batchCameraThumbnailUploads(candidates, 6);

    ASSERT_EQ(batches.size(), 2);
    EXPECT_EQ(batches[0].page_index, 0);
    EXPECT_EQ(batches[0].candidate_indices, (std::vector<std::size_t>{0, 2}));
    EXPECT_EQ(batches[0].byte_size, 4);
    EXPECT_EQ(batches[1].page_index, 1);
    EXPECT_EQ(batches[1].candidate_indices, (std::vector<std::size_t>{1}));
    EXPECT_EQ(batches[1].byte_size, 2);

    std::size_t uploaded_bytes = 0;
    std::size_t atlas_generation = 11;
    for (const auto& batch : batches) {
        uploaded_bytes += batch.byte_size;
        ++atlas_generation;
    }
    EXPECT_LE(uploaded_bytes, 6);
    EXPECT_EQ(atlas_generation, 13);
}
