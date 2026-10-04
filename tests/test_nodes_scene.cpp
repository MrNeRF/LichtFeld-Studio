/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene.hpp"
#include "io/session_chapters.hpp"

#include <gtest/gtest.h>

namespace {
    using lfs::core::Device;
    using lfs::core::Scene;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    std::unique_ptr<SplatData> splats(const size_t count, const float start) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotation(count * 4, 0.0f);
        for (size_t index = 0; index < count; ++index) {
            means[index * 3] = start + static_cast<float>(index);
            rotation[index * 4] = 1.0f;
        }
        return std::make_unique<SplatData>(
            1,
            Tensor::from_vector(std::move(means), {count, 3}, Device::CPU),
            Tensor::zeros({count, 1, 3}, Device::CPU),
            Tensor::zeros({count, 3, 3}, Device::CPU),
            Tensor::zeros({count, 3}, Device::CPU),
            Tensor::from_vector(std::move(rotation), {count, 4}, Device::CPU),
            Tensor::zeros({count, 1}, Device::CPU),
            1.0f);
    }

    std::shared_ptr<SplatData> shared_splats(const size_t count, const float start) {
        return std::shared_ptr<SplatData>(splats(count, start).release());
    }
} // namespace

TEST(NodesSceneEffectivePayload, CombinedSelectionAndSnapshotUseEvaluatedSplats) {
    Scene scene;
    const auto id = scene.addSplat("host", splats(2, 0.0f));
    ASSERT_NE(id, lfs::core::NULL_NODE);
    scene.setSelectionMask(std::make_shared<Tensor>(
        Tensor::from_vector({true, false}, {2}, Device::CPU)));

    auto evaluated = shared_splats(3, 10.0f);
    scene.setNodeEvaluatedPayload(id, evaluated);

    ASSERT_TRUE(scene.hasEvaluatedPayload(id));
    ASSERT_NE(scene.getCombinedModel(), nullptr);
    EXPECT_EQ(scene.getCombinedModel()->size(), 3);
    EXPECT_EQ(scene.getSelectionGaussianCount(), 3u);
    const auto selection = scene.getSelectionMask();
    EXPECT_TRUE(!selection || selection->count_nonzero() == 0u);

    const auto snapshots = scene.snapshotVisibleSplats();
    ASSERT_EQ(snapshots.size(), 1u);
    ASSERT_NE(snapshots.front().data, nullptr);
    EXPECT_EQ(snapshots.front().row_count, 3u);
    EXPECT_EQ(snapshots.front().data->means_raw().to_vector().front(), 10.0f);

    const auto* stored = scene.getNodeById(id)->model.get();
    ASSERT_NE(stored, nullptr);
    EXPECT_EQ(stored->size(), 2);
    EXPECT_EQ(stored->means_raw().to_vector().front(), 0.0f);

    scene.clearNodeEvaluatedPayload(id);
    EXPECT_FALSE(scene.hasEvaluatedPayload(id));
    ASSERT_NE(scene.getCombinedModel(), nullptr);
    EXPECT_EQ(scene.getCombinedModel()->size(), 2);
}

TEST(NodesProjectChapter, JsonRoundTripPreservesTreesAndStacks) {
    const nlohmann::json source = {
        {"schema_version", 1},
        {"trees", nlohmann::json::array({{{"uuid", "tree-a"}, {"name", "Tree"}}})},
        {"stacks", {{"00000000-0000-4000-8000-000000000001", nlohmann::json::array({{{"name", "Modifier"}, {"tree_uuid", "tree-a"}}})}}},
    };
    auto parsed = lfs::io::JsonChapterDom::parse(source.dump());
    ASSERT_TRUE(parsed) << lfs::format_for_developer(parsed.error());
    lfs::io::project::NodesSessionChapter chapter(std::move(*parsed));
    ASSERT_TRUE(chapter.validate());

    auto restored = lfs::io::project::NodesSessionChapter::from_bytes(chapter.to_bytes());
    ASSERT_TRUE(restored) << lfs::format_for_developer(restored.error());
    EXPECT_EQ(restored->dom().dump(), chapter.dom().dump());
}
