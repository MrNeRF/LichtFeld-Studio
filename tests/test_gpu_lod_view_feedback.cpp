/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/gpu_lod_view_feedback.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace {

    using lfs::vis::kLegacyMainOutputKey;
    using lfs::vis::kLegacyPreviewOutputKey;
    using lfs::vis::kLegacySplitLeftOutputKey;
    using lfs::vis::kLegacySplitRightOutputKey;
    using lfs::vis::sceneOutputKey;
    using lfs::vis::ViewOutputKey;
    using lfs::vis::detail::GpuLodSelectorSample;
    using lfs::vis::detail::GpuLodViewFeedbackTable;

    GpuLodSelectorSample sampleWith(const std::vector<std::uint32_t>& protected_chunks,
                                    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& misses,
                                    const std::size_t candidates = 100) {
        GpuLodSelectorSample sample;
        sample.threshold_scale = 1.0f;
        sample.candidate_count = candidates;
        sample.rendered_capacity = 1000;
        sample.overflow_count = 0;
        sample.protected_chunks = protected_chunks;
        sample.miss_candidates = misses;
        return sample;
    }

    TEST(GpuLodViewFeedback, OutOfOrderCompletionsStayOnOriginatingKey) {
        GpuLodViewFeedbackTable table;
        table.setSceneGeneration(7, 3);
        const auto key_a = sceneOutputKey(11);
        const auto key_b = sceneOutputKey(22);
        table.noteLiveRender(key_a, 1000, true);
        table.noteLiveRender(key_b, 800, true);

        ASSERT_TRUE(table.applyTaggedSample(
            key_b, 7, 3, sampleWith({20, 21}, {{30, 50}}, 200)));
        ASSERT_TRUE(table.applyTaggedSample(
            key_a, 7, 3, sampleWith({10}, {{11, 9}}, 50)));

        const auto* view_a = table.find(key_a);
        const auto* view_b = table.find(key_b);
        ASSERT_NE(view_a, nullptr);
        ASSERT_NE(view_b, nullptr);
        EXPECT_EQ(view_a->last_candidate_count, 50u);
        EXPECT_EQ(view_b->last_candidate_count, 200u);
        ASSERT_EQ(view_a->demand.protected_chunks, std::vector<std::uint32_t>({10}));
        ASSERT_EQ(view_b->demand.protected_chunks, (std::vector<std::uint32_t>{20, 21}));
        ASSERT_EQ(view_a->demand.prefetch_requests.size(), 1u);
        EXPECT_EQ(view_a->demand.prefetch_requests[0].chunk, 11u);
        ASSERT_EQ(view_b->demand.prefetch_requests.size(), 1u);
        EXPECT_EQ(view_b->demand.prefetch_requests[0].chunk, 30u);
    }

    TEST(GpuLodViewFeedback, ClosedKeyDiscardsLateCompletion) {
        GpuLodViewFeedbackTable table;
        table.setSceneGeneration(1, 1);
        const auto key_a = sceneOutputKey(4);
        const auto key_b = sceneOutputKey(5);
        table.noteLiveRender(key_a, 100, true);
        table.noteLiveRender(key_b, 100, true);
        ASSERT_TRUE(table.applyTaggedSample(key_a, 1, 1, sampleWith({1}, {{2, 8}})));
        ASSERT_TRUE(table.applyTaggedSample(key_b, 1, 1, sampleWith({3}, {{4, 7}})));
        table.retire(key_a);

        EXPECT_EQ(table.find(key_a), nullptr);
        EXPECT_FALSE(table.applyTaggedSample(key_a, 1, 1, sampleWith({99}, {{100, 1}})));

        const auto unified = table.unionLiveDemand();
        ASSERT_TRUE(unified.valid);
        EXPECT_EQ(unified.protected_chunks, std::vector<std::uint32_t>({3}));
        ASSERT_EQ(unified.prefetch_requests.size(), 1u);
        EXPECT_EQ(unified.prefetch_requests[0].chunk, 4u);
        EXPECT_TRUE(std::none_of(unified.protected_chunks.begin(),
                                 unified.protected_chunks.end(),
                                 [](std::uint32_t chunk) { return chunk == 1u || chunk == 99u; }));
    }

    TEST(GpuLodViewFeedback, UnionReplacesViewDemandAndKeepsNeighbor) {
        GpuLodViewFeedbackTable table;
        table.setSceneGeneration(2, 9);
        const auto key_a = kLegacyMainOutputKey;
        const auto key_b = kLegacySplitLeftOutputKey;
        table.noteLiveRender(key_a, 64, true);
        table.noteLiveRender(key_b, 64, true);

        ASSERT_TRUE(table.applyTaggedSample(
            key_a, 2, 9, sampleWith({1, 2}, {{10, 1}, {11, 4}})));
        ASSERT_TRUE(table.applyTaggedSample(
            key_b, 2, 9, sampleWith({2, 3}, {{11, 2}, {12, 9}})));

        auto unified = table.unionLiveDemand();
        ASSERT_TRUE(unified.valid);
        EXPECT_EQ(unified.protected_chunks, (std::vector<std::uint32_t>{1, 2, 3}));
        ASSERT_EQ(unified.prefetch_requests.size(), 3u);
        EXPECT_EQ(unified.prefetch_requests[0].chunk, 12u);
        EXPECT_EQ(unified.prefetch_requests[0].priority, 9u);
        auto chunk_11 = std::find_if(unified.prefetch_requests.begin(),
                                     unified.prefetch_requests.end(),
                                     [](const auto& request) { return request.chunk == 11u; });
        ASSERT_NE(chunk_11, unified.prefetch_requests.end());
        EXPECT_EQ(chunk_11->priority, 4u);

        ASSERT_TRUE(table.applyTaggedSample(
            key_a, 2, 9, sampleWith({8}, {{20, 3}})));
        unified = table.unionLiveDemand();
        EXPECT_EQ(unified.protected_chunks, (std::vector<std::uint32_t>{2, 3, 8}));
        EXPECT_TRUE(std::none_of(unified.protected_chunks.begin(),
                                 unified.protected_chunks.end(),
                                 [](std::uint32_t chunk) { return chunk == 1u; }));
        EXPECT_TRUE(std::none_of(unified.prefetch_requests.begin(),
                                 unified.prefetch_requests.end(),
                                 [](const auto& request) { return request.chunk == 10u; }));
        auto remaining_11 = std::find_if(unified.prefetch_requests.begin(),
                                         unified.prefetch_requests.end(),
                                         [](const auto& request) { return request.chunk == 11u; });
        ASSERT_NE(remaining_11, unified.prefetch_requests.end());
        EXPECT_EQ(remaining_11->priority, 2u);
        EXPECT_TRUE(std::any_of(unified.prefetch_requests.begin(),
                                unified.prefetch_requests.end(),
                                [](const auto& request) { return request.chunk == 12u; }));
        EXPECT_TRUE(std::any_of(unified.prefetch_requests.begin(),
                                unified.prefetch_requests.end(),
                                [](const auto& request) { return request.chunk == 20u; }));
    }

    TEST(GpuLodViewFeedback, StaleGenerationAndUnknownKeyAreDropped) {
        GpuLodViewFeedbackTable table;
        table.setSceneGeneration(4, 5);
        const auto key = kLegacyPreviewOutputKey;
        table.noteLiveRender(key, 32, true);
        EXPECT_FALSE(table.applyTaggedSample(key, 3, 5, sampleWith({1}, {})));
        EXPECT_FALSE(table.applyTaggedSample(key, 4, 4, sampleWith({1}, {})));
        EXPECT_FALSE(table.applyTaggedSample(kLegacySplitRightOutputKey, 4, 5, sampleWith({1}, {})));
        EXPECT_FALSE(table.unionLiveDemand().valid);

        table.setSceneGeneration(6, 5);
        EXPECT_EQ(table.find(key), nullptr);
        EXPECT_EQ(table.lastRenderedKey(), lfs::vis::kInvalidViewOutputKey);
    }

    TEST(GpuLodViewFeedback, UnionDoesNotTruncateFrontier) {
        GpuLodViewFeedbackTable table;
        table.setSceneGeneration(1, 1);
        const auto key_a = sceneOutputKey(1);
        const auto key_b = sceneOutputKey(2);
        table.noteLiveRender(key_a, 1, true);
        table.noteLiveRender(key_b, 1, true);

        std::vector<std::uint32_t> protected_a;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> misses_a;
        protected_a.reserve(64);
        misses_a.reserve(64);
        for (std::uint32_t i = 0; i < 64; ++i) {
            protected_a.push_back(i);
            misses_a.emplace_back(1000 + i, 64 - i);
        }
        std::vector<std::uint32_t> protected_b;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> misses_b;
        for (std::uint32_t i = 50; i < 90; ++i) {
            protected_b.push_back(i);
            misses_b.emplace_back(2000 + i, i);
        }
        ASSERT_TRUE(table.applyTaggedSample(key_a, 1, 1, sampleWith(protected_a, misses_a)));
        ASSERT_TRUE(table.applyTaggedSample(key_b, 1, 1, sampleWith(protected_b, misses_b)));

        const auto unified = table.unionLiveDemand();
        EXPECT_EQ(unified.protected_chunks.size(), 90u);
        EXPECT_EQ(unified.prefetch_requests.size(), 64u + 40u);
        EXPECT_EQ(unified.protected_chunks.front(), 0u);
        EXPECT_EQ(unified.protected_chunks.back(), 89u);
    }

    TEST(GpuLodViewFeedback, HiddenWorkspaceDemandIsExcludedUntilViewIsVisible) {
        GpuLodViewFeedbackTable table;
        table.setSceneGeneration(8, 2);
        const auto visible = sceneOutputKey(41);
        const auto hidden = sceneOutputKey(42);
        table.noteLiveRender(visible, 100, true);
        table.noteLiveRender(hidden, 100, true);
        ASSERT_TRUE(table.applyTaggedSample(visible, 8, 2, sampleWith({1}, {{11, 4}})));
        ASSERT_TRUE(table.applyTaggedSample(hidden, 8, 2, sampleWith({2}, {{22, 9}})));

        const std::array<ViewOutputKey, 1> visible_keys{visible};
        table.setVisibleKeys(visible_keys);
        auto unified = table.unionLiveDemand();
        ASSERT_TRUE(unified.valid);
        EXPECT_EQ(unified.protected_chunks, std::vector<std::uint32_t>({1}));
        ASSERT_EQ(unified.prefetch_requests.size(), 1u);
        EXPECT_EQ(unified.prefetch_requests.front().chunk, 11u);

        // Restoring the pane must retain its cached feedback and demand.
        const std::array<ViewOutputKey, 2> restored_keys{visible, hidden};
        table.setVisibleKeys(restored_keys);
        unified = table.unionLiveDemand();
        EXPECT_EQ(unified.protected_chunks, (std::vector<std::uint32_t>{1, 2}));
        ASSERT_EQ(unified.prefetch_requests.size(), 2u);
        EXPECT_EQ(unified.prefetch_requests.front().chunk, 22u);
    }

} // namespace
