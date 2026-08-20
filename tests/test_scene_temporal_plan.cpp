/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_temporal_plan.hpp"

#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {
        constexpr SceneTemporalRequirements temporalRequirements() {
            return {
                .depth = true,
                .motion = true,
                .jitter = true,
                .history_color = true,
                .history_depth = true,
            };
        }

        SceneTemporalPlan temporalPlan(const glm::ivec2 render = {1280, 720},
                                       const glm::ivec2 output = {1920, 1080}) {
            return makeSceneTemporalPlan(temporalRequirements(), render, output);
        }
    } // namespace

    TEST(SceneTemporalPlan, DisabledRequirementsProduceCanonicalZeroCostPlan) {
        const auto plan = makeSceneTemporalPlan({}, {1280, 720}, {1920, 1080});
        EXPECT_TRUE(plan.zeroCost());
        EXPECT_EQ(plan.render_extent, glm::ivec2(0));
        EXPECT_EQ(plan.output_extent, glm::ivec2(0));
    }

    TEST(SceneTemporalPlan, ActivePlanPreservesDistinctRenderAndOutputExtents) {
        const auto plan = temporalPlan();
        EXPECT_TRUE(plan.active());
        EXPECT_TRUE(plan.valid());
        EXPECT_EQ(plan.render_extent, glm::ivec2(1280, 720));
        EXPECT_EQ(plan.output_extent, glm::ivec2(1920, 1080));
    }

    TEST(SceneTemporalPlan, RejectsInvalidRequirementAndExtentCombinations) {
        auto requirements = temporalRequirements();
        requirements.history_color = false;
        EXPECT_FALSE(makeSceneTemporalPlan(requirements, {1280, 720}, {1920, 1080}).valid());
        EXPECT_FALSE(makeSceneTemporalPlan(temporalRequirements(), {0, 720}, {1920, 1080})
                         .valid());
        EXPECT_FALSE(makeSceneTemporalPlan(temporalRequirements(), {1280, 720}, {0, 1080})
                         .valid());
    }

    TEST(SceneHistoryContract, UnavailableHistoryIsCanonicalAndValid) {
        const SceneHistoryContract history;
        EXPECT_FALSE(history.available());
        EXPECT_FALSE(history.hasDepth());
        EXPECT_TRUE(history.valid());
        EXPECT_FALSE(history.matches(temporalPlan()));
    }

    TEST(SceneHistoryTracker, CommitsOutputColorAndRenderDepthForNextSequence) {
        SceneHistoryTracker histories;
        const auto plan = temporalPlan();
        TemporalFrameState first;
        ASSERT_TRUE(histories.commit(TemporalViewId::Main,
                                     plan,
                                     first,
                                     SceneHistoryStorage::VulkanImage,
                                     SceneHistoryStorage::VulkanImage));

        TemporalFrameState next;
        next.history_valid = true;
        next.sequence = 1;
        const auto history = histories.prepare(TemporalViewId::Main, plan, next);
        EXPECT_TRUE(history.available());
        EXPECT_TRUE(history.hasDepth());
        EXPECT_EQ(history.color_extent, plan.output_extent);
        EXPECT_EQ(history.depth_extent, plan.render_extent);
    }

    TEST(SceneHistoryTracker, MissingRequiredStorageFailsClosedAndClearsHistory) {
        SceneHistoryTracker histories;
        const auto plan = temporalPlan();
        TemporalFrameState first;
        ASSERT_TRUE(histories.commit(TemporalViewId::Main,
                                     plan,
                                     first,
                                     SceneHistoryStorage::VulkanImage,
                                     SceneHistoryStorage::VulkanImage));
        EXPECT_FALSE(histories.commit(TemporalViewId::Main,
                                      plan,
                                      first,
                                      SceneHistoryStorage::VulkanImage,
                                      SceneHistoryStorage::None));

        TemporalFrameState next;
        next.history_valid = true;
        next.sequence = 1;
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main, plan, next).available());
    }

    TEST(SceneHistoryTracker, RejectsWrongExtentSequenceAndInvalidFrameHistory) {
        SceneHistoryTracker histories;
        const auto plan = temporalPlan();
        TemporalFrameState first;
        ASSERT_TRUE(histories.commit(TemporalViewId::Main,
                                     plan,
                                     first,
                                     SceneHistoryStorage::Tensor,
                                     SceneHistoryStorage::Tensor));

        TemporalFrameState next;
        next.history_valid = true;
        next.sequence = 1;
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main,
                                       temporalPlan({960, 540}, {1920, 1080}),
                                       next)
                         .available());
        next.sequence = 2;
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main, plan, next).available());
        next.sequence = 1;
        next.history_valid = false;
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main, plan, next).available());
    }

    TEST(SceneHistoryTracker, ViewsAreIndependentAndExplicitlyResettable) {
        SceneHistoryTracker histories;
        const auto plan = temporalPlan();
        TemporalFrameState first;
        ASSERT_TRUE(histories.commit(TemporalViewId::SplitLeft,
                                     plan,
                                     first,
                                     SceneHistoryStorage::VulkanImage,
                                     SceneHistoryStorage::VulkanImage));
        TemporalFrameState next;
        next.history_valid = true;
        next.sequence = 1;
        EXPECT_TRUE(histories.prepare(TemporalViewId::SplitLeft, plan, next).available());
        EXPECT_FALSE(histories.prepare(TemporalViewId::SplitRight, plan, next).available());
        histories.reset(TemporalViewId::SplitLeft);
        EXPECT_FALSE(histories.prepare(TemporalViewId::SplitLeft, plan, next).available());
    }

} // namespace lfs::vis
