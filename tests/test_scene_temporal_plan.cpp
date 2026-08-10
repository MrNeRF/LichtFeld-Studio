/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_temporal_plan.hpp"

#include <gtest/gtest.h>

namespace lfs::vis {

    namespace {
        SceneTemporalPlan temporalHistoryPlan(const glm::ivec2 extent = {1920, 1080},
                                              const bool depth = true) {
            SceneUpscalerRequirements requirements;
            requirements.depth = depth;
            requirements.motion_vectors = true;
            requirements.jitter = true;
            requirements.history = true;
            return makeSceneTemporalPlan(requirements, extent, extent);
        }

        TemporalFrameInput temporalFrameInput(const glm::ivec2 extent = {1920, 1080}) {
            TemporalFrameInput input;
            input.view.size = extent;
            return input;
        }
    } // namespace

    TEST(SceneTemporalPlan, NativeAndSpatialRemainValidZeroCostPlans) {
        const auto native = makeSceneTemporalPlan(
            nativeSceneUpscalerDescriptor().requirements, {2200, 1476}, {2200, 1476});
        const auto spatial = makeSceneTemporalPlan(
            spatialSceneUpscalerDescriptor().requirements, {1100, 738}, {2200, 1476});

        EXPECT_TRUE(native.zeroCost());
        EXPECT_TRUE(spatial.zeroCost());
        EXPECT_EQ(native.render_extent, glm::ivec2(0));
        EXPECT_EQ(spatial.output_extent, glm::ivec2(0));
        EXPECT_FALSE(native.temporal());
        EXPECT_FALSE(spatial.needsHistoryColor());
    }

    TEST(SceneTemporalPlan, DeclaresOnlyRequestedTemporalInputsAndHistory) {
        SceneUpscalerRequirements requirements;
        requirements.depth = true;
        requirements.motion_vectors = true;
        requirements.jitter = true;
        requirements.history = true;
        const auto plan = makeSceneTemporalPlan(requirements, {1100, 738}, {2200, 1476});

        EXPECT_TRUE(plan.active());
        EXPECT_TRUE(plan.temporal());
        EXPECT_TRUE(plan.valid());
        EXPECT_TRUE(plan.needsHistoryColor());
        EXPECT_TRUE(plan.needsHistoryDepth());
        EXPECT_EQ(plan.render_extent, glm::ivec2(1100, 738));
        EXPECT_EQ(plan.output_extent, glm::ivec2(2200, 1476));
    }

    TEST(SceneTemporalPlan, HistoryWithoutDepthDoesNotAllocateDepthHistory) {
        SceneUpscalerRequirements requirements;
        requirements.history = true;
        const auto plan = makeSceneTemporalPlan(requirements, {1280, 720}, {1920, 1080});
        EXPECT_TRUE(plan.needsHistoryColor());
        EXPECT_FALSE(plan.needsHistoryDepth());
    }

    TEST(SceneTemporalPlan, RejectsInvalidExtentsOnlyWhenWorkIsRequested) {
        SceneUpscalerRequirements temporal;
        temporal.motion_vectors = true;
        EXPECT_FALSE(makeSceneTemporalPlan(temporal, {0, 720}, {1920, 1080}).valid());
        EXPECT_FALSE(makeSceneTemporalPlan(temporal, {1280, 720}, {0, 0}).valid());

        EXPECT_TRUE(makeSceneTemporalPlan({}, {0, 720}, {1920, 1080}).zeroCost());
    }

    TEST(SceneHistoryContract, UnavailableHistoryIsAValidZeroCostState) {
        const SceneHistoryContract history{};
        EXPECT_FALSE(history.available());
        EXPECT_FALSE(history.hasDepth());
        EXPECT_TRUE(history.valid());
        EXPECT_FALSE(history.matchesOutputExtent(0, 0));
        EXPECT_FALSE(history.matchesOutputExtent(1920, 1080));
    }

    TEST(SceneHistoryContract, DescribesColorAndOptionalDepthAtOutputExtent) {
        const auto history = makeSceneHistoryContract(
            true,
            SceneHistoryStorage::VulkanImage,
            SceneHistoryStorage::VulkanImage,
            1920,
            1080,
            42);
        EXPECT_TRUE(history.available());
        EXPECT_TRUE(history.hasDepth());
        EXPECT_TRUE(history.valid());
        EXPECT_TRUE(history.matchesOutputExtent(1920, 1080));
        EXPECT_FALSE(history.matchesOutputExtent(1280, 720));
        EXPECT_EQ(history.sequence, 42u);
    }

    TEST(SceneHistoryContract, RejectsMalformedAvailableHistory) {
        EXPECT_FALSE(makeSceneHistoryContract(
                         true,
                         SceneHistoryStorage::VulkanImage,
                         SceneHistoryStorage::None,
                         0,
                         1080,
                         1)
                         .valid());
        EXPECT_FALSE(makeSceneHistoryContract(
                         true,
                         SceneHistoryStorage::None,
                         SceneHistoryStorage::VulkanImage,
                         1920,
                         1080,
                         1)
                         .valid());
    }

    TEST(SceneHistoryTracker, FirstFrameBecomesHistoryForTheNextCompatibleFrame) {
        TemporalFrameTracker frames;
        SceneHistoryTracker histories;
        const auto plan = temporalHistoryPlan();
        const auto input = temporalFrameInput();

        const auto first = frames.prepare(TemporalViewId::Main, input);
        EXPECT_FALSE(first.history_valid);
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main, plan, first).available());
        EXPECT_TRUE(histories.commit(
            TemporalViewId::Main,
            plan,
            first,
            SceneHistoryStorage::VulkanImage,
            SceneHistoryStorage::VulkanImage));
        frames.commit(TemporalViewId::Main, input);

        const auto second = frames.prepare(TemporalViewId::Main, input);
        ASSERT_TRUE(second.history_valid);
        const auto history = histories.prepare(TemporalViewId::Main, plan, second);
        EXPECT_TRUE(history.available());
        EXPECT_TRUE(history.hasDepth());
        EXPECT_EQ(history.sequence, second.sequence);
    }

    TEST(SceneHistoryTracker, RejectsMissingRequiredColorOrDepthStorage) {
        SceneHistoryTracker histories;
        TemporalFrameState frame;
        const auto plan = temporalHistoryPlan();
        EXPECT_FALSE(histories.commit(
            TemporalViewId::Main,
            plan,
            frame,
            SceneHistoryStorage::None,
            SceneHistoryStorage::VulkanImage));
        EXPECT_FALSE(histories.commit(
            TemporalViewId::Main,
            plan,
            frame,
            SceneHistoryStorage::VulkanImage,
            SceneHistoryStorage::None));
    }

    TEST(SceneHistoryTracker, StoresNoDepthWhenThePlanDoesNotRequestIt) {
        SceneHistoryTracker histories;
        TemporalFrameState first;
        const auto plan = temporalHistoryPlan({1280, 720}, false);
        EXPECT_TRUE(histories.commit(
            TemporalViewId::Main,
            plan,
            first,
            SceneHistoryStorage::Tensor,
            SceneHistoryStorage::VulkanImage));

        TemporalFrameState second;
        second.history_valid = true;
        second.sequence = 1;
        const auto history = histories.prepare(TemporalViewId::Main, plan, second);
        EXPECT_TRUE(history.available());
        EXPECT_FALSE(history.hasDepth());
    }

    TEST(SceneHistoryTracker, InvalidatesIncompatibleExtentOrFrameHistory) {
        SceneHistoryTracker histories;
        const auto plan = temporalHistoryPlan();
        TemporalFrameState first;
        ASSERT_TRUE(histories.commit(
            TemporalViewId::Main,
            plan,
            first,
            SceneHistoryStorage::VulkanImage,
            SceneHistoryStorage::VulkanImage));

        TemporalFrameState next;
        next.history_valid = true;
        next.sequence = 1;
        EXPECT_FALSE(histories.prepare(
                                  TemporalViewId::Main,
                                  temporalHistoryPlan({1280, 720}),
                                  next)
                         .available());
        next.sequence = 2;
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main, plan, next).available());
        next.sequence = 1;
        next.history_valid = false;
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main, plan, next).available());
    }

    TEST(SceneHistoryTracker, KeepsViewsIndependentAndSupportsExplicitReset) {
        SceneHistoryTracker histories;
        const auto plan = temporalHistoryPlan();
        TemporalFrameState first;
        ASSERT_TRUE(histories.commit(
            TemporalViewId::SplitLeft,
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

        ASSERT_TRUE(histories.commit(
            TemporalViewId::Main,
            plan,
            first,
            SceneHistoryStorage::VulkanImage,
            SceneHistoryStorage::VulkanImage));
        histories.resetAll();
        EXPECT_FALSE(histories.prepare(TemporalViewId::Main, plan, next).available());
    }

} // namespace lfs::vis
