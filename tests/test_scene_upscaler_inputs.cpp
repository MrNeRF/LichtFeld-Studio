/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/passes/vulkan_scene_depth_history_contract.hpp"
#include "visualizer/rendering/scene_upscaler_inputs.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {
    namespace {
        SceneTemporalPlan plan() {
            return makeSceneTemporalPlan({.depth = true,
                                          .motion = true,
                                          .jitter = true,
                                          .history_color = true,
                                          .history_depth = true},
                                         {1280, 720},
                                         {1920, 1080});
        }

        SceneUpscalerInputs inputs() {
            return {
                .depth = makeSceneDepthContract(true,
                                                SceneDepthStorage::VulkanImage,
                                                SceneDepthEncoding::VulkanNdc,
                                                {1280, 720},
                                                0.1f,
                                                1000.0f,
                                                false,
                                                false),
                .motion = makeSceneMotionContract(
                    true,
                    SceneMotionStorage::VulkanImage,
                    SceneMotionEncoding::PixelDisplacement,
                    SceneMotionDirection::CurrentToPrevious,
                    1280,
                    720,
                    true,
                    false),
                .history = {
                    .color_storage = SceneHistoryStorage::VulkanImage,
                    .depth_storage = SceneHistoryStorage::VulkanImage,
                    .color_extent = {1920, 1080},
                    .depth_extent = {1280, 720},
                    .sequence = 1,
                },
                .jitter_applied = true,
                .history_expected = true,
            };
        }
    } // namespace

    TEST(SceneDepthContract, UnavailableDepthIsCanonicalZeroCostState) {
        const SceneDepthContract depth;
        EXPECT_TRUE(depth.valid());
        EXPECT_FALSE(depth.available());
        EXPECT_FALSE(depth.requiresLinearization());
    }

    TEST(SceneDepthContract, DescribesLinearAndVulkanDepthPrecisely) {
        const auto linear = makeSceneDepthContract(true,
                                                   SceneDepthStorage::Tensor,
                                                   SceneDepthEncoding::LinearView,
                                                   {1280, 720},
                                                   0.1f,
                                                   1000.0f,
                                                   false,
                                                   true);
        EXPECT_TRUE(linear.valid());
        EXPECT_TRUE(linear.matchesRenderExtent({1280, 720}));
        EXPECT_FALSE(linear.requiresLinearization());
        EXPECT_TRUE(linear.flip_y);

        auto ndc = linear;
        ndc.encoding = SceneDepthEncoding::VulkanNdc;
        ndc.storage = SceneDepthStorage::VulkanImage;
        EXPECT_TRUE(ndc.requiresLinearization());
    }

    TEST(SceneDepthContract, RejectsMalformedOrNonCanonicalContracts) {
        EXPECT_FALSE(makeSceneDepthContract(true,
                                            SceneDepthStorage::None,
                                            SceneDepthEncoding::LinearView,
                                            {1280, 720},
                                            0.1f,
                                            1000.0f,
                                            false,
                                            false)
                         .valid());
        EXPECT_FALSE(makeSceneDepthContract(true,
                                            SceneDepthStorage::Tensor,
                                            SceneDepthEncoding::LinearView,
                                            {0, 720},
                                            0.1f,
                                            1000.0f,
                                            false,
                                            false)
                         .valid());
        EXPECT_FALSE(makeSceneDepthContract(true,
                                            SceneDepthStorage::Tensor,
                                            SceneDepthEncoding::LinearView,
                                            {1280, 720},
                                            std::numeric_limits<float>::quiet_NaN(),
                                            1000.0f,
                                            false,
                                            false)
                         .valid());
    }

    TEST(SceneDepthContract, ConvertsLinearPerspectiveAndOrthographicDepthToVulkanNdc) {
        auto depth = makeSceneDepthContract(true,
                                            SceneDepthStorage::VulkanImage,
                                            SceneDepthEncoding::LinearView,
                                            {1280, 720},
                                            0.1f,
                                            1000.0f,
                                            false,
                                            false);
        const auto perspective = sceneDepthToVulkanNdc(10.0f, depth);
        ASSERT_TRUE(perspective.has_value());
        EXPECT_NEAR(*perspective, 1000.0f / 999.9f * 0.99f, 1e-6f);

        depth.orthographic = true;
        const auto orthographic = sceneDepthToVulkanNdc(500.05f, depth);
        ASSERT_TRUE(orthographic.has_value());
        EXPECT_NEAR(*orthographic, 0.5f, 1e-6f);
        EXPECT_FALSE(sceneDepthToVulkanNdc(0.0f, depth).has_value());

        depth.encoding = SceneDepthEncoding::VulkanNdc;
        EXPECT_EQ(sceneDepthToVulkanNdc(0.25f, depth), 0.25f);
        EXPECT_FALSE(sceneDepthToVulkanNdc(1.0f, depth).has_value());
    }

    TEST(SceneUpscalerInputs, ZeroCostPlanRequiresNoResources) {
        EXPECT_TRUE(validateSceneUpscalerInputs(makeSceneTemporalPlan({}, {}, {}), {}).valid());
    }

    TEST(SceneUpscalerInputs, FirstFrameMayInitializeHistoryFromCurrentColor) {
        auto first = inputs();
        first.history = {};
        first.history_expected = false;
        EXPECT_TRUE(validateSceneUpscalerInputs(plan(), first).valid());
        first.history_expected = true;
        EXPECT_TRUE(hasSceneUpscalerInputIssue(
            validateSceneUpscalerInputs(plan(), first).issues,
            SceneUpscalerInputIssue::MissingHistory));
    }

    TEST(SceneUpscalerInputs, AcceptsCompleteCanonicalInputs) {
        EXPECT_TRUE(validateSceneUpscalerInputs(plan(), inputs()).valid());
    }

    TEST(SceneUpscalerInputs, ReportsAllIndependentMissingInputs) {
        SceneUpscalerInputs missing;
        missing.history_expected = true;
        const auto result = validateSceneUpscalerInputs(plan(), missing);
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::MissingDepth));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::MissingMotion));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::MissingJitter));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::MissingHistory));
    }

    TEST(SceneUpscalerInputs, RejectsExtentDirectionAndJitterMismatches) {
        auto mismatched = inputs();
        mismatched.depth.width = 1279;
        mismatched.motion.height = 719;
        mismatched.motion.direction = SceneMotionDirection::PreviousToCurrent;
        mismatched.motion.includes_jitter = false;
        mismatched.history.depth_extent = {1279, 720};
        const auto result = validateSceneUpscalerInputs(plan(), mismatched);
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::DepthExtent));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::MotionExtent));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::MotionDirection));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(
            result.issues, SceneUpscalerInputIssue::MotionJitterMismatch));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(result.issues,
                                               SceneUpscalerInputIssue::HistoryExtent));
    }

    TEST(VulkanSceneDepthHistoryContract, DisabledAndMalformedRequestsRemainZeroCost) {
        VulkanSceneDepthHistoryParams params;
        EXPECT_FALSE(canRecordVulkanSceneDepthHistory(params));
        params.enabled = true;
        params.current_depth_view =
            reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(1));
        params.depth = makeSceneDepthContract(true,
                                              SceneDepthStorage::VulkanImage,
                                              SceneDepthEncoding::LinearView,
                                              {1280, 720},
                                              0.1f,
                                              1000.0f,
                                              false,
                                              false);
        EXPECT_TRUE(canRecordVulkanSceneDepthHistory(params));
        params.view = TemporalViewId::Count;
        EXPECT_FALSE(canRecordVulkanSceneDepthHistory(params));
    }

    TEST(VulkanSceneDepthHistoryContract, AcceptsLinearAndNdcVulkanDepthOnly) {
        VulkanSceneDepthHistoryParams params;
        params.enabled = true;
        params.current_depth_view =
            reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(1));
        params.depth = makeSceneDepthContract(true,
                                              SceneDepthStorage::VulkanImage,
                                              SceneDepthEncoding::LinearView,
                                              {1280, 720},
                                              0.1f,
                                              1000.0f,
                                              false,
                                              false);
        EXPECT_TRUE(canRecordVulkanSceneDepthHistory(params));
        EXPECT_EQ(sceneDepthHistoryEncodingCode(params.depth), 1u);
        const auto history = makeVulkanSceneDepthHistoryContract(params);
        EXPECT_TRUE(history.valid());
        EXPECT_EQ(history.encoding, SceneDepthEncoding::LinearView);
        EXPECT_EQ(history.storage, SceneDepthStorage::VulkanImage);
        EXPECT_FALSE(history.flip_y);
        params.depth.encoding = SceneDepthEncoding::VulkanNdc;
        EXPECT_TRUE(canRecordVulkanSceneDepthHistory(params));
        EXPECT_EQ(sceneDepthHistoryEncodingCode(params.depth), 2u);
        params.depth.orthographic = true;
        EXPECT_EQ(sceneDepthHistoryEncodingCode(params.depth), 3u);
        params.depth.storage = SceneDepthStorage::Tensor;
        EXPECT_FALSE(canRecordVulkanSceneDepthHistory(params));
    }

    TEST(VulkanSceneDepthHistoryContract, PaddedUvTransformIsBoundedAndRejectsOverflow) {
        const auto transform = sceneDepthHistoryUvTransform({1100, 738}, {1152, 768});
        ASSERT_TRUE(transform.has_value());
        EXPECT_NEAR(transform->x, 1100.0f / 1152.0f, 1e-6f);
        EXPECT_NEAR(transform->y, 738.0f / 768.0f, 1e-6f);
        EXPECT_LT(transform->z, transform->x);
        EXPECT_LT(transform->w, transform->y);
        EXPECT_FALSE(sceneDepthHistoryUvTransform({1200, 738}, {1152, 768}).has_value());
    }

} // namespace lfs::vis
