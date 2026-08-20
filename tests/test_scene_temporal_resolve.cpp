/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/passes/vulkan_scene_temporal_pipeline.hpp"
#include "visualizer/rendering/passes/vulkan_scene_temporal_resolve_pass.hpp"
#include "visualizer/rendering/scene_temporal_resolve.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {
    namespace {
        SceneTemporalResolveSample sample() {
            return {
                .current = {0.2f, 0.4f, 0.6f, 0.7f},
                .history = {0.4f, 0.6f, 0.8f, 0.1f},
                .neighborhood_min = {0.1f, 0.2f, 0.3f},
                .neighborhood_max = {0.7f, 0.8f, 0.9f},
                .current_pixel_center = {639.5f, 359.5f},
                .current_to_previous_pixels = {0.0f, 0.0f},
                .motion_extent = {1280, 720},
                .output_extent = {1280, 720},
                .current_linear_depth = 10.0f,
                .history_linear_depth = 10.0f,
                .history_valid = true,
                .depth_available = true,
            };
        }

        VulkanSceneTemporalPipelineRequest pipelineRequest() {
            VulkanSceneTemporalPipelineRequest request;
            request.temporal.requirements = {
                .depth = true,
                .motion = true,
                .jitter = true,
                .history_color = true,
                .history_depth = true};
            request.temporal.render_extent = {640, 360};
            request.temporal.output_extent = {1280, 720};
            request.temporal.frame.view.size = request.temporal.render_extent;
            request.motion.enabled = true;
            request.motion.depth_view = reinterpret_cast<VkImageView>(1);
            request.motion.depth = makeSceneDepthContract(true,
                                                          SceneDepthStorage::VulkanImage,
                                                          SceneDepthEncoding::VulkanNdc,
                                                          {640, 360},
                                                          0.1f,
                                                          1000.0f,
                                                          false,
                                                          false);
            request.motion.render_extent = {640, 360};
            request.motion.includes_jitter = true;
            request.resolve.enabled = true;
            request.resolve.current_color_view = reinterpret_cast<VkImageView>(2);
            request.resolve.render_extent = {640, 360};
            request.resolve.output_extent = {1280, 720};
            request.resolve.current_depth.enabled = true;
            request.resolve.current_depth.current_depth_view = request.motion.depth_view;
            request.resolve.current_depth.depth = request.motion.depth;
            return request;
        }
    } // namespace

    TEST(SceneTemporalResolve, QualitySettingsRemainOrderedAndUnknownIsBalanced) {
        const auto performance = sceneTemporalQualitySettings(SceneTemporalQuality::Performance);
        const auto balanced = sceneTemporalQualitySettings(SceneTemporalQuality::Balanced);
        const auto quality = sceneTemporalQualitySettings(SceneTemporalQuality::Quality);
        const auto unknown = sceneTemporalQualitySettings(static_cast<SceneTemporalQuality>(255));
        EXPECT_LT(performance.history_weight, balanced.history_weight);
        EXPECT_LT(balanced.history_weight, quality.history_weight);
        EXPECT_GT(performance.depth_relative_threshold, balanced.depth_relative_threshold);
        EXPECT_GT(balanced.depth_relative_threshold, quality.depth_relative_threshold);
        EXPECT_EQ(unknown.history_weight, balanced.history_weight);
    }

    TEST(SceneTemporalResolve, StableSampleUsesClampedHistoryAndPreservesCurrentAlpha) {
        auto stable = sample();
        stable.history = {2.0f, 0.6f, -1.0f, 0.1f};
        const auto result = resolveSceneTemporalSample(stable);
        EXPECT_TRUE(result.usedHistory());
        EXPECT_NEAR(result.color.x, 0.65f, 1e-6f);
        EXPECT_NEAR(result.color.y, 0.58f, 1e-6f);
        EXPECT_NEAR(result.color.z, 0.33f, 1e-6f);
        EXPECT_FLOAT_EQ(result.color.a, stable.current.a);
    }

    TEST(SceneTemporalResolve, PixelCentersMapToUnambiguousHistoryUv) {
        auto center = sample();
        center.current_pixel_center = {0.5f, 0.5f};
        const auto first = resolveSceneTemporalSample(center);
        EXPECT_NEAR(first.previous_uv.x, 0.5f / 1280.0f, 1e-7f);
        EXPECT_NEAR(first.previous_uv.y, 0.5f / 720.0f, 1e-7f);

        center.current_pixel_center = {1279.5f, 719.5f};
        EXPECT_TRUE(resolveSceneTemporalSample(center).usedHistory());
    }

    TEST(SceneTemporalResolve, RejectsAbsentHistoryAndOutOfBoundsReprojection) {
        auto current = sample();
        current.history_valid = false;
        EXPECT_EQ(resolveSceneTemporalSample(current).rejection,
                  SceneHistoryRejection::NoHistory);
        current = sample();
        current.current_pixel_center = {0.5f, 0.5f};
        current.current_to_previous_pixels = {-0.01f, 0.0f};
        EXPECT_EQ(resolveSceneTemporalSample(current).rejection,
                  SceneHistoryRejection::OutsideHistory);
    }

    TEST(SceneTemporalResolve, UsesRelativeLinearDepthForDisocclusion) {
        auto depth = sample();
        depth.history_linear_depth = 10.09f;
        EXPECT_TRUE(resolveSceneTemporalSample(depth).usedHistory());
        depth.history_linear_depth = 10.11f;
        EXPECT_EQ(resolveSceneTemporalSample(depth).rejection,
                  SceneHistoryRejection::Disocclusion);
        depth.history_linear_depth = 0.0f;
        EXPECT_EQ(resolveSceneTemporalSample(depth).rejection,
                  SceneHistoryRejection::Disocclusion);
    }

    TEST(SceneTemporalResolve, MotionReducesWeightAndExcessMotionIsRejected) {
        auto moving = sample();
        moving.current_to_previous_pixels = {64.0f, 0.0f};
        const auto result = resolveSceneTemporalSample(moving);
        EXPECT_TRUE(result.usedHistory());
        EXPECT_FLOAT_EQ(result.effective_history_weight, 0.45f);
        moving.current_to_previous_pixels = {129.0f, 0.0f};
        EXPECT_EQ(resolveSceneTemporalSample(moving).rejection,
                  SceneHistoryRejection::InvalidMotion);
    }

    TEST(SceneTemporalResolve, RenderPixelMotionIsNormalizedBeforeOutputHistoryLookup) {
        auto scaled = sample();
        scaled.motion_extent = {640, 360};
        scaled.output_extent = {1280, 720};
        scaled.current_pixel_center = {639.5f, 359.5f};
        scaled.current_to_previous_pixels = {32.0f, 18.0f};
        const auto result = resolveSceneTemporalSample(scaled);
        ASSERT_TRUE(result.usedHistory());
        EXPECT_NEAR(result.previous_uv.x,
                    639.5f / 1280.0f + 32.0f / 640.0f,
                    1e-6f);
        EXPECT_NEAR(result.previous_uv.y,
                    359.5f / 720.0f + 18.0f / 360.0f,
                    1e-6f);
    }

    TEST(SceneTemporalResolve, SanitizesWeightsAndRejectsNonFiniteData) {
        auto stable = sample();
        SceneTemporalResolveSettings settings;
        settings.history_weight = 4.0f;
        EXPECT_FLOAT_EQ(resolveSceneTemporalSample(stable, settings).effective_history_weight,
                        1.0f);
        stable.current.x = std::numeric_limits<float>::quiet_NaN();
        const auto invalid_current = resolveSceneTemporalSample(stable);
        EXPECT_EQ(invalid_current.rejection, SceneHistoryRejection::InvalidCurrent);
        EXPECT_EQ(invalid_current.color, glm::vec4(0.0f));
        stable = sample();
        stable.history.x = std::numeric_limits<float>::infinity();
        EXPECT_EQ(resolveSceneTemporalSample(stable).rejection,
                  SceneHistoryRejection::InvalidHistory);
    }

    TEST(VulkanSceneTemporalResolveContract, PingPongSelectionIsDeterministic) {
        EXPECT_TRUE(validTemporalViewId(TemporalViewId::Main));
        EXPECT_TRUE(validTemporalViewId(TemporalViewId::SplitRight));
        EXPECT_FALSE(validTemporalViewId(TemporalViewId::Count));
        EXPECT_EQ(nextTemporalHistoryWriteIndex(false, 0), 0u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 0), 1u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 1), 0u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 99), 0u);
    }

    TEST(VulkanSceneTemporalResolveContract, PaddedCurrentUvNeverSamplesOutsideValidRegion) {
        const auto transform = temporalCurrentUvTransform({1100, 738}, {1152, 768});
        EXPECT_NEAR(transform.x, 1100.0f / 1152.0f, 1e-6f);
        EXPECT_NEAR(transform.y, 738.0f / 768.0f, 1e-6f);
        EXPECT_LT(transform.z, transform.x);
        EXPECT_LT(transform.w, transform.y);
        EXPECT_EQ(temporalCurrentUvTransform({1200, 738}, {1152, 768}), glm::vec4(0.0f));
    }

    TEST(VulkanSceneTemporalResolveContract, DepthRejectionRequiresOwnedCurrentDepth) {
        VulkanSceneTemporalResolveParams params;
        params.render_extent = {1280, 720};
        EXPECT_TRUE(validTemporalDepthInputs(params));
        params.current_depth.enabled = true;
        EXPECT_FALSE(validTemporalDepthInputs(params));
        params.current_depth.current_depth_view = reinterpret_cast<VkImageView>(1);
        params.current_depth.depth = makeSceneDepthContract(true,
                                                            SceneDepthStorage::VulkanImage,
                                                            SceneDepthEncoding::LinearView,
                                                            params.render_extent,
                                                            0.1f,
                                                            1000.0f,
                                                            false,
                                                            false);
        EXPECT_TRUE(validTemporalDepthInputs(params));
    }

    TEST(VulkanSceneTemporalResolveContract, DepthRejectionRejectsUndeclaredLayouts) {
        VulkanSceneTemporalResolveParams params;
        params.render_extent = {1280, 720};
        params.current_depth.enabled = true;
        params.current_depth.current_depth_view = reinterpret_cast<VkImageView>(1);
        params.current_depth.depth = makeSceneDepthContract(true,
                                                            SceneDepthStorage::VulkanImage,
                                                            SceneDepthEncoding::LinearView,
                                                            params.render_extent,
                                                            0.1f,
                                                            1000.0f,
                                                            false,
                                                            false);
        params.current_depth.current_depth_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        EXPECT_FALSE(validTemporalDepthInputs(params));
        params.current_depth.current_depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        EXPECT_TRUE(validTemporalDepthInputs(params));
        params.view = TemporalViewId::SplitLeft;
        EXPECT_FALSE(validTemporalDepthInputs(params));
        params.current_depth.view = TemporalViewId::SplitLeft;
        EXPECT_TRUE(validTemporalDepthInputs(params));
    }

    TEST(VulkanSceneTemporalResolveContract, DepthPingPongSlotsArePerViewAndBounded) {
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::Main, 0), 0u);
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::Main, 1), 1u);
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::SplitLeft, 0), 2u);
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::SplitRight, 1), 5u);
        EXPECT_FALSE(temporalDepthHistoryResourceSlot(TemporalViewId::Main, 2));
        EXPECT_FALSE(temporalDepthHistoryResourceSlot(TemporalViewId::Count, 0));
    }

    TEST(VulkanSceneTemporalPipelineContract, RequiresCoherentViewsExtentsAndInputs) {
        auto request = pipelineRequest();
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
        request.resolve.output_extent.x += 1;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.resolve.view = TemporalViewId::SplitLeft;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.resolve.motion_view = reinterpret_cast<VkImageView>(3);
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.resolve.current_depth.current_depth_view = reinterpret_cast<VkImageView>(4);
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.temporal.frame.view.size.x -= 1;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
    }

    TEST(VulkanSceneTemporalPipelineContract, DepthHistoryRequirementMatchesOwnership) {
        auto request = pipelineRequest();
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
        request.resolve.current_depth.enabled = false;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request.temporal.requirements.history_depth = false;
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.motion.includes_jitter = false;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
    }

    TEST(VulkanSceneTemporalPipelineContract, InactiveRequestRemainsZeroCost) {
        VulkanSceneTemporalPipelineRequest request;
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
        request.temporal.render_extent = {640, 360};
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
    }

} // namespace lfs::vis
