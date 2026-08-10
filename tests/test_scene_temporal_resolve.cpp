/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/passes/vulkan_scene_temporal_resolve_pass.hpp"
#include "visualizer/rendering/scene_temporal_resolve.hpp"
#include "visualizer/rendering/scene_upscaler_inputs.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace lfs::vis {
    namespace {
        SceneUpscalerRequirements temporalRequirements() {
            return {
                .depth = true,
                .motion_vectors = true,
                .jitter = true,
                .history = true,
                .reactive_mask = true,
                .exposure = true,
            };
        }

        SceneTemporalResolveSample validSample() {
            return {
                .current = {0.2f, 0.4f, 0.6f, 1.0f},
                .history = {0.4f, 0.6f, 0.8f, 1.0f},
                .neighborhood_min = {0.1f, 0.2f, 0.3f, 1.0f},
                .neighborhood_max = {0.7f, 0.8f, 0.9f, 1.0f},
                .motion_pixels = {0.0f, 0.0f},
                .current_pixel = {639.0f, 359.0f},
                .output_extent = {1280, 720},
                .current_depth = 0.5f,
                .history_depth = 0.5f,
                .history_valid = true,
                .depth_available = true,
            };
        }

        SceneUpscalerInputs validInputs() {
            return {
                .depth = makeSceneDepthContract(true,
                                                SceneDepthStorage::VulkanImage,
                                                true,
                                                1280,
                                                720,
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
                .history = makeSceneHistoryContract(true,
                                                    SceneHistoryStorage::VulkanImage,
                                                    SceneHistoryStorage::VulkanImage,
                                                    1920,
                                                    1080,
                                                    1),
                .jitter_applied = true,
                .reactive_mask_available = true,
                .exposure_available = true,
            };
        }
    } // namespace

    TEST(SceneUpscalerInputs, NativeAndSpatialAcceptTheZeroCostContract) {
        EXPECT_TRUE(validateSceneUpscalerInputs(
                        makeSceneTemporalPlan(nativeSceneUpscalerDescriptor().requirements, {}, {}),
                        {})
                        .valid());
        EXPECT_TRUE(validateSceneUpscalerInputs(
                        makeSceneTemporalPlan(spatialSceneUpscalerDescriptor().requirements, {}, {}),
                        {})
                        .valid());
    }

    TEST(SceneUpscalerInputs, ReportsEveryMissingTemporalInput) {
        const auto plan = makeSceneTemporalPlan(
            temporalRequirements(), {1280, 720}, {1920, 1080});
        const auto validation = validateSceneUpscalerInputs(plan, {});

        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::MissingDepth));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::MissingMotion));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::MissingJitter));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::MissingHistory));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::MissingReactiveMask));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::MissingExposure));
    }

    TEST(SceneUpscalerInputs, AcceptsACompleteTemporalContract) {
        const auto plan = makeSceneTemporalPlan(
            temporalRequirements(), {1280, 720}, {1920, 1080});
        EXPECT_TRUE(validateSceneUpscalerInputs(plan, validInputs()).valid());
    }

    TEST(SceneUpscalerInputs, RejectsExtentAndDirectionMismatches) {
        const auto plan = makeSceneTemporalPlan(
            temporalRequirements(), {1280, 720}, {1920, 1080});
        auto inputs = validInputs();
        inputs.depth.width = 1279;
        inputs.motion.direction = SceneMotionDirection::PreviousToCurrent;
        inputs.motion.includes_jitter = false;
        inputs.history.height = 1079;
        const auto validation = validateSceneUpscalerInputs(plan, inputs);

        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::DepthExtent));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::MotionDirection));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(
            validation.issues, SceneUpscalerInputIssue::MotionJitterMismatch));
        EXPECT_TRUE(hasSceneUpscalerInputIssue(validation.issues,
                                               SceneUpscalerInputIssue::HistoryExtent));
    }

    TEST(SceneTemporalResolve, UsesClampedHistoryForAStableSample) {
        auto sample = validSample();
        sample.history = {2.0f, 0.6f, -1.0f, 1.0f};
        const auto result = resolveSceneTemporalSample(sample);

        EXPECT_TRUE(result.usedHistory());
        EXPECT_FLOAT_EQ(result.effective_history_weight, 0.9f);
        EXPECT_NEAR(result.color.x, 0.65f, 1e-6f);
        EXPECT_NEAR(result.color.y, 0.58f, 1e-6f);
        EXPECT_NEAR(result.color.z, 0.33f, 1e-6f);
    }

    TEST(SceneTemporalResolve, RejectsFirstFrameAndDisocclusion) {
        auto sample = validSample();
        sample.history_valid = false;
        EXPECT_EQ(resolveSceneTemporalSample(sample).rejection,
                  SceneHistoryRejection::NoHistory);

        sample.history_valid = true;
        sample.history_depth = 0.8f;
        EXPECT_EQ(resolveSceneTemporalSample(sample).rejection,
                  SceneHistoryRejection::Disocclusion);
    }

    TEST(SceneTemporalResolve, RejectsInvalidOrOutOfBoundsMotion) {
        auto sample = validSample();
        sample.motion_pixels.x = std::numeric_limits<float>::infinity();
        EXPECT_EQ(resolveSceneTemporalSample(sample).rejection,
                  SceneHistoryRejection::InvalidMotion);

        sample = validSample();
        sample.current_pixel = {0.0f, 0.0f};
        sample.motion_pixels = {-2.0f, 0.0f};
        EXPECT_EQ(resolveSceneTemporalSample(sample).rejection,
                  SceneHistoryRejection::OutsideHistory);
    }

    TEST(SceneTemporalResolve, ReducesHistoryWeightAsMotionIncreases) {
        auto sample = validSample();
        const auto stationary = resolveSceneTemporalSample(sample);
        sample.motion_pixels = {64.0f, 0.0f};
        const auto moving = resolveSceneTemporalSample(sample);

        EXPECT_TRUE(moving.usedHistory());
        EXPECT_LT(moving.effective_history_weight, stationary.effective_history_weight);
        EXPECT_FLOAT_EQ(moving.effective_history_weight, 0.45f);
    }

    TEST(SceneTemporalResolve, SanitizesSettingsAndInvalidColors) {
        auto sample = validSample();
        SceneTemporalResolveSettings settings;
        settings.history_weight = 4.0f;
        EXPECT_FLOAT_EQ(resolveSceneTemporalSample(sample, settings).effective_history_weight,
                        1.0f);

        sample.current.x = std::numeric_limits<float>::quiet_NaN();
        const auto invalid = resolveSceneTemporalSample(sample, settings);
        EXPECT_EQ(invalid.rejection, SceneHistoryRejection::InvalidColor);
        EXPECT_EQ(invalid.color, glm::vec4(0.0f));
    }

    TEST(SceneTemporalResolve, PingPongAndPaddedUvContractsAreDeterministic) {
        EXPECT_EQ(nextTemporalHistoryWriteIndex(false, 0), 0u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 0), 1u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 1), 0u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 99), 0u);

        const auto uv = temporalCurrentUvTransform({1100, 738}, {1152, 768});
        EXPECT_NEAR(uv.x, 1100.0f / 1152.0f, 1e-6f);
        EXPECT_NEAR(uv.y, 738.0f / 768.0f, 1e-6f);
        EXPECT_LT(uv.z, uv.x);
        EXPECT_LT(uv.w, uv.y);
        EXPECT_EQ(temporalCurrentUvTransform({1200, 738}, {1152, 768}), glm::vec4(0.0f));
    }

} // namespace lfs::vis
