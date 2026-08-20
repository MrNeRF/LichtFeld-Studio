/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

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
                .output_extent = {1280, 720},
                .current_linear_depth = 10.0f,
                .history_linear_depth = 10.0f,
                .history_valid = true,
                .depth_available = true,
            };
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

} // namespace lfs::vis
