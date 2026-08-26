/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/temporal_frame_tracker.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {
    namespace {
        TemporalFrameInput frameInput() {
            TemporalFrameInput input;
            input.view.size = {1280, 720};
            input.output_extent = {1920, 1080};
            input.scene_generation = 4;
            input.backend_key = 2;
            return input;
        }
    } // namespace

    TEST(TemporalFrameTracker, FirstFrameHasNoHistoryAndCommitEnablesNextFrame) {
        TemporalFrameTracker tracker;
        const auto input = frameInput();
        EXPECT_TRUE(hasTemporalResetReason(tracker.prepare(TemporalViewId::Main, input).reset_reasons,
                                           TemporalResetReason::FirstFrame));
        tracker.commit(TemporalViewId::Main, input);
        const auto next = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_TRUE(next.history_valid);
        EXPECT_EQ(next.sequence, 1u);
    }

    TEST(TemporalFrameTracker, CarriesCurrentAndPreviousJitterWithoutApplyingIt) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        input.jitter = {0.25f, -0.125f};
        tracker.commit(TemporalViewId::Main, input);

        input.jitter = {-0.25f, 0.125f};
        const auto state = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_TRUE(state.history_valid);
        EXPECT_EQ(state.current_jitter, input.jitter);
        EXPECT_EQ(state.previous_jitter, glm::vec2(0.25f, -0.125f));
    }

    TEST(TemporalFrameTracker, HaltonJitterIsBoundedAndResolutionAware) {
        const auto first = temporalJitterPixels(0);
        EXPECT_FLOAT_EQ(first.x, 0.0f);
        EXPECT_NEAR(first.y, -1.0f / 6.0f, 1e-6f);
        for (std::uint64_t sequence = 0; sequence < 64; ++sequence) {
            const auto jitter = temporalJitterPixels(sequence);
            EXPECT_GE(jitter.x, -0.5f);
            EXPECT_LT(jitter.x, 0.5f);
            EXPECT_GE(jitter.y, -0.5f);
            EXPECT_LT(jitter.y, 0.5f);
        }

        const auto ndc = temporalJitterNdc(0, {100, 50});
        EXPECT_FLOAT_EQ(ndc.x, 0.0f);
        EXPECT_NEAR(ndc.y, 1.0f / 150.0f, 1e-6f);
        EXPECT_EQ(temporalJitterNdc(3, {0, 50}), glm::vec2(0.0f));
        EXPECT_EQ(temporalJitterNdc(glm::vec2(std::numeric_limits<float>::infinity(), 0.0f),
                                    {100, 50}),
                  glm::vec2(0.0f));
    }

    TEST(TemporalFrameTracker, PixelJitterMatchesTopLeftProjectionCoordinates) {
        constexpr glm::ivec2 extent{1280, 720};
        constexpr glm::vec2 jitter_pixels{0.25f, -0.375f};
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            extent,
            lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
            false,
            lfs::rendering::DEFAULT_ORTHO_SCALE);
        const glm::vec4 point(0.2f, -0.1f, -4.0f, 1.0f);
        const glm::vec4 base_clip = projection * point;
        const glm::vec4 jittered_clip =
            applySceneProjectionJitter(projection, temporalJitterNdc(jitter_pixels, extent)) *
            point;
        const auto to_top_left_pixel = [extent](const glm::vec4& clip) {
            const glm::vec2 ndc = glm::vec2(clip) / clip.w;
            return glm::vec2{
                (ndc.x * 0.5f + 0.5f) * static_cast<float>(extent.x),
                (0.5f - ndc.y * 0.5f) * static_cast<float>(extent.y)};
        };

        const glm::vec2 observed_jitter =
            to_top_left_pixel(jittered_clip) - to_top_left_pixel(base_clip);
        EXPECT_NEAR(observed_jitter.x, jitter_pixels.x, 1e-4f);
        EXPECT_NEAR(observed_jitter.y, jitter_pixels.y, 1e-4f);
    }

    TEST(TemporalFrameTracker, ConvergenceIsFiniteAndOnlyRestartsForNewSourceWork) {
        TemporalConvergenceController convergence;
        convergence.prepare(true, true);
        EXPECT_TRUE(convergence.enabled());
        EXPECT_EQ(convergence.remaining(), TemporalConvergenceController::SAMPLE_COUNT);

        std::uint32_t follow_ups = 0;
        for (std::uint32_t sample = 0;
             sample < TemporalConvergenceController::SAMPLE_COUNT;
             ++sample) {
            EXPECT_EQ(convergence.jitter(), temporalJitterPixels(sample));
            if (convergence.completeSuccessfulFrame())
                ++follow_ups;
            convergence.prepare(true, false);
        }
        EXPECT_EQ(follow_ups, TemporalConvergenceController::SAMPLE_COUNT - 1);
        EXPECT_EQ(convergence.remaining(), 0u);
        EXPECT_EQ(convergence.jitter(), temporalJitterPixels(
                                            TemporalConvergenceController::SAMPLE_COUNT));
        EXPECT_FALSE(convergence.completeSuccessfulFrame());

        const auto steady_sequence = convergence.sequence();
        convergence.prepare(true, true);
        EXPECT_EQ(convergence.sequence(), steady_sequence);
        EXPECT_EQ(convergence.remaining(), TemporalConvergenceController::SAMPLE_COUNT);
        EXPECT_EQ(convergence.jitter(), temporalJitterPixels(steady_sequence));

        convergence.prepare(false, true);
        EXPECT_FALSE(convergence.enabled());
        EXPECT_EQ(convergence.sequence(), 0u);
    }

    TEST(TemporalFrameTracker, VolatileRefreshAdvancesJitterWithoutSchedulingSettleFrames) {
        TemporalConvergenceController convergence;
        convergence.prepare(true, true, false);

        EXPECT_TRUE(convergence.enabled());
        EXPECT_EQ(convergence.remaining(), 0u);
        EXPECT_EQ(convergence.jitter(), temporalJitterPixels(0));
        EXPECT_FALSE(convergence.completeSuccessfulFrame());
        EXPECT_EQ(convergence.sequence(), 1u);
        EXPECT_EQ(convergence.jitter(), temporalJitterPixels(1));

        convergence.prepare(true, false, true);
        EXPECT_EQ(convergence.remaining(), 0u);
        convergence.prepare(true, true, true);
        EXPECT_EQ(convergence.remaining(), TemporalConvergenceController::SAMPLE_COUNT);
        EXPECT_EQ(convergence.sequence(), 1u);
    }

    TEST(TemporalFrameTracker, CancelSettleClearsRemainingWithoutResettingSequence) {
        TemporalConvergenceController convergence;
        convergence.prepare(true, true);
        EXPECT_EQ(convergence.remaining(), TemporalConvergenceController::SAMPLE_COUNT);
        const auto sequence = convergence.sequence();
        EXPECT_EQ(convergence.jitter(), temporalJitterPixels(sequence));

        convergence.cancelSettle();
        EXPECT_EQ(convergence.remaining(), 0u);
        EXPECT_TRUE(convergence.enabled());
        EXPECT_EQ(convergence.sequence(), sequence);
        EXPECT_EQ(convergence.jitter(), temporalJitterPixels(sequence));
    }

    TEST(TemporalFrameTracker, JitterOffsetsOnlyTheSuppliedSceneProjection) {
        const glm::mat4 projection(1.0f);
        const glm::vec4 point(0.2f, -0.1f, 0.5f, 1.0f);
        const glm::vec2 jitter(0.25f, -0.5f);
        const glm::vec4 shifted = applySceneProjectionJitter(projection, jitter) * point;
        EXPECT_NEAR(shifted.x / shifted.w, point.x / point.w + jitter.x, 1e-6f);
        EXPECT_NEAR(shifted.y / shifted.w, point.y / point.w + jitter.y, 1e-6f);
        EXPECT_EQ(applySceneProjectionJitter(projection, glm::vec2(0.0f)), projection);
        EXPECT_EQ(applySceneProjectionJitter(
                      projection,
                      {std::numeric_limits<float>::quiet_NaN(), 0.0f}),
                  projection);
    }

    TEST(TemporalFrameTracker, ProjectionPairUsesCurrentAndPreviousJitter) {
        TemporalFrameState state;
        state.current_jitter = {0.25f, 0.0f};
        state.previous_jitter = {-0.25f, 0.0f};
        const auto pair = makeTemporalProjectionPair(
            state, glm::mat4(1.0f), glm::mat4(1.0f));
        const glm::vec4 point(0.0f, 0.0f, 0.0f, 1.0f);
        EXPECT_NEAR((pair.current * point).x, 0.25f, 1e-6f);
        EXPECT_NEAR((pair.previous * point).x, -0.25f, 1e-6f);
    }

    TEST(TemporalFrameTracker, ViewProjectionPairOwnsCameraMotionAndJitter) {
        auto input = frameInput();
        TemporalFrameTracker tracker;
        tracker.commit(TemporalViewId::Main, input);

        input.view.translation.x += 1.0f;
        input.jitter = {0.125f, -0.25f};
        const auto state = tracker.prepare(TemporalViewId::Main, input);
        const auto pair = makeTemporalViewProjectionPair(state);

        ASSERT_TRUE(pair.has_value());
        const auto current_projection = lfs::rendering::createProjectionMatrixFromFocal(
            state.current.size,
            state.current.focal_length_mm,
            state.current.orthographic,
            state.current.ortho_scale,
            state.current.near_plane,
            state.current.far_plane);
        const auto expected = applySceneProjectionJitter(current_projection, state.current_jitter) *
                              state.current.getViewMatrix();
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                EXPECT_FLOAT_EQ((*pair).current[column][row], expected[column][row]);
            }
        }
        EXPECT_NE((*pair).current[3][0], (*pair).previous[3][0]);
    }

    TEST(TemporalFrameTracker, ViewProjectionPairRejectsUnsupportedIntrinsics) {
        auto input = frameInput();
        input.view.intrinsics_override = lfs::rendering::CameraIntrinsics{};
        TemporalFrameTracker tracker;
        const auto state = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_FALSE(makeTemporalViewProjectionPair(state).has_value());
    }

    TEST(TemporalFrameTracker, SceneViewJitterPreservesExplicitFocalLength) {
        auto input = frameInput();
        input.view.intrinsics_override = lfs::rendering::CameraIntrinsics{
            .focal_x = 900.0f,
            .focal_y = 880.0f,
            .center_x = 640.0f,
            .center_y = 360.0f,
        };
        const auto jittered = applySceneViewJitter(input.view, {0.25f, -0.125f});
        ASSERT_TRUE(jittered.intrinsics_override);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->focal_x, 900.0f);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->focal_y, 880.0f);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->center_x, 640.25f);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->center_y, 359.875f);
        EXPECT_FLOAT_EQ(input.view.intrinsics_override->center_x, 640.0f);
    }

    TEST(TemporalFrameTracker, SceneViewJitterSynthesizesTileStableIntrinsics) {
        auto input = frameInput();
        input.view.size = {320, 180};
        input.view.subregion_origin = {320, 180};
        input.view.subregion_full_size = {1280, 720};
        const auto jittered = applySceneViewJitter(input.view, {-0.25f, 0.25f});
        ASSERT_TRUE(jittered.intrinsics_override);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->center_x, 639.75f);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->center_y, 360.25f);
        const auto focal = lfs::rendering::computePixelFocalLengths(
            glm::ivec2(1280, 720), input.view.focal_length_mm);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->focal_x, focal.first);
        EXPECT_FLOAT_EQ(jittered.intrinsics_override->focal_y, focal.second);
    }

    TEST(TemporalFrameTracker, SceneViewJitterIsNoOpWhenUnsupported) {
        const auto input = frameInput();
        EXPECT_FALSE(applySceneViewJitter(input.view, {}).intrinsics_override);
        EXPECT_FALSE(applySceneViewJitter(
                         input.view,
                         {std::numeric_limits<float>::quiet_NaN(), 0.0f})
                         .intrinsics_override);
        auto orthographic = input.view;
        orthographic.orthographic = true;
        EXPECT_FALSE(applySceneViewJitter(orthographic, {0.25f, 0.25f}).intrinsics_override);
    }

    TEST(TemporalFrameTracker, CameraMotionPreservesHistoryButExplicitCutResetsIt) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        input.view.translation.x = 1.0f;
        EXPECT_TRUE(tracker.prepare(TemporalViewId::Main, input).history_valid);
        input.camera_cut = true;
        const auto cut = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_FALSE(cut.history_valid);
        EXPECT_TRUE(hasTemporalResetReason(cut.reset_reasons, TemporalResetReason::CameraCut));
    }

    TEST(TemporalFrameTracker, RenderAndProjectionChangesResetHistory) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        input.view.size = {960, 540};
        input.output_extent = {1440, 810};
        input.render_scale = 0.75f;
        input.view.focal_length_mm = 50.0f;
        const auto changed = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::RenderSize));
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::RenderScale));
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::Projection));
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::OutputExtent));
    }

    TEST(TemporalFrameTracker, SceneAndBackendChangesResetHistory) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        ++input.scene_generation;
        ++input.backend_key;
        const auto changed = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::Scene));
        EXPECT_TRUE(hasTemporalResetReason(changed.reset_reasons, TemporalResetReason::Backend));
        EXPECT_EQ(changed.sequence, 0u);
    }

    TEST(TemporalFrameTracker, PrepareDoesNotAdvanceAndViewsRemainIndependent) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        EXPECT_EQ(tracker.prepare(TemporalViewId::Main, input).sequence, 1u);
        EXPECT_EQ(tracker.prepare(TemporalViewId::Main, input).sequence, 1u);
        EXPECT_TRUE(hasTemporalResetReason(
            tracker.prepare(TemporalViewId::SplitRight, input).reset_reasons,
            TemporalResetReason::FirstFrame));
    }

    TEST(TemporalFrameTracker, ExplicitResetRestartsAccumulationSequence) {
        TemporalFrameTracker tracker;
        const auto input = frameInput();
        tracker.commit(TemporalViewId::Main, input);
        tracker.commit(TemporalViewId::Main, input);
        ASSERT_EQ(tracker.prepare(TemporalViewId::Main, input).sequence, 2u);
        tracker.reset(TemporalViewId::Main, TemporalResetReason::Scene);
        const auto reset = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_EQ(reset.sequence, 0u);
        EXPECT_FALSE(reset.history_valid);
        EXPECT_TRUE(hasTemporalResetReason(reset.reset_reasons, TemporalResetReason::Scene));
    }

    TEST(TemporalFrameTracker, InvalidInputCannotBecomeHistory) {
        TemporalFrameTracker tracker;
        auto input = frameInput();
        input.view.translation.x = std::numeric_limits<float>::quiet_NaN();
        tracker.commit(TemporalViewId::Main, input);
        const auto state = tracker.prepare(TemporalViewId::Main, input);
        EXPECT_FALSE(state.history_valid);
        EXPECT_TRUE(hasTemporalResetReason(state.reset_reasons, TemporalResetReason::InvalidInput));

        input = frameInput();
        input.jitter.y = std::numeric_limits<float>::infinity();
        tracker.commit(TemporalViewId::Main, input);
        EXPECT_TRUE(hasTemporalResetReason(
            tracker.prepare(TemporalViewId::Main, input).reset_reasons,
            TemporalResetReason::InvalidInput));
    }
} // namespace lfs::vis
