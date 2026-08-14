/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_motion_contract.hpp"
#include "visualizer/rendering/scene_motion_reprojection.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {

    TEST(SceneMotionContract, UnavailableMotionIsAValidZeroCostState) {
        const SceneMotionContract motion{};
        EXPECT_FALSE(motion.available());
        EXPECT_TRUE(motion.valid());
        EXPECT_FALSE(motion.matchesRenderExtent(1920, 1080));
        EXPECT_FALSE(motion.matchesRenderExtent(0, 0));
        EXPECT_FALSE(motion.requiresPixelConversion());
    }

    TEST(SceneMotionContract, DescribesCurrentToPreviousPixelMotion) {
        const auto motion = makeSceneMotionContract(
            true,
            SceneMotionStorage::VulkanImage,
            SceneMotionEncoding::PixelDisplacement,
            SceneMotionDirection::CurrentToPrevious,
            1280,
            720,
            true,
            false);
        EXPECT_TRUE(motion.available());
        EXPECT_TRUE(motion.valid());
        EXPECT_TRUE(motion.matchesRenderExtent(1280, 720));
        EXPECT_FALSE(motion.matchesRenderExtent(1920, 1080));
        EXPECT_FALSE(motion.requiresPixelConversion());
        EXPECT_TRUE(motion.includes_jitter);
    }

    TEST(SceneMotionContract, IdentifiesNormalizedMotionThatNeedsConversion) {
        const auto uv_motion = makeSceneMotionContract(
            true,
            SceneMotionStorage::Tensor,
            SceneMotionEncoding::NormalizedUvDisplacement,
            SceneMotionDirection::PreviousToCurrent,
            1920,
            1080,
            false,
            true);
        EXPECT_TRUE(uv_motion.valid());
        EXPECT_TRUE(uv_motion.requiresPixelConversion());
        EXPECT_TRUE(uv_motion.flip_y);

        const auto ndc_motion = makeSceneMotionContract(
            true,
            SceneMotionStorage::VulkanImage,
            SceneMotionEncoding::NdcDisplacement,
            SceneMotionDirection::CurrentToPrevious,
            1920,
            1080,
            false,
            false);
        EXPECT_TRUE(ndc_motion.valid());
        EXPECT_TRUE(ndc_motion.requiresPixelConversion());
    }

    TEST(SceneMotionContract, RejectsMalformedAvailableMotion) {
        EXPECT_FALSE(makeSceneMotionContract(
                         true,
                         SceneMotionStorage::VulkanImage,
                         SceneMotionEncoding::PixelDisplacement,
                         SceneMotionDirection::CurrentToPrevious,
                         0,
                         720,
                         false,
                         false)
                         .valid());
        EXPECT_FALSE(makeSceneMotionContract(
                         true,
                         SceneMotionStorage::None,
                         SceneMotionEncoding::PixelDisplacement,
                         SceneMotionDirection::CurrentToPrevious,
                         1280,
                         720,
                         false,
                         false)
                         .valid());
        EXPECT_FALSE(makeSceneMotionContract(
                         true,
                         SceneMotionStorage::Tensor,
                         SceneMotionEncoding::Unavailable,
                         SceneMotionDirection::CurrentToPrevious,
                         1280,
                         720,
                         false,
                         false)
                         .valid());
    }

    TEST(SceneMotionContract, CanonicalizesEncodingDirectionAndVerticalOrigin) {
        const auto pixels = makeSceneMotionContract(
            true,
            SceneMotionStorage::VulkanImage,
            SceneMotionEncoding::PixelDisplacement,
            SceneMotionDirection::CurrentToPrevious,
            200,
            100,
            false,
            false);
        const auto canonical_pixels = canonicalSceneMotionPixels({20.0f, -10.0f}, pixels);
        ASSERT_TRUE(canonical_pixels.has_value());
        EXPECT_EQ(*canonical_pixels, glm::vec2(20.0f, -10.0f));

        auto uv = pixels;
        uv.encoding = SceneMotionEncoding::NormalizedUvDisplacement;
        uv.direction = SceneMotionDirection::PreviousToCurrent;
        uv.flip_y = true;
        const auto canonical_uv = canonicalSceneMotionPixels({0.1f, 0.1f}, uv);
        ASSERT_TRUE(canonical_uv.has_value());
        EXPECT_EQ(*canonical_uv, glm::vec2(-20.0f, 10.0f));

        auto ndc = pixels;
        ndc.encoding = SceneMotionEncoding::NdcDisplacement;
        const auto canonical_ndc = canonicalSceneMotionPixels({0.2f, -0.2f}, ndc);
        ASSERT_TRUE(canonical_ndc.has_value());
        EXPECT_EQ(*canonical_ndc, glm::vec2(20.0f, -10.0f));
    }

    TEST(SceneMotionContract, RejectsUnavailableOrNonFiniteMotionSamples) {
        EXPECT_FALSE(canonicalSceneMotionPixels({1.0f, 1.0f}, {}).has_value());
        const auto contract = makeSceneMotionContract(
            true,
            SceneMotionStorage::Tensor,
            SceneMotionEncoding::PixelDisplacement,
            SceneMotionDirection::CurrentToPrevious,
            1280,
            720,
            false,
            false);
        EXPECT_FALSE(canonicalSceneMotionPixels(
                         {std::numeric_limits<float>::quiet_NaN(), 0.0f}, contract)
                         .has_value());
    }

    TEST(SceneMotionContract, DerivesCurrentToPreviousDisplacementFromClipPositions) {
        const auto motion = currentToPreviousSceneMotionNdc(
            glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
            glm::vec4(0.0f, 0.5f, 0.0f, 2.0f));
        ASSERT_TRUE(motion.has_value());
        EXPECT_EQ(*motion, glm::vec2(-0.5f, 0.25f));

        EXPECT_FALSE(currentToPreviousSceneMotionNdc(
                         glm::vec4(0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f))
                         .has_value());
    }

    TEST(SceneMotionReprojection, IdentityCameraProducesZeroMotion) {
        SceneMotionReprojectionParams params;
        params.render_extent = {200, 100};
        const auto motion = reprojectSceneMotionPixels(params, {100.0f, 50.0f}, 0.5f);
        ASSERT_TRUE(motion.has_value());
        EXPECT_NEAR(motion->x, 0.0f, 1e-6f);
        EXPECT_NEAR(motion->y, 0.0f, 1e-6f);
    }

    TEST(SceneMotionReprojection, PreviousProjectionProducesPixelDisplacement) {
        SceneMotionReprojectionParams params;
        params.render_extent = {200, 100};
        params.previous_view_projection[3][0] = 0.2f;
        const auto motion = reprojectSceneMotionPixels(params, {100.0f, 50.0f}, 0.5f);
        ASSERT_TRUE(motion.has_value());
        EXPECT_NEAR(motion->x, 20.0f, 1e-5f);
        EXPECT_NEAR(motion->y, 0.0f, 1e-5f);

        params.flip_y = true;
        params.previous_view_projection[3][1] = 0.2f;
        const auto flipped = reprojectSceneMotionPixels(params, {100.0f, 50.0f}, 0.5f);
        ASSERT_TRUE(flipped.has_value());
        EXPECT_NEAR(flipped->y, -10.0f, 1e-5f);
    }

    TEST(SceneMotionReprojection, RejectsMalformedInputsAndHomogeneousCoordinates) {
        SceneMotionReprojectionParams params;
        EXPECT_FALSE(reprojectSceneMotionPixels(params, {0.5f, 0.5f}, 0.5f).has_value());
        params.render_extent = {1280, 720};
        EXPECT_FALSE(reprojectSceneMotionPixels(params, {0.5f, 0.5f}, -0.1f).has_value());
        EXPECT_FALSE(reprojectSceneMotionPixels(params, {1280.0f, 0.5f}, 0.5f).has_value());
        EXPECT_FALSE(reprojectSceneMotionPixels(params, {0.5f, 0.5f}, 1.0f).has_value());
        params.inverse_current_view_projection = glm::mat4(0.0f);
        EXPECT_FALSE(reprojectSceneMotionPixels(params, {0.5f, 0.5f}, 0.5f).has_value());
    }

} // namespace lfs::vis
