/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_motion_contract.hpp"

#include <gtest/gtest.h>

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

} // namespace lfs::vis
