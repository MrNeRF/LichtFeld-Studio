/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/passes/vulkan_scene_motion_pass.hpp"
#include "visualizer/rendering/scene_motion_reprojection.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {

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

    TEST(VulkanSceneMotionContract, DisabledOrIncompleteRequestsRemainZeroCost) {
        VulkanSceneMotionParams params;
        params.render_extent = {1280, 720};
        EXPECT_FALSE(canRecordVulkanSceneMotion(params));
        params.enabled = true;
        EXPECT_FALSE(canRecordVulkanSceneMotion(params));
        params.depth_view = reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(1));
        params.depth = makeSceneDepthContract(true,
                                              SceneDepthStorage::VulkanImage,
                                              SceneDepthEncoding::VulkanNdc,
                                              params.render_extent,
                                              0.1f,
                                              1000.0f,
                                              false,
                                              false);
        EXPECT_TRUE(canRecordVulkanSceneMotion(params));
        params.depth.encoding = SceneDepthEncoding::LinearView;
        EXPECT_TRUE(canRecordVulkanSceneMotion(params));
        params.depth.encoding = SceneDepthEncoding::VulkanNdc;
        params.render_extent = {0, 720};
        EXPECT_FALSE(canRecordVulkanSceneMotion(params));
    }

    TEST(VulkanSceneMotionContract, ResourceSlotsAreUniquePerFrameAndView) {
        EXPECT_EQ(temporalMotionResourceSlot(0, TemporalViewId::Main), 0u);
        EXPECT_EQ(temporalMotionResourceSlot(0, TemporalViewId::SplitLeft), 1u);
        EXPECT_EQ(temporalMotionResourceSlot(0, TemporalViewId::SplitRight), 2u);
        EXPECT_EQ(temporalMotionResourceSlot(1, TemporalViewId::Main), 3u);
        EXPECT_EQ(temporalMotionResourceSlot(1, TemporalViewId::SplitLeft), 4u);
        EXPECT_EQ(temporalMotionResourceSlot(1, TemporalViewId::SplitRight), 5u);
        EXPECT_FALSE(temporalMotionResourceSlot(0, TemporalViewId::Count).has_value());
        EXPECT_FALSE(temporalMotionResourceSlot(std::numeric_limits<std::size_t>::max(),
                                                TemporalViewId::SplitRight)
                         .has_value());
    }

} // namespace lfs::vis
