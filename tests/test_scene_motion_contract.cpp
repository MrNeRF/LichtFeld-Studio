/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/passes/vulkan_scene_motion_pass.hpp"
#include "visualizer/rendering/scene_motion_reprojection.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <rendering/coordinate_conventions.hpp>

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
        EXPECT_NEAR(flipped->y, 10.0f, 1e-5f);
    }

    TEST(SceneMotionReprojection, PerspectiveCameraMotionUsesTopLeftImageCoordinates) {
        constexpr glm::ivec2 extent{640, 360};
        constexpr float focal_length_mm = 35.0f;
        constexpr float near_plane = 0.1f;
        constexpr float far_plane = 100.0f;
        const glm::mat3 rotation(1.0f);
        const glm::vec3 current_position(0.1f, -0.2f, 0.3f);
        const glm::vec3 previous_position(-0.2f, 0.4f, 0.1f);
        const glm::vec3 world_point(0.3f, 0.2f, -4.0f);

        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            extent, focal_length_mm, false, lfs::rendering::DEFAULT_ORTHO_SCALE,
            near_plane, far_plane);
        const glm::mat4 current_view = lfs::rendering::makeViewMatrix(
            rotation, current_position);
        const glm::mat4 previous_view = lfs::rendering::makeViewMatrix(
            rotation, previous_position);
        const glm::vec4 current_clip = projection * current_view * glm::vec4(world_point, 1.0f);
        ASSERT_GT(std::abs(current_clip.w), 1e-6f);
        const float ndc_depth = current_clip.z / current_clip.w;

        const auto current_pixel = lfs::rendering::projectWorldPoint(
            rotation, current_position, extent, world_point, focal_length_mm);
        const auto previous_pixel = lfs::rendering::projectWorldPoint(
            rotation, previous_position, extent, world_point, focal_length_mm);
        ASSERT_TRUE(current_pixel.has_value());
        ASSERT_TRUE(previous_pixel.has_value());

        const SceneMotionReprojectionParams params{
            .inverse_current_view_projection = glm::inverse(projection * current_view),
            .previous_view_projection = projection * previous_view,
            .render_extent = extent,
        };
        const auto motion = reprojectSceneMotionPixels(params, *current_pixel, ndc_depth);
        ASSERT_TRUE(motion.has_value());
        const glm::vec2 expected = *previous_pixel - *current_pixel;
        // The matrix unprojection and direct pinhole projection intentionally
        // use independent float paths. Keep their agreement well below one
        // thousandth of a pixel without requiring bit-identical evaluation.
        constexpr float PIXEL_TOLERANCE = 1e-3f;
        EXPECT_NEAR(motion->x, expected.x, PIXEL_TOLERANCE);
        EXPECT_NEAR(motion->y, expected.y, PIXEL_TOLERANCE);
        EXPECT_GT(motion->x, 0.0f);
        EXPECT_GT(motion->y, 0.0f);
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
