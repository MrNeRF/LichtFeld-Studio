/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_depth_contract.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {

    TEST(SceneDepthContract, UnavailableDepthIsAValidZeroCostState) {
        const SceneDepthContract depth{};
        EXPECT_FALSE(depth.available());
        EXPECT_TRUE(depth.valid());
        EXPECT_FALSE(depth.matchesRenderExtent(1920, 1080));
        EXPECT_FALSE(depth.requiresLinearization());
    }

    TEST(SceneDepthContract, DescribesLinearTensorDepth) {
        const auto depth = makeSceneDepthContract(
            true, SceneDepthStorage::Tensor, false, 1280, 720, 0.1f, 1000.0f, false, true);
        EXPECT_TRUE(depth.available());
        EXPECT_TRUE(depth.valid());
        EXPECT_EQ(depth.encoding, SceneDepthEncoding::LinearView);
        EXPECT_EQ(depth.storage, SceneDepthStorage::Tensor);
        EXPECT_TRUE(depth.matchesRenderExtent(1280, 720));
        EXPECT_FALSE(depth.matchesRenderExtent(1920, 1080));
        EXPECT_FALSE(depth.requiresLinearization());
        EXPECT_TRUE(depth.flip_y);
    }

    TEST(SceneDepthContract, DescribesVulkanNdcImageDepth) {
        const auto depth = makeSceneDepthContract(
            true, SceneDepthStorage::VulkanImage, true, 1920, 1080, 0.01f, 500.0f, true, false);
        EXPECT_TRUE(depth.valid());
        EXPECT_EQ(depth.encoding, SceneDepthEncoding::VulkanNdc);
        EXPECT_EQ(depth.storage, SceneDepthStorage::VulkanImage);
        EXPECT_TRUE(depth.requiresLinearization());
        EXPECT_TRUE(depth.orthographic);
    }

    TEST(SceneDepthContract, RejectsMalformedAvailableDepth) {
        EXPECT_FALSE(makeSceneDepthContract(
                         true, SceneDepthStorage::Tensor, false, 0, 720, 0.1f, 1000.0f, false, false)
                         .valid());
        EXPECT_FALSE(makeSceneDepthContract(
                         true, SceneDepthStorage::Tensor, false, 1280, 720, 1.0f, 1.0f, false, false)
                         .valid());
        EXPECT_FALSE(makeSceneDepthContract(
                         true, SceneDepthStorage::Tensor, false, 1280, 720,
                         std::numeric_limits<float>::quiet_NaN(), 1000.0f, false, false)
                         .valid());
        EXPECT_FALSE(makeSceneDepthContract(
                         true, SceneDepthStorage::None, false, 1280, 720, 0.1f, 1000.0f, false, false)
                         .valid());
    }

} // namespace lfs::vis
