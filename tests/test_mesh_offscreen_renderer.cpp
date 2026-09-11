/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/mesh_offscreen_renderer.hpp"

#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace {

    [[nodiscard]] float projectedDepth(const glm::mat4& projection, const float view_depth) {
        const glm::vec4 clip = projection * glm::vec4(0.0f, 0.0f, -view_depth, 1.0f);
        return clip.z / clip.w;
    }

    TEST(MeshOffscreenDepthTest, LinearizesPerspectiveProjectionDepth) {
        const glm::mat4 projection =
            glm::perspective(glm::radians(55.0f), 16.0f / 9.0f, 0.1f, 250.0f);
        ASSERT_FLOAT_EQ(projection[2][3], -1.0f);

        for (const float expected_depth : std::array{0.1f, 1.0f, 37.0f, 200.0f}) {
            const float z_ndc = projectedDepth(projection, expected_depth);
            EXPECT_NEAR(
                lfs::vis::linearizeMeshViewDepth(z_ndc, projection),
                expected_depth,
                expected_depth * 2e-4f);
        }
    }

    TEST(MeshOffscreenDepthTest, LinearizesOrthographicProjectionDepth) {
        const glm::mat4 projection =
            glm::ortho(-4.0f, 4.0f, -3.0f, 3.0f, 0.25f, 80.0f);
        ASSERT_FLOAT_EQ(projection[2][3], 0.0f);

        for (const float expected_depth : std::array{0.25f, 1.0f, 23.0f, 79.0f}) {
            const float z_ndc = projectedDepth(projection, expected_depth);
            EXPECT_NEAR(
                lfs::vis::linearizeMeshViewDepth(z_ndc, projection),
                expected_depth,
                1e-4f);
        }
    }

} // namespace
