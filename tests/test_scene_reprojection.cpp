/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/coordinate_conventions.hpp"
#include "rendering/passes/scene_reprojection.hpp"
#include "rendering/render_constants.hpp"

#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace {

    struct Camera {
        glm::mat4 view;
        glm::mat4 projection;
    };

    Camera orbitCamera(const float yaw_degrees, const float distance) {
        const glm::mat3 rotation(glm::rotate(glm::mat4(1.0f), glm::radians(yaw_degrees), glm::vec3(0, 1, 0)));
        const glm::vec3 position = rotation * glm::vec3(0.0f, 0.4f, distance);
        return {lfs::rendering::makeViewMatrix(rotation, position),
                lfs::rendering::createProjectionMatrix({1920, 1080}, 50.0f, false, 1.0f)};
    }

    struct Projected {
        glm::vec2 ndc;
        float depth;
    };

    Projected project(const Camera& camera, const glm::vec3& world) {
        const glm::vec4 clip = camera.projection * camera.view * glm::vec4(world, 1.0f);
        return {glm::vec2(clip) / clip.w, clip.w};
    }

    glm::vec2 warp(const glm::mat4& source_to_current, const glm::vec2 source_ndc, const float depth) {
        const glm::vec4 clip = source_to_current * glm::vec4(source_ndc * depth, depth, 1.0f);
        return glm::vec2(clip) / clip.w;
    }

    const std::array<glm::vec3, 5> kScenePoints{{
        {0.0f, 0.0f, 0.0f},
        {0.8f, -0.3f, 0.5f},
        {-1.2f, 0.6f, -0.9f},
        {0.3f, 1.1f, 1.4f},
        {-0.5f, -0.8f, 2.0f},
    }};

} // namespace

TEST(SceneReprojectionTest, SameCameraMapsEveryPixelToItself) {
    const Camera camera = orbitCamera(20.0f, 4.0f);
    const glm::mat4 m = lfs::vis::sceneReprojectionMatrix(camera.view, camera.projection, camera.view);
    for (const auto& point : kScenePoints) {
        const auto p = project(camera, point);
        const glm::vec2 warped = warp(m, p.ndc, p.depth);
        EXPECT_NEAR(warped.x, p.ndc.x, 1e-5f);
        EXPECT_NEAR(warped.y, p.ndc.y, 1e-5f);
    }
}

TEST(SceneReprojectionTest, MovedCameraSeesSourcePixelsWhereItProjectsTheirPoints) {
    const Camera source = orbitCamera(20.0f, 4.0f);
    const Camera current = orbitCamera(26.0f, 3.7f);
    const glm::mat4 m = lfs::vis::sceneReprojectionMatrix(source.view, source.projection, current.view);
    for (const auto& point : kScenePoints) {
        const auto from = project(source, point);
        const auto expected = project(current, point);
        const glm::vec2 warped = warp(m, from.ndc, from.depth);
        EXPECT_NEAR(warped.x, expected.ndc.x, 1e-4f);
        EXPECT_NEAR(warped.y, expected.ndc.y, 1e-4f);
    }
}

// Mirrors scene_reproject.frag: six fixed-point steps must recover the source
// pixel for an orbit step far larger than one frame at interactive rates.
TEST(SceneReprojectionTest, ShaderIterationFindsTheSourcePixel) {
    const Camera source = orbitCamera(20.0f, 4.0f);
    const Camera current = orbitCamera(28.0f, 3.6f);
    const glm::mat4 m = lfs::vis::sceneReprojectionMatrix(source.view, source.projection, current.view);
    for (const auto& point : kScenePoints) {
        const auto from = project(source, point);
        const glm::vec2 target = project(current, point).ndc;
        glm::vec2 guess = target;
        for (int i = 0; i < 6; ++i) {
            guess += target - warp(m, guess, from.depth);
        }
        EXPECT_NEAR(guess.x, from.ndc.x, 1e-3f);
        EXPECT_NEAR(guess.y, from.ndc.y, 1e-3f);
    }
}
