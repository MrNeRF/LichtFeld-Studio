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

glm::vec2 applyHomography(const glm::mat3& h, const glm::vec2 ndc) {
    const glm::vec3 source = h * glm::vec3(ndc, 1.0f);
    EXPECT_GT(source.z, 0.0f);
    return glm::vec2(source) / source.z;
}

TEST(SceneReprojectionTest, HomographyIsExactOnThePivotPlane) {
    const Camera source = orbitCamera(20.0f, 4.0f);
    const Camera current = orbitCamera(28.0f, 3.6f);
    const glm::vec3 pivot(0.0f, 0.0f, 0.0f);
    const glm::mat3 h = lfs::vis::sceneReprojectionHomography(source.view, source.projection, current.view, pivot);
    const glm::mat3 camera_to_world = glm::transpose(glm::mat3(source.view));
    for (const glm::vec2 offset : {glm::vec2(0.0f), glm::vec2(0.7f, -0.2f), glm::vec2(-0.5f, 0.4f)}) {
        const glm::vec3 on_plane = pivot + camera_to_world * glm::vec3(offset, 0.0f);
        const glm::vec2 expected = project(source, on_plane).ndc;
        const glm::vec2 warped = applyHomography(h, project(current, on_plane).ndc);
        EXPECT_NEAR(warped.x, expected.x, 1e-4f);
        EXPECT_NEAR(warped.y, expected.y, 1e-4f);
    }
}

TEST(SceneReprojectionTest, HomographyKeepsAnUnmovedCameraStill) {
    const Camera camera = orbitCamera(20.0f, 4.0f);
    const glm::mat3 h = lfs::vis::sceneReprojectionHomography(camera.view, camera.projection, camera.view,
                                                              glm::vec3(0.0f));
    for (const glm::vec2 ndc : {glm::vec2(0.0f), glm::vec2(0.9f, -0.9f), glm::vec2(-0.3f, 0.6f)}) {
        const glm::vec2 warped = applyHomography(h, ndc);
        EXPECT_NEAR(warped.x, ndc.x, 1e-5f);
        EXPECT_NEAR(warped.y, ndc.y, 1e-5f);
    }
}

TEST(SceneReprojectionTest, PivotBehindTheCameraFallsBackToARotationWarp) {
    const Camera source = orbitCamera(20.0f, 4.0f);
    const glm::mat3 rotation(glm::rotate(glm::mat4(1.0f), glm::radians(6.0f), glm::vec3(0, 1, 0)));
    const glm::mat4 current_view = glm::mat4(rotation) * source.view;
    const glm::vec3 behind = glm::vec3(glm::inverse(source.view) * glm::vec4(0.0f, 0.0f, 5.0f, 1.0f));
    const glm::mat3 h = lfs::vis::sceneReprojectionHomography(source.view, source.projection, current_view, behind);
    const Camera current{current_view, source.projection};
    for (const auto& point : kScenePoints) {
        const glm::vec2 expected = project(source, point).ndc;
        const glm::vec2 warped = applyHomography(h, project(current, point).ndc);
        EXPECT_NEAR(warped.x, expected.x, 1e-3f);
        EXPECT_NEAR(warped.y, expected.y, 1e-3f);
    }
}
