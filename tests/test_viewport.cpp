/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/viewport.hpp"
#include "rendering/rendering.hpp"

#include <gtest/gtest.h>

namespace {

    struct TestRasterRequest {
        glm::mat3 view_rotation{1.0f};
        glm::vec3 view_translation{0.0f};
        glm::ivec2 viewport_size{128, 128};
        float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        glm::vec3 background_color{0.0f};
        float far_plane = 100.0f;
        bool orthographic = false;
        float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
    };

    TestRasterRequest makeTestRasterRequest() {
        TestRasterRequest request;
        request.view_rotation = glm::mat3(1.0f);
        request.view_translation = glm::vec3(0.0f);
        request.viewport_size = {128, 128};
        request.focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        request.background_color = glm::vec3(0.0f);
        request.far_plane = 100.0f;
        return request;
    }

    std::optional<glm::vec2> projectWithClipSpaceMatrices(const glm::mat3& rotation,
                                                          const glm::vec3& translation,
                                                          const glm::ivec2& viewport_size,
                                                          const glm::vec3& world_point,
                                                          const float focal_length_mm,
                                                          const bool orthographic = false,
                                                          const float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE) {
        const glm::mat4 view = lfs::rendering::makeViewMatrix(rotation, translation);
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            viewport_size,
            focal_length_mm,
            orthographic,
            ortho_scale);
        const glm::vec4 clip = projection * view * glm::vec4(world_point, 1.0f);
        if (!orthographic && clip.w <= 1e-6f) {
            return std::nullopt;
        }

        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        return glm::vec2(
            (ndc.x * 0.5f + 0.5f) * static_cast<float>(viewport_size.x),
            (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(viewport_size.y));
    }

} // namespace

TEST(ViewportTest, InvalidWorldPositionUsesNamedSentinel) {
    Viewport viewport(100, 100);

    const glm::vec3 invalid = viewport.unprojectPixel(50.0f, 50.0f, -1.0f);

    EXPECT_FALSE(Viewport::isValidWorldPosition(invalid));
    EXPECT_FLOAT_EQ(invalid.x, Viewport::INVALID_WORLD_POS);
    EXPECT_FLOAT_EQ(invalid.y, Viewport::INVALID_WORLD_POS);
    EXPECT_FLOAT_EQ(invalid.z, Viewport::INVALID_WORLD_POS);
}

TEST(ViewportTest, DefaultCameraStartsAboveWorldYAxis) {
    Viewport viewport(100, 100);

    EXPECT_GT(viewport.camera.t.y, 0.0f);
}

TEST(ViewportTest, UnprojectPixelDependsOnScreenPixel) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.t = glm::vec3(0.0f);

    const glm::vec3 center = viewport.unprojectPixel(50.0f, 50.0f, 10.0f);
    const glm::vec3 top_left = viewport.unprojectPixel(0.0f, 0.0f, 10.0f);

    ASSERT_TRUE(Viewport::isValidWorldPosition(center));
    ASSERT_TRUE(Viewport::isValidWorldPosition(top_left));
    EXPECT_NEAR(center.x, 0.0f, 1e-4f);
    EXPECT_NEAR(center.y, 0.0f, 1e-4f);
    EXPECT_NEAR(center.z, -10.0f, 1e-4f);
    EXPECT_LT(top_left.x, center.x);
    EXPECT_GT(top_left.y, center.y);
    EXPECT_NEAR(top_left.z, center.z, 1e-4f);
}

TEST(ViewportTest, ViewportDataViewMatrixMatchesViewport) {
    Viewport viewport(160, 90);
    viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(3.0f, 2.0f, 5.0f),
        glm::vec3(0.0f, 0.0f, 0.0f));
    viewport.camera.t = glm::vec3(3.0f, 2.0f, 5.0f);

    const lfs::rendering::ViewportData data{
        .rotation = viewport.getRotationMatrix(),
        .translation = viewport.getTranslation(),
        .size = viewport.windowSize,
    };

    const glm::mat4 expected = viewport.getViewMatrix();
    const glm::mat4 actual = data.getViewMatrix();
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            EXPECT_NEAR(actual[col][row], expected[col][row], 1e-5f);
        }
    }
}

TEST(ViewportTest, ProjectWorldPointUsesVisualizerConvention) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.t = glm::vec3(0.0f);

    const auto center = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        glm::vec3(0.0f, 0.0f, -10.0f),
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);
    const auto behind = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        glm::vec3(0.0f, 0.0f, 10.0f),
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);

    ASSERT_TRUE(center.has_value());
    EXPECT_NEAR(center->x, 50.0f, 1e-4f);
    EXPECT_NEAR(center->y, 50.0f, 1e-4f);
    EXPECT_FALSE(behind.has_value());
}

TEST(ViewportTest, WasdAdvanceSupportsFlatAdditionalSpeedInVisualizerSpace) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.t = glm::vec3(0.0f);
    viewport.camera.pivot = glm::vec3(0.0f);

    constexpr float dt = 0.1f;
    constexpr float bonus = 20.0f;
    const float base_speed = viewport.camera.getWasdSpeed();
    for (int i = 0; i < 100; ++i)
        viewport.camera.advanceWasd(dt, true, false, false, false, false, false, bonus);

    const glm::vec3 t_before = viewport.camera.t;
    const glm::vec3 pivot_before = viewport.camera.pivot;
    viewport.camera.advanceWasd(dt, true, false, false, false, false, false, bonus);
    const glm::vec3 t_step = viewport.camera.t - t_before;
    const glm::vec3 pivot_step = viewport.camera.pivot - pivot_before;

    // Once the inertial velocity saturates, a settled step advances along -Z at
    // (wasdSpeed + bonus), confirming the bonus is additive, not multiplicative.
    EXPECT_FLOAT_EQ(viewport.camera.getWasdSpeed(), base_speed);
    EXPECT_NEAR(t_step.x, 0.0f, 1e-5f);
    EXPECT_NEAR(t_step.y, 0.0f, 1e-5f);
    EXPECT_NEAR(t_step.z, -(base_speed + bonus) * dt, 1e-4f);
    EXPECT_NEAR(glm::length(pivot_step - t_step), 0.0f, 1e-5f);
}

TEST(ViewportTest, OrbitDraggingRightMovesCameraLeftAroundPivot) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);

    viewport.camera.startRotateAroundCenter(glm::vec2(0.0f, 0.0f), 0.0f);
    viewport.camera.updateRotateAroundCenter(glm::vec2(100.0f, 0.0f), 0.0f);
    viewport.camera.endRotateAroundCenter();

    EXPECT_LT(viewport.camera.t.x, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.t - viewport.camera.getPivot()), 5.0f, 1e-4f);
}

TEST(ViewportTest, RotateDraggingUpTiltsCameraUp) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    viewport.camera.rotate(glm::vec2(0.0f, -100.0f));

    EXPECT_GT(lfs::rendering::cameraForward(viewport.camera.R).y, 0.0f);
}

TEST(ViewportTest, FpvRotateKeepsPivotInFrontOfCamera) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    viewport.camera.rotateFpv(glm::vec2(0.0f, -100.0f));

    const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
    const glm::vec3 to_pivot = glm::normalize(viewport.camera.getPivot() - viewport.camera.t);
    EXPECT_GT(forward.y, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-4f);
    EXPECT_NEAR(glm::dot(forward, to_pivot), 1.0f, 1e-4f);
}

TEST(ViewportTest, OrbitDraggingUpMovesCameraDownAroundPivot) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);

    viewport.camera.startRotateAroundCenter(glm::vec2(0.0f, 0.0f), 0.0f);
    viewport.camera.updateRotateAroundCenter(glm::vec2(0.0f, -100.0f), 0.0f);
    viewport.camera.endRotateAroundCenter();

    EXPECT_LT(viewport.camera.t.y, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.t - viewport.camera.getPivot()), 5.0f, 1e-4f);
}

TEST(ViewportTest, OrbitDraggingDownMovesCameraUpAroundPivot) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);

    viewport.camera.startRotateAroundCenter(glm::vec2(0.0f, 0.0f), 0.0f);
    viewport.camera.updateRotateAroundCenter(glm::vec2(0.0f, 100.0f), 0.0f);
    viewport.camera.endRotateAroundCenter();

    EXPECT_GT(viewport.camera.t.y, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.t - viewport.camera.getPivot()), 5.0f, 1e-4f);
}

TEST(ViewportTest, PanDraggingRightMovesCameraLeftAndDraggingUpMovesCameraDown) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f, 0.0f, 0.0f));
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    viewport.camera.translate(glm::vec2(20.0f, -30.0f));

    EXPECT_LT(viewport.camera.t.x, 0.0f);
    EXPECT_LT(viewport.camera.t.y, 0.0f);
    EXPECT_LT(viewport.camera.getPivot().x, 0.0f);
    EXPECT_LT(viewport.camera.getPivot().y, 0.0f);
    EXPECT_NEAR(viewport.camera.t.z, 5.0f, 1e-4f);
    EXPECT_NEAR(viewport.camera.getPivot().z, 0.0f, 1e-4f);
}

TEST(ViewportTest, PanDraggingMovesProjectedContentWithCursorInVisualizerSpace) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(3.0f, 2.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f, 0.0f, 0.0f));
    viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
        viewport.camera.t, viewport.camera.getPivot());
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    const glm::vec3 world_point(0.0f, 0.0f, 0.0f);
    const auto before = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        world_point,
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);
    ASSERT_TRUE(before.has_value());

    viewport.camera.translate(glm::vec2(20.0f, -30.0f));

    const auto after = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        world_point,
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);
    ASSERT_TRUE(after.has_value());

    EXPECT_GT(after->x, before->x);
    EXPECT_LT(after->y, before->y);
}

TEST(ViewportTest, ClipSpaceMatrixProjectionMatchesVisualizerProjectionWhenYawed) {
    const auto request = makeTestRasterRequest();
    const glm::mat3 rotation = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.25f, -5.0f));
    const glm::vec3 translation(0.0f, 0.0f, 0.0f);
    const glm::vec3 world_point(0.5f, -0.2f, -4.0f);

    const auto explicit_projection = lfs::rendering::projectWorldPoint(
        rotation, translation, request.viewport_size, world_point, request.focal_length_mm);
    const auto matrix_projection = projectWithClipSpaceMatrices(
        rotation, translation, request.viewport_size, world_point, request.focal_length_mm);

    ASSERT_TRUE(explicit_projection.has_value());
    ASSERT_TRUE(matrix_projection.has_value());
    EXPECT_NEAR(matrix_projection->x, explicit_projection->x, 1e-3f);
    EXPECT_NEAR(matrix_projection->y, explicit_projection->y, 1e-3f);
}

TEST(ViewportTest, OrthographicProjectionMatchesClipSpaceAndIgnoresDepth) {
    const auto request = makeTestRasterRequest();
    const glm::mat3 rotation = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(1.0f, 0.5f, 2.0f),
        glm::vec3(1.0f, 0.5f, -3.0f));
    const glm::vec3 translation(1.0f, 0.5f, 2.0f);
    constexpr float ortho_scale = 75.0f;

    const glm::vec3 near_point(1.5f, 0.8f, -4.0f);
    const glm::vec3 far_point(1.5f, 0.8f, -12.0f);

    const auto near_projection = lfs::rendering::projectWorldPoint(
        rotation, translation, request.viewport_size, near_point, request.focal_length_mm, true, ortho_scale);
    const auto far_projection = lfs::rendering::projectWorldPoint(
        rotation, translation, request.viewport_size, far_point, request.focal_length_mm, true, ortho_scale);
    const auto matrix_projection = projectWithClipSpaceMatrices(
        rotation, translation, request.viewport_size, near_point, request.focal_length_mm, true, ortho_scale);

    ASSERT_TRUE(near_projection.has_value());
    ASSERT_TRUE(far_projection.has_value());
    ASSERT_TRUE(matrix_projection.has_value());
    EXPECT_NEAR(near_projection->x, far_projection->x, 1e-4f);
    EXPECT_NEAR(near_projection->y, far_projection->y, 1e-4f);
    EXPECT_NEAR(matrix_projection->x, near_projection->x, 1e-3f);
    EXPECT_NEAR(matrix_projection->y, near_projection->y, 1e-3f);
}

TEST(ViewportTest, OrthographicUnprojectMatchesProjectedWorldPoint) {
    Viewport viewport(128, 128);
    viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(1.0f, 0.5f, 2.0f),
        glm::vec3(1.0f, 0.5f, -3.0f));
    viewport.camera.t = glm::vec3(1.0f, 0.5f, 2.0f);
    constexpr float ortho_scale = 75.0f;
    const glm::vec3 world_point(1.5f, 0.8f, -4.0f);

    const auto projected = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        world_point,
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
        true,
        ortho_scale);
    ASSERT_TRUE(projected.has_value());

    const glm::vec3 view = glm::transpose(viewport.camera.R) * (world_point - viewport.camera.t);
    const glm::vec3 unprojected = viewport.unprojectPixel(
        projected->x,
        projected->y,
        -view.z,
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
        true,
        ortho_scale);

    ASSERT_TRUE(Viewport::isValidWorldPosition(unprojected));
    EXPECT_NEAR(unprojected.x, world_point.x, 1e-4f);
    EXPECT_NEAR(unprojected.y, world_point.y, 1e-4f);
    EXPECT_NEAR(unprojected.z, world_point.z, 1e-4f);
}

TEST(ViewportTest, OrthographicUnprojectRejectsInvalidScale) {
    Viewport viewport(128, 128);

    const glm::vec3 invalid = viewport.unprojectPixel(
        64.0f,
        64.0f,
        10.0f,
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
        true,
        0.0f);

    EXPECT_FALSE(Viewport::isValidWorldPosition(invalid));
}

namespace {
    constexpr float kDroneDt = 1.0f / 60.0f;

    void expectOrthonormal(const glm::mat3& R) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(glm::length(R[i]), 1.0f, 1e-5f);
            for (int j = i + 1; j < 3; ++j) {
                EXPECT_NEAR(glm::dot(R[i], R[j]), 0.0f, 1e-5f);
            }
        }
    }
} // namespace

TEST(ViewportTest, DroneBrakesToRestFromForwardFlight) {
    Viewport viewport(100, 100);
    viewport.camera.enterDrone();

    for (int i = 0; i < 60; ++i)
        viewport.camera.advanceDrone(kDroneDt, true, false, false, false, false, false);
    EXPECT_TRUE(viewport.camera.hasDroneMotion());

    int settle_steps = 0;
    while (viewport.camera.hasDroneMotion() && settle_steps < 300) {
        viewport.camera.advanceDrone(kDroneDt, false, false, false, false, false, false);
        ++settle_steps;
    }
    EXPECT_FALSE(viewport.camera.hasDroneMotion());

    const glm::vec3 t_at_rest = viewport.camera.t;
    const glm::mat3 R_at_rest = viewport.camera.R;
    viewport.camera.advanceDrone(kDroneDt, false, false, false, false, false, false);
    EXPECT_EQ(viewport.camera.t, t_at_rest);
    EXPECT_EQ(viewport.camera.R, R_at_rest);
}

TEST(ViewportTest, DroneHorizontalFlightIgnoresGimbalPitch) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 10.0f, 0.0f);
    viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
        viewport.camera.t, viewport.camera.t + glm::normalize(glm::vec3(0.0f, -1.0f, -1.0f)));
    viewport.camera.enterDrone();

    for (int i = 0; i < 120; ++i) {
        viewport.camera.advanceDrone(kDroneDt, true, false, false, false, false, false);
        EXPECT_NEAR(viewport.camera.t.y, 10.0f, 1e-4f);
    }
    EXPECT_LT(viewport.camera.t.z, -0.5f);
}

TEST(ViewportTest, DroneClimbIsPureVertical) {
    Viewport viewport(100, 100);
    viewport.camera.enterDrone();
    const glm::vec3 t0 = viewport.camera.t;

    for (int i = 0; i < 120; ++i)
        viewport.camera.advanceDrone(kDroneDt, false, false, false, false, true, false);

    EXPECT_NEAR(viewport.camera.t.x, t0.x, 1e-4f);
    EXPECT_NEAR(viewport.camera.t.z, t0.z, 1e-4f);
    EXPECT_GT(viewport.camera.t.y, t0.y + 0.5f);
}

TEST(ViewportTest, DroneNoRollAfterEnterExit) {
    Viewport viewport(100, 100);
    viewport.camera.enterDrone();
    viewport.camera.initScreenPos(glm::vec2(0.0f));
    viewport.camera.droneLook(glm::vec2(150.0f, -80.0f));

    bool saw_bank = false;
    for (int i = 0; i < 90; ++i) {
        viewport.camera.advanceDrone(kDroneDt, true, false, false, true, false, false);
        saw_bank = saw_bank || std::abs(viewport.camera.R[0].y) > 0.01f;
    }
    EXPECT_TRUE(saw_bank);

    viewport.camera.finishDrone();
    EXPECT_FALSE(viewport.camera.hasDroneMotion());
    EXPECT_NEAR(viewport.camera.R[0].y, 0.0f, 1e-5f);
    expectOrthonormal(viewport.camera.R);
}

TEST(ViewportTest, DroneResyncsAfterExternalRotation) {
    Viewport viewport(100, 100);
    viewport.camera.enterDrone();
    for (int i = 0; i < 60; ++i)
        viewport.camera.advanceDrone(kDroneDt, true, false, false, false, false, false);

    viewport.camera.setAxisAlignedView(0, false);
    EXPECT_FALSE(viewport.camera.hasDroneMotion());

    const glm::vec3 t_after_view = viewport.camera.t;
    viewport.camera.advanceDrone(kDroneDt, false, false, false, false, false, false);

    EXPECT_EQ(viewport.camera.t, t_after_view);
    const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
    EXPECT_NEAR(forward.x, -1.0f, 1e-4f);
    EXPECT_NEAR(forward.y, 0.0f, 1e-4f);
    EXPECT_NEAR(forward.z, 0.0f, 1e-4f);
    expectOrthonormal(viewport.camera.R);
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            EXPECT_TRUE(std::isfinite(viewport.camera.R[col][row]));
}

TEST(ViewportTest, DroneBanksIntoYawTurnWhileFlyingForward) {
    Viewport viewport(100, 100);
    viewport.camera.enterDrone();
    for (int i = 0; i < 90; ++i)
        viewport.camera.advanceDrone(kDroneDt, true, false, false, false, false, false);

    viewport.camera.initScreenPos(glm::vec2(0.0f));
    viewport.camera.droneLook(glm::vec2(-400.0f, 0.0f));

    float max_left_bank = 0.0f;
    for (int i = 0; i < 30; ++i) {
        viewport.camera.advanceDrone(kDroneDt, true, false, false, false, false, false);
        max_left_bank = std::max(max_left_bank, viewport.camera.R[0].y);
    }
    EXPECT_GT(max_left_bank, 0.05f);
}
