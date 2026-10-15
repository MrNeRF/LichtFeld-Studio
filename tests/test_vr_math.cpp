/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vr/vr_math.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <rendering/frame_contract.hpp>

namespace {
    using namespace lfs::vis::vr;

    constexpr float TOLERANCE = 1e-5f;
    const glm::quat IDENTITY{1.0f, 0.0f, 0.0f, 0.0f};

    void expectVecNear(const glm::vec3 actual, const glm::vec3 expected) {
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(actual[i], expected[i], TOLERANCE);
    }

    TEST(VrMath, AsymmetricFovMapsSignedEndpointsToTopLeftPixelBounds) {
        const EyeFov fov{-0.7f, 0.9f, 0.8f, -0.5f};
        const glm::ivec2 size{1200, 900};
        const auto intrinsics = eyeIntrinsics(fov, size);
        const auto project = [&](const float horizontal, const float vertical) {
            return glm::vec2(intrinsics.center_x + intrinsics.focal_x * std::tan(horizontal),
                             intrinsics.center_y - intrinsics.focal_y * std::tan(vertical));
        };
        const auto top_left = project(fov.left, fov.up);
        const auto bottom_right = project(fov.right, fov.down);
        EXPECT_NEAR(top_left.x, 0.0f, 1e-4f);
        EXPECT_NEAR(top_left.y, 0.0f, 1e-4f);
        EXPECT_NEAR(bottom_right.x, size.x, 1e-4f);
        EXPECT_NEAR(bottom_right.y, size.y, 1e-4f);
        EXPECT_NE(intrinsics.center_x, size.x * 0.5f);
        EXPECT_NE(intrinsics.center_y, size.y * 0.5f);
        lfs::rendering::FrameView view;
        view.size = size;
        view.intrinsics_override = intrinsics;
        EXPECT_FLOAT_EQ(view.getCameraIntrinsics().center_y, intrinsics.center_y);
    }

    TEST(VrMath, SymmetricFovAndInvalidProjectionInputs) {
        const auto intrinsics = eyeIntrinsics({-glm::quarter_pi<float>(), glm::quarter_pi<float>(),
                                               glm::quarter_pi<float>(), -glm::quarter_pi<float>()},
                                              {1000, 800});
        EXPECT_NEAR(intrinsics.focal_x, 500.0f, TOLERANCE);
        EXPECT_NEAR(intrinsics.focal_y, 400.0f, TOLERANCE);
        EXPECT_NEAR(intrinsics.center_x, 500.0f, TOLERANCE);
        EXPECT_NEAR(intrinsics.center_y, 400.0f, TOLERANCE);
        EXPECT_THROW((void)eyeIntrinsics({0.5f, -0.5f, 0.5f, -0.5f}, {100, 100}), std::invalid_argument);
        EXPECT_THROW((void)eyeIntrinsics({-0.5f, 0.5f, 0.5f, -0.5f}, {0, 100}), std::invalid_argument);
        EXPECT_THROW((void)eyeIntrinsics({-0.5f, glm::half_pi<float>(), 0.5f, -0.5f}, {100, 100}),
                     std::invalid_argument);
    }
    TEST(VrMath, FrameProjectionHonorsAsymmetricIntrinsicsAndDepthOverrides) {
        const EyeFov fov{-0.7f, 0.9f, 0.8f, -0.5f};
        lfs::rendering::FrameView view;
        view.size = {600, 450};
        view.subregion_full_size = {1200, 900};
        const auto original = view.getProjectionMatrix(0.2f, 500.0f);
        view.intrinsics_override = eyeIntrinsics(fov, view.cameraSize());
        const auto projection = view.getProjectionMatrix(0.2f, 500.0f);
        const auto ndc = [&](const float horizontal, const float vertical) {
            const glm::vec4 clip = projection * glm::vec4(std::tan(horizontal), std::tan(vertical), -1, 1);
            return glm::vec3(clip) / clip.w;
        };
        const auto top_left = ndc(fov.left, fov.up);
        const auto bottom_right = ndc(fov.right, fov.down);
        EXPECT_NEAR(top_left.x, -1.0f, TOLERANCE);
        EXPECT_NEAR(top_left.y, 1.0f, TOLERANCE);
        EXPECT_NEAR(bottom_right.x, 1.0f, TOLERANCE);
        EXPECT_NEAR(bottom_right.y, -1.0f, TOLERANCE);
        for (int col = 0; col < 4; ++col) {
            EXPECT_FLOAT_EQ(projection[col][2], original[col][2]);
            EXPECT_FLOAT_EQ(projection[col][3], original[col][3]);
        }
    }

    TEST(VrMath, FrameProjectionPreservesSymmetricDefaultsAndOrthographicMode) {
        lfs::rendering::FrameView view;
        view.size = {1200, 900};
        const auto original = view.getProjectionMatrix();
        view.intrinsics_override = view.getCameraIntrinsics();
        const auto calibrated = view.getProjectionMatrix();
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                EXPECT_NEAR(calibrated[col][row], original[col][row], TOLERANCE);
        view.orthographic = true;
        const auto ortho_with_override = view.getProjectionMatrix();
        view.intrinsics_override.reset();
        const auto ortho_without_override = view.getProjectionMatrix();
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                EXPECT_FLOAT_EQ(ortho_with_override[col][row], ortho_without_override[col][row]);
    }

    TEST(VrMath, PoseIsNormalizedRightHandedCameraToWorld) {
        const auto orientation = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0, 1, 0));
        const glm::vec3 position{2, 3, 4};
        const glm::mat4 pose = poseMatrix(orientation * 3.0f, position);
        expectVecNear(glm::vec3(pose[3]), position);
        expectVecNear(glm::mat3(pose) * glm::vec3(0, 0, -1), {-1, 0, 0});
        EXPECT_NEAR(glm::determinant(glm::mat3(pose)), 1.0f, TOLERANCE);
        const auto view = lfs::rendering::makeViewMatrix(glm::mat3(pose), position);
        expectVecNear(glm::vec3(view * pose * glm::vec4(1, 2, -3, 1)), {1, 2, -3});
        expectVecNear(glm::mat3(poseMatrix(glm::quat(0, 0, 0, 0), {})) * glm::vec3(1, 0, 0),
                      {1, 0, 0});
    }

    TEST(VrMath, YawOnlyRemovesPitchRollAndHandlesVerticalHead) {
        const auto yaw = glm::angleAxis(0.6f, glm::vec3(0, 1, 0));
        const auto pitch = glm::angleAxis(0.4f, glm::vec3(1, 0, 0));
        const auto roll = glm::angleAxis(-0.3f, glm::vec3(0, 0, 1));
        const auto result = yawOnlyOrientation((yaw * pitch * roll) * 2.0f);
        EXPECT_NEAR(glm::length(result), 1.0f, TOLERANCE);
        expectVecNear(result * glm::vec3(0, 1, 0), {0, 1, 0});
        expectVecNear(result * glm::vec3(0, 0, -1), yaw * glm::vec3(0, 0, -1));
        const auto vertical = yaw * glm::angleAxis(glm::half_pi<float>(), glm::vec3(1, 0, 0));
        expectVecNear(yawOnlyOrientation(vertical) * glm::vec3(0, 0, -1),
                      yaw * glm::vec3(0, 0, -1));
        expectVecNear(yawOnlyOrientation(glm::quat(0, 0, 0, 0)) * glm::vec3(0, 0, -1),
                      {0, 0, -1});
    }

    TEST(VrMath, RigScalesEyeSeparationButNeverCameraRotation) {
        NavigationRig rig;
        rig.position = {10, 20, 30};
        rig.orientation = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0, 1, 0));
        rig.scene_units_per_meter = 25.0f;
        const glm::vec3 head{0.7f, 1.6f, -0.3f};
        const auto left = rig.cameraPose(IDENTITY, head + glm::vec3(-0.032f, 0, 0));
        const auto right = rig.cameraPose(IDENTITY, head + glm::vec3(0.032f, 0, 0));
        EXPECT_NEAR(glm::length(right.translation - left.translation), 0.064f * 25.0f, TOLERANCE);
        expectVecNear(right.translation - left.translation, {0, 0, -1.6f});
        expectVecNear(left.rotation[0], {0, 0, -1});
        EXPECT_NEAR(glm::determinant(left.rotation), 1.0f, TOLERANCE);
        const auto eye_orientation = glm::angleAxis(0.2f, glm::vec3(1, 0, 0));
        expectVecNear(rig.cameraPose(eye_orientation, head).rotation[2],
                      rig.orientation * (eye_orientation * glm::vec3(0, 0, 1)));
    }

    TEST(VrMath, SnapTurnPivotsAroundActualHeadCenter) {
        NavigationRig rig;
        rig.position = {4, -2, 3};
        rig.orientation = glm::angleAxis(0.4f, glm::vec3(0, 1, 0));
        rig.scene_units_per_meter = 7.0f;
        const glm::vec3 head{1.2f, 1.7f, -0.8f};
        const glm::vec3 eye = head + glm::vec3(0.032f, 0, 0);
        const auto center_before = rig.cameraPose(IDENTITY, head);
        const auto eye_before = rig.cameraPose(IDENTITY, eye);
        const auto turn = glm::angleAxis(-glm::half_pi<float>(), glm::vec3(0, 1, 0));
        rig.snapTurn(-glm::half_pi<float>(), head);
        const auto center_after = rig.cameraPose(IDENTITY, head);
        const auto eye_after = rig.cameraPose(IDENTITY, eye);
        expectVecNear(center_after.translation, center_before.translation);
        expectVecNear(eye_after.translation - center_after.translation,
                      turn * (eye_before.translation - center_before.translation));
        expectVecNear(center_after.rotation[2], turn * center_before.rotation[2]);
        rig.snapTurn(glm::half_pi<float>(), head);
        expectVecNear(rig.position, {4, -2, 3});
    }

    TEST(VrMath, ScalarDeadzoneRescalesAndSaturates) {
        EXPECT_FLOAT_EQ(deadzone(0.2f, 0.2f), 0.0f);
        EXPECT_FLOAT_EQ(deadzone(-0.1f, 0.2f), 0.0f);
        EXPECT_NEAR(deadzone(0.6f, 0.2f), 0.5f, TOLERANCE);
        EXPECT_NEAR(deadzone(-0.6f, 0.2f), -0.5f, TOLERANCE);
        EXPECT_FLOAT_EQ(deadzone(2.0f, 0.2f), 1.0f);
        EXPECT_FLOAT_EQ(deadzone(-2.0f, 0.2f), -1.0f);
        EXPECT_FLOAT_EQ(deadzone(0.5f, 1.0f), 0.0f);
        EXPECT_FLOAT_EQ(deadzone(std::numeric_limits<float>::quiet_NaN(), 0.2f), 0.0f);
    }

    TEST(VrMath, RadialDeadzonePreservesDirectionAndUnitDisk) {
        const auto zero = deadzone(glm::vec2(0.1f, 0.1f), 0.2f);
        EXPECT_FLOAT_EQ(glm::length(zero), 0.0f);
        const auto result = deadzone(glm::vec2(0.36f, 0.48f), 0.2f);
        EXPECT_NEAR(result.x, 0.3f, TOLERANCE);
        EXPECT_NEAR(result.y, 0.4f, TOLERANCE);
        const auto diagonal = deadzone(glm::vec2(1, 1), 0.2f);
        EXPECT_NEAR(glm::length(diagonal), 1.0f, TOLERANCE);
        EXPECT_NEAR(diagonal.x, diagonal.y, TOLERANCE);
        EXPECT_FLOAT_EQ(glm::length(deadzone(glm::vec2(0), 0.0f)), 0.0f);
    }

    TEST(VrMath, SnapTurnHysteresisRequiresNeutralBetweenSignedPulses) {
        SnapTurnHysteresis snap;
        EXPECT_EQ(snap.update(0.69f), 0);
        EXPECT_EQ(snap.update(0.7f), 1);
        EXPECT_EQ(snap.update(1.0f), 0);
        EXPECT_EQ(snap.update(0.5f), 0);
        EXPECT_EQ(snap.update(-1.0f), 0);
        EXPECT_EQ(snap.update(-0.3f), 0);
        EXPECT_EQ(snap.update(-0.7f), -1);
        snap.reset();
        EXPECT_EQ(snap.update(0.9f), 1);
        EXPECT_EQ(snap.update(std::numeric_limits<float>::quiet_NaN()), 0);
        EXPECT_EQ(snap.update(0.9f), 1);
        EXPECT_THROW(SnapTurnHysteresis(0.3f, 0.7f), std::invalid_argument);
        EXPECT_THROW(SnapTurnHysteresis(0.7f, -0.1f), std::invalid_argument);
    }
} // namespace
