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

    TEST(VrMath, AsymmetricEyeFovMapsToImageBoundsAndProjection) {
        const EyeFov fov{-0.7f, 0.9f, 0.8f, -0.5f};
        const glm::ivec2 size{1200, 900};
        const auto intrinsics = eyeIntrinsics(fov, size);
        ASSERT_TRUE(intrinsics);
        const auto pixel = [&](const float horizontal, const float vertical) {
            return glm::vec2(intrinsics->center_x + intrinsics->focal_x * std::tan(horizontal),
                             intrinsics->center_y - intrinsics->focal_y * std::tan(vertical));
        };
        EXPECT_NEAR(pixel(fov.left, fov.up).x, 0.0f, 1e-4f);
        EXPECT_NEAR(pixel(fov.left, fov.up).y, 0.0f, 1e-4f);
        EXPECT_NEAR(pixel(fov.right, fov.down).x, size.x, 1e-4f);
        EXPECT_NEAR(pixel(fov.right, fov.down).y, size.y, 1e-4f);

        lfs::rendering::FrameView view;
        view.size = size;
        const auto symmetric = view.getProjectionMatrix(0.2f, 500.0f);
        view.intrinsics_override = intrinsics;
        const auto projection = view.getProjectionMatrix(0.2f, 500.0f);
        const auto ndc = [&](const float horizontal, const float vertical) {
            const glm::vec4 clip = projection * glm::vec4(std::tan(horizontal), std::tan(vertical), -1, 1);
            return glm::vec3(clip) / clip.w;
        };
        EXPECT_NEAR(ndc(fov.left, fov.up).x, -1.0f, TOLERANCE);
        EXPECT_NEAR(ndc(fov.left, fov.up).y, 1.0f, TOLERANCE);
        EXPECT_NEAR(ndc(fov.right, fov.down).x, 1.0f, TOLERANCE);
        EXPECT_NEAR(ndc(fov.right, fov.down).y, -1.0f, TOLERANCE);
        for (int col = 0; col < 4; ++col) {
            EXPECT_FLOAT_EQ(projection[col][2], symmetric[col][2]);
            EXPECT_FLOAT_EQ(projection[col][3], symmetric[col][3]);
        }

        EXPECT_FALSE(eyeIntrinsics({0.5f, -0.5f, 0.5f, -0.5f}, {100, 100}));
        EXPECT_FALSE(eyeIntrinsics({-0.5f, 0.5f, 0.5f, -0.5f}, {0, 100}));
        EXPECT_FALSE(eyeIntrinsics({-0.5f, glm::half_pi<float>(), 0.5f, -0.5f}, {100, 100}));
    }

    TEST(VrMath, RecenterPutsHeadAtTargetWithTargetHeadingOnly) {
        NavigationRig rig;
        rig.scene_units_per_meter = 4.0f;
        const auto head_pitch = glm::angleAxis(0.3f, glm::vec3(1, 0, 0));
        const auto head_orientation = glm::angleAxis(0.9f, glm::vec3(0, 1, 0)) * head_pitch;
        const glm::vec3 head_position{0.4f, 1.6f, -0.2f};
        const auto target_yaw = glm::angleAxis(-0.5f, glm::vec3(0, 1, 0));
        const glm::vec3 target{3, 2, 1};
        rig.recenter(head_orientation, head_position, target_yaw * glm::angleAxis(0.7f, glm::vec3(1, 0, 0)), target);
        const auto pose = rig.cameraPose(head_orientation, head_position);
        expectVecNear(pose.translation, target);
        // Heading follows the target; the tracked pitch stays physical.
        expectVecNear(pose.rotation * glm::vec3(0, 0, -1), target_yaw * (head_pitch * glm::vec3(0, 0, -1)));
    }

    TEST(VrMath, SnapTurnKeepsTrackedHeadFixed) {
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
        expectVecNear(center_after.translation, center_before.translation);
        expectVecNear(rig.cameraPose(IDENTITY, eye).translation - center_after.translation,
                      turn * (eye_before.translation - center_before.translation));
        expectVecNear(center_after.rotation[2], turn * center_before.rotation[2]);
        rig.snapTurn(glm::half_pi<float>(), head);
        expectVecNear(rig.position, {4, -2, 3});
    }

    TEST(VrMath, DeadzoneRescalesSaturatesAndRejectsInvalidInput) {
        EXPECT_FLOAT_EQ(deadzone(0.2f, 0.2f), 0.0f);
        EXPECT_NEAR(deadzone(0.6f, 0.2f), 0.5f, TOLERANCE);
        EXPECT_NEAR(deadzone(-0.6f, 0.2f), -0.5f, TOLERANCE);
        EXPECT_FLOAT_EQ(deadzone(-2.0f, 0.2f), -1.0f);
        EXPECT_FLOAT_EQ(deadzone(0.5f, 1.0f), 0.0f);
        EXPECT_FLOAT_EQ(deadzone(std::numeric_limits<float>::quiet_NaN(), 0.2f), 0.0f);
        EXPECT_FLOAT_EQ(glm::length(deadzone(glm::vec2(0.1f, 0.1f), 0.2f)), 0.0f);
        const auto radial = deadzone(glm::vec2(0.36f, 0.48f), 0.2f);
        EXPECT_NEAR(radial.x, 0.3f, TOLERANCE);
        EXPECT_NEAR(radial.y, 0.4f, TOLERANCE);
        EXPECT_NEAR(glm::length(deadzone(glm::vec2(1, 1), 0.2f)), 1.0f, TOLERANCE);
    }

    TEST(VrMath, SnapTurnHysteresisNeedsNeutralBetweenPulses) {
        SnapTurnHysteresis snap;
        EXPECT_EQ(snap.update(0.69f), 0);
        EXPECT_EQ(snap.update(0.7f), 1);
        EXPECT_EQ(snap.update(1.0f), 0);
        EXPECT_EQ(snap.update(-1.0f), 0);
        EXPECT_EQ(snap.update(-0.3f), 0);
        EXPECT_EQ(snap.update(-0.7f), -1);
        EXPECT_EQ(snap.update(std::numeric_limits<float>::quiet_NaN()), 0);
        EXPECT_EQ(snap.update(0.9f), 1);
    }
} // namespace
