/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vr/vr_hand_model.hpp"
#include "vr/vr_hands.hpp"
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

    constexpr float GRAB_TOLERANCE = 1e-4f;

    // World position of a tracked point.
    [[nodiscard]] glm::vec3 worldOf(const NavigationRig& rig, const glm::vec3 tracked) {
        return rig.cameraPose(IDENTITY, tracked).translation;
    }

    void expectScenePointNear(const glm::vec3 actual, const glm::vec3 expected) {
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(actual[i], expected[i], GRAB_TOLERANCE);
    }

    [[nodiscard]] NavigationRig grabRig() {
        NavigationRig rig;
        rig.position = {4, -2, 3};
        rig.orientation = glm::angleAxis(0.4f, glm::vec3(0, 1, 0));
        rig.scene_units_per_meter = 7.0f;
        return rig;
    }

    TEST(VrMath, OneGripCarriesAndTurnsTheScene) {
        auto rig = grabRig();
        const GrabHand before{{0.2f, 1.1f, -0.4f}, glm::angleAxis(0.3f, glm::vec3(1, 0, 0))};
        const GrabHand after{{-0.1f, 1.4f, -0.2f}, glm::angleAxis(1.1f, glm::normalize(glm::vec3(1, 2, -1)))};
        const glm::vec3 held{0.5f, 0.9f, -1.2f};
        const glm::vec3 world = worldOf(rig, held);

        GrabManipulation grab;
        grab.update(rig, before, std::nullopt);
        EXPECT_TRUE(grab.active());
        grab.update(rig, after, std::nullopt);
        // The scene keeps its place in the hand, at its size.
        const glm::vec3 carried =
            after.position + after.orientation * (glm::inverse(before.orientation) * (held - before.position));
        expectScenePointNear(worldOf(rig, carried), world);
        EXPECT_FLOAT_EQ(rig.scene_units_per_meter, 7.0f);

        // The right hand alone behaves the same.
        auto mirrored = grabRig();
        GrabManipulation right_grab;
        right_grab.update(mirrored, std::nullopt, before);
        right_grab.update(mirrored, std::nullopt, after);
        expectScenePointNear(worldOf(mirrored, carried), world);
    }

    TEST(VrMath, TwoGripsScaleAboutTheirMidpoint) {
        auto rig = grabRig();
        const glm::vec3 midpoint{0.1f, 1.2f, -0.5f};
        const glm::vec3 half{0.15f, 0.0f, 0.0f};
        const glm::vec3 held{0.4f, 1.0f, -0.9f};
        const glm::vec3 world = worldOf(rig, held);
        const glm::quat orientation_before = rig.orientation;

        GrabManipulation grab;
        grab.update(rig, GrabHand{midpoint - half, IDENTITY}, GrabHand{midpoint + half, IDENTITY});
        grab.update(rig, GrabHand{midpoint - 2.0f * half, IDENTITY}, GrabHand{midpoint + 2.0f * half, IDENTITY});
        // Hands twice as far apart: the scene looks twice as large around the midpoint.
        EXPECT_NEAR(rig.scene_units_per_meter, 3.5f, GRAB_TOLERANCE);
        expectScenePointNear(worldOf(rig, midpoint + 2.0f * (held - midpoint)), world);
        expectScenePointNear(rig.orientation * glm::vec3(1, 0, 0), orientation_before * glm::vec3(1, 0, 0));
        expectScenePointNear(rig.orientation * glm::vec3(0, 1, 0), orientation_before * glm::vec3(0, 1, 0));
    }

    TEST(VrMath, TwoGripsTurnTheSceneWithTheLineBetweenThem) {
        auto rig = grabRig();
        const glm::vec3 midpoint{-0.2f, 1.3f, -0.4f};
        const glm::vec3 half{0.2f, 0.0f, 0.0f};
        const glm::vec3 held{0.3f, 0.8f, -1.0f};
        const glm::vec3 world = worldOf(rig, held);
        const glm::quat turn = glm::angleAxis(0.8f, glm::normalize(glm::vec3(0.3f, 1.0f, -0.5f)));
        const glm::vec3 moved{0.25f, -0.1f, 0.15f};

        GrabManipulation grab;
        grab.update(rig, GrabHand{midpoint - half, IDENTITY}, GrabHand{midpoint + half, IDENTITY});
        grab.update(rig, GrabHand{midpoint + moved - turn * half, turn}, GrabHand{midpoint + moved + turn * half, turn});
        expectScenePointNear(worldOf(rig, midpoint + moved + turn * (held - midpoint)), world);
        EXPECT_NEAR(rig.scene_units_per_meter, 7.0f, GRAB_TOLERANCE);

        // Twisting both wrists about the line between the hands rolls the scene.
        auto rolled = grabRig();
        const glm::quat roll = glm::angleAxis(0.6f, glm::vec3(1, 0, 0));
        GrabManipulation roll_grab;
        roll_grab.update(rolled, GrabHand{midpoint - half, IDENTITY}, GrabHand{midpoint + half, IDENTITY});
        roll_grab.update(rolled, GrabHand{midpoint - half, roll}, GrabHand{midpoint + half, roll});
        expectScenePointNear(worldOf(rolled, midpoint + roll * (held - midpoint)), world);
    }

    TEST(VrMath, GrippingAndReleasingNeverMovesTheView) {
        auto rig = grabRig();
        const auto reference = grabRig();
        const GrabHand left{{-0.2f, 1.0f, -0.3f}, glm::angleAxis(0.5f, glm::vec3(0, 0, 1))};
        const GrabHand right{{0.3f, 1.2f, -0.4f}, glm::angleAxis(-0.2f, glm::vec3(0, 1, 0))};
        const GrabHand elsewhere{{0.9f, 0.4f, 0.6f}, glm::angleAxis(1.3f, glm::vec3(1, 0, 0))};
        const auto expectUnchanged = [&] {
            expectScenePointNear(rig.position, reference.position);
            expectScenePointNear(rig.orientation * glm::vec3(1, 0, 0), reference.orientation * glm::vec3(1, 0, 0));
            expectScenePointNear(rig.orientation * glm::vec3(0, 0, 1), reference.orientation * glm::vec3(0, 0, 1));
            EXPECT_FLOAT_EQ(rig.scene_units_per_meter, reference.scene_units_per_meter);
        };

        GrabManipulation grab;
        EXPECT_FALSE(grab.active());
        // Each change of the gripping hands only sets what is followed next,
        // wherever the hands are by then.
        grab.update(rig, left, std::nullopt);
        expectUnchanged();
        grab.update(rig, left, right);
        expectUnchanged();
        grab.update(rig, std::nullopt, elsewhere);
        expectUnchanged();
        grab.update(rig, std::nullopt, std::nullopt);
        EXPECT_FALSE(grab.active());
        grab.update(rig, elsewhere, std::nullopt);
        expectUnchanged();
        // Still hands, and a hand whose tracking is not a number, move nothing.
        grab.update(rig, elsewhere, std::nullopt);
        expectUnchanged();
        grab.update(rig, GrabHand{glm::vec3(std::numeric_limits<float>::quiet_NaN()), IDENTITY}, std::nullopt);
        EXPECT_FALSE(grab.active());
        expectUnchanged();
        grab.update(rig, left, std::nullopt);
        grab.reset();
        grab.update(rig, elsewhere, std::nullopt);
        expectUnchanged();
    }

    TEST(VrMath, GripsTooCloseOnlyDragTheScene) {
        auto rig = grabRig();
        const glm::quat orientation_before = rig.orientation;
        const glm::vec3 midpoint{0.0f, 1.0f, -0.3f};
        const glm::vec3 half{0.5f * GrabManipulation::kMinHandDistance * 0.5f, 0.0f, 0.0f};
        const glm::vec3 moved{0.1f, 0.2f, -0.3f};
        const glm::vec3 world = worldOf(rig, midpoint);

        GrabManipulation grab;
        grab.update(rig, GrabHand{midpoint - half, IDENTITY}, GrabHand{midpoint + half, IDENTITY});
        // Closer than the minimum: the spread and the wrists are ignored.
        const glm::quat twist = glm::angleAxis(1.0f, glm::vec3(0, 1, 0));
        grab.update(rig, GrabHand{midpoint + moved - 0.5f * half, twist}, GrabHand{midpoint + moved + 0.5f * half, twist});
        expectScenePointNear(worldOf(rig, midpoint + moved), world);
        EXPECT_FLOAT_EQ(rig.scene_units_per_meter, 7.0f);
        expectScenePointNear(rig.orientation * glm::vec3(1, 0, 0), orientation_before * glm::vec3(1, 0, 0));
    }

    TEST(VrMath, GrabScaleStaysWithinThePreferenceRange) {
        NavigationRig rig;
        const auto hands = [](const float half_distance) {
            return std::pair{GrabHand{{-half_distance, 1.0f, 0.0f}, IDENTITY}, GrabHand{{half_distance, 1.0f, 0.0f}, IDENTITY}};
        };
        GrabManipulation grab;
        for (int i = 0; i < 8; ++i) {
            // Spread from 6 cm to 6 m, release, and again: 100 times larger each time.
            const auto [left_near, right_near] = hands(0.03f);
            const auto [left_far, right_far] = hands(3.0f);
            grab.update(rig, left_near, right_near);
            grab.update(rig, left_far, right_far);
            grab.update(rig, std::nullopt, std::nullopt);
        }
        EXPECT_FLOAT_EQ(rig.scene_units_per_meter, GrabManipulation::kMinScale);
        for (int i = 0; i < 8; ++i) {
            const auto [left_near, right_near] = hands(0.03f);
            const auto [left_far, right_far] = hands(3.0f);
            grab.update(rig, left_far, right_far);
            grab.update(rig, left_near, right_near);
            grab.update(rig, std::nullopt, std::nullopt);
        }
        EXPECT_FLOAT_EQ(rig.scene_units_per_meter, GrabManipulation::kMaxScale);
    }

    TEST(VrMath, GrabComposesWithOtherRigMotion) {
        auto rig = grabRig();
        const GrabHand hand{{0.2f, 1.1f, -0.4f}, glm::angleAxis(0.3f, glm::vec3(1, 0, 0))};
        const glm::vec3 head{0.0f, 1.6f, 0.0f};

        GrabManipulation grab;
        grab.update(rig, hand, std::nullopt);
        // A snap turn while holding still is kept: the grab adds only the
        // hand's own motion since the last update.
        rig.snapTurn(glm::half_pi<float>(), head);
        const auto turned = rig;
        grab.update(rig, hand, std::nullopt);
        expectScenePointNear(rig.position, turned.position);
        expectScenePointNear(rig.orientation * glm::vec3(1, 0, 0), turned.orientation * glm::vec3(1, 0, 0));

        const glm::vec3 step{0.0f, 0.0f, -0.25f};
        const glm::vec3 world = worldOf(rig, hand.position);
        grab.update(rig, GrabHand{hand.position + step, hand.orientation}, std::nullopt);
        expectScenePointNear(worldOf(rig, hand.position + step), world);
    }

    constexpr float FRAME = 1.0f / 90.0f;

    [[nodiscard]] float angleBetween(const glm::quat a, const glm::quat b) {
        return 2.0f * std::acos(std::min(1.0f, std::abs(glm::dot(a, b))));
    }

    TEST(VrMath, HandSmootherStartsOnTheHandAndRestsOnIt) {
        const GrabHand hand{{0.2f, 1.1f, -0.4f}, glm::angleAxis(0.3f, glm::vec3(1, 0, 0))};
        HandSmoother smoother;
        // No jump when a grip closes: the first hand is returned as it is.
        expectVecNear(smoother.update(hand, 0.0f).position, hand.position);
        for (int i = 0; i < 90; ++i) {
            const GrabHand steady = smoother.update(hand, FRAME);
            expectVecNear(steady.position, hand.position);
            EXPECT_NEAR(angleBetween(steady.orientation, hand.orientation), 0.0f, 1e-3f);
        }
    }

    TEST(VrMath, HandSmootherDampsTremor) {
        const glm::vec3 rest{0.2f, 1.1f, -0.4f};
        const float shake = 0.001f;              // 1 mm
        const float wobble = glm::radians(0.5f); // of the wrist
        HandSmoother smoother;
        float moved = 0.0f;
        float turned = 0.0f;
        for (int i = 0; i < 270; ++i) {
            // A 10 Hz tremor, as a hand held still shows.
            const float phase = std::sin(glm::two_pi<float>() * 10.0f * static_cast<float>(i) * FRAME);
            const GrabHand steady = smoother.update(
                {rest + glm::vec3(shake * phase, 0.0f, 0.0f), glm::angleAxis(wobble * phase, glm::vec3(0, 0, 1))},
                FRAME);
            if (i >= 90) {
                moved = std::max(moved, std::abs(steady.position.x - rest.x));
                turned = std::max(turned, angleBetween(steady.orientation, IDENTITY));
            }
        }
        EXPECT_TRUE(moved < 0.4f * shake);
        EXPECT_TRUE(turned < 0.4f * wobble);
    }

    TEST(VrMath, HandSmootherKeepsUpWithAMovingHand) {
        const glm::vec3 speed{0.5f, 0.0f, 0.0f}; // m/s
        const float turn_speed = 2.0f;           // rad/s
        HandSmoother smoother;
        GrabHand hand;
        GrabHand steady;
        for (int i = 0; i <= 90; ++i) {
            const float time = static_cast<float>(i) * FRAME;
            hand = {speed * time, glm::angleAxis(turn_speed * time, glm::vec3(0, 1, 0))};
            steady = smoother.update(hand, i == 0 ? 0.0f : FRAME);
        }
        EXPECT_TRUE(glm::length(hand.position - steady.position) < 0.01f);
        EXPECT_TRUE(angleBetween(hand.orientation, steady.orientation) < glm::radians(4.0f));
    }

    TEST(VrMath, HandSmootherRestartsAfterAResetOrAnInvalidHand) {
        const GrabHand near_hand{{0.0f, 1.0f, 0.0f}, IDENTITY};
        const GrabHand far_hand{{0.6f, 1.4f, -0.3f}, glm::angleAxis(1.0f, glm::vec3(0, 1, 0))};
        HandSmoother smoother;
        static_cast<void>(smoother.update(near_hand, FRAME));
        static_cast<void>(smoother.update(near_hand, FRAME));
        smoother.reset();
        expectVecNear(smoother.update(far_hand, FRAME).position, far_hand.position);

        const float nan = std::numeric_limits<float>::quiet_NaN();
        const GrabHand lost = smoother.update({{nan, 0.0f, 0.0f}, IDENTITY}, FRAME);
        EXPECT_TRUE(std::isnan(lost.position.x));
        // Tracking is back somewhere else: nothing of the old hand is left.
        expectVecNear(smoother.update(near_hand, FRAME).position, near_hand.position);
    }

    TEST(VrMath, EyeViewProjectionMapsTheFovToVulkanClipSpace) {
        const EyeFov fov{-0.7f, 0.9f, 0.8f, -0.5f};
        const glm::quat orientation = glm::angleAxis(0.4f, glm::normalize(glm::vec3(0.2f, 1.0f, -0.1f)));
        const glm::vec3 position{0.3f, 1.6f, -0.2f};
        const float near_plane = 0.02f, far_plane = 20.0f;
        const glm::mat4 matrix = eyeViewProjection(fov, orientation, position, near_plane, far_plane);
        // A point given in the eye's own frame: -Z ahead, +X right, +Y up.
        const auto clip = [&](const glm::vec3 in_eye) {
            const glm::vec4 projected = matrix * glm::vec4(position + orientation * in_eye, 1.0f);
            return glm::vec3(projected) / projected.w;
        };
        const float depth = 0.5f;
        // The edges of the field of view land on the edges of the image, up at -Y.
        EXPECT_NEAR(clip({std::tan(fov.left) * depth, 0.0f, -depth}).x, -1.0f, 1e-4f);
        EXPECT_NEAR(clip({std::tan(fov.right) * depth, 0.0f, -depth}).x, 1.0f, 1e-4f);
        EXPECT_NEAR(clip({0.0f, std::tan(fov.up) * depth, -depth}).y, -1.0f, 1e-4f);
        EXPECT_NEAR(clip({0.0f, std::tan(fov.down) * depth, -depth}).y, 1.0f, 1e-4f);
        // Depth runs from 0 at the near plane to 1 at the far plane.
        EXPECT_NEAR(clip({0.0f, 0.0f, -near_plane}).z, 0.0f, 1e-4f);
        EXPECT_NEAR(clip({0.0f, 0.0f, -far_plane}).z, 1.0f, 1e-4f);
        EXPECT_TRUE(clip({0.0f, 0.0f, -depth}).z > 0.0f && clip({0.0f, 0.0f, -depth}).z < 1.0f);
    }

    // An open right hand: fingers along -Z, back of the hand at +Y.
    [[nodiscard]] HandJoints openHand() {
        return {{
            {0.000f, 0.0f, -0.045f},
            {0.000f, 0.0f, 0.000f},
            {-0.024f, -0.008f, -0.012f},
            {-0.052f, -0.014f, -0.045f},
            {-0.068f, -0.016f, -0.075f},
            {-0.078f, -0.016f, -0.100f},
            {-0.022f, 0.0f, -0.020f},
            {-0.027f, 0.0f, -0.090f},
            {-0.029f, 0.0f, -0.130f},
            {-0.030f, 0.0f, -0.155f},
            {-0.030f, 0.0f, -0.175f},
            {-0.006f, 0.0f, -0.022f},
            {-0.004f, 0.0f, -0.093f},
            {-0.004f, 0.0f, -0.137f},
            {-0.004f, 0.0f, -0.165f},
            {-0.004f, 0.0f, -0.187f},
            {0.008f, 0.0f, -0.020f},
            {0.016f, 0.0f, -0.087f},
            {0.018f, 0.0f, -0.127f},
            {0.019f, 0.0f, -0.153f},
            {0.019f, 0.0f, -0.173f},
            {0.020f, 0.0f, -0.016f},
            {0.034f, 0.0f, -0.076f},
            {0.038f, 0.0f, -0.106f},
            {0.040f, 0.0f, -0.125f},
            {0.041f, 0.0f, -0.142f},
        }};
    }

    [[nodiscard]] HandPose openHandPose() {
        HandPose pose;
        const HandJoints joints = openHand();
        for (std::size_t joint = 0; joint < pose.size(); ++joint)
            pose[joint] = {joints[joint], IDENTITY};
        return pose;
    }

    TEST(VrMath, HandPoseSmootherSteadiesAShakingWristAndKeepsTheHandsShape) {
        const HandPose rest = openHandPose();
        const glm::vec3 wrist = rest[1].position;
        const std::size_t fingertip = 15; // of the middle finger, the farthest from the wrist
        const float reach = glm::length(rest[fingertip].position - wrist);
        const float shift = 0.005f;
        const float swing = glm::radians(3.0f);
        HandPoseSmoother smoother;
        float moved = 0.0f;
        for (int i = 0; i < 270; ++i) {
            // The whole hand shakes about its wrist at 15 Hz, as when tracking
            // loses a finger.
            const float phase = std::sin(glm::two_pi<float>() * 15.0f * static_cast<float>(i) * FRAME);
            const glm::quat turn = glm::angleAxis(swing * phase, glm::vec3(0, 0, 1));
            HandPose shaking;
            for (std::size_t joint = 0; joint < shaking.size(); ++joint)
                shaking[joint] = {wrist + glm::vec3(shift * phase, 0.0f, 0.0f) + turn * (rest[joint].position - wrist), turn};
            const HandPose steady = smoother.update(shaking, i == 0 ? 0.0f : FRAME);
            if (i >= 90) {
                moved = std::max(moved, glm::length(steady[fingertip].position - rest[fingertip].position));
                EXPECT_NEAR(glm::length(steady[fingertip].position - steady[1].position), reach, 1e-4f);
            }
        }
        // Unsteadied, the fingertip swings by the shift plus the turn at arm's length.
        EXPECT_TRUE(moved < 0.5f * (shift + reach * swing));
    }

    TEST(VrMath, HandPoseSmootherFollowsMovingFingers) {
        const HandPose rest = openHandPose();
        const std::size_t fingertip = 10;         // of the index finger
        const glm::vec3 speed{0.0f, -0.3f, 0.0f}; // m/s, toward the palm side
        HandPoseSmoother smoother;
        HandPose hand = rest;
        HandPose steady = rest;
        for (int i = 0; i <= 45; ++i) {
            hand[fingertip].position = rest[fingertip].position + speed * (static_cast<float>(i) * FRAME);
            steady = smoother.update(hand, i == 0 ? 0.0f : FRAME);
        }
        EXPECT_TRUE(glm::length(steady[fingertip].position - hand[fingertip].position) < 0.01f);
        // The joints that did not move stay where they are.
        expectVecNear(steady[5].position, rest[5].position);
    }

    // A hand model as lfs::io::load_skinned_mesh() reads one: an unturned bone
    // at every joint of the open hand and one triangle, two corners on the
    // wrist and one on the tip of the index.
    [[nodiscard]] lfs::io::SkinnedMesh handSkinnedMesh(const bool hand_bones = true) {
        const HandJoints joints = openHand();
        const glm::vec3 wrist = joints[1], tip = joints[10];
        lfs::io::SkinnedMesh mesh;
        for (std::size_t joint = 1; joint < joints.size(); ++joint) {
            glm::mat4 inverse_bind(1.0f);
            inverse_bind[3] = glm::vec4(-joints[joint], 1.0f);
            mesh.bones.push_back({hand_bones ? std::string(hand_detail::kBoneNames[joint - 1]) : "bone" + std::to_string(joint),
                                  inverse_bind});
        }
        const glm::vec3 up{0.0f, 1.0f, 0.0f};
        mesh.vertices = {{wrist, up, {0, 0, 0, 0}, {1.0f, 0.0f, 0.0f, 0.0f}},
                         {wrist + glm::vec3(0.01f, 0.0f, 0.0f), up, {0, 0, 0, 0}, {1.0f, 0.0f, 0.0f, 0.0f}},
                         {tip, up, {9, 0, 0, 0}, {1.0f, 0.0f, 0.0f, 0.0f}}};
        mesh.indices = {0, 1, 2};
        return mesh;
    }

    TEST(VrMath, HandPoseFromJointsTurnsWithTheHand) {
        const HandJoints joints = openHand();
        const HandPose pose = handPoseFromJoints(joints);
        // Every bone points at the next joint with its -Z.
        expectVecNear(pose[7].orientation * glm::vec3(0, 0, -1), glm::normalize(joints[8] - joints[7]));
        expectVecNear(pose[3].orientation * glm::vec3(0, 0, -1), glm::normalize(joints[4] - joints[3]));
        expectVecNear(pose[1].orientation * glm::vec3(0, 0, -1), glm::normalize(joints[12] - joints[1]));

        // A hand moved as a whole turns every joint with it.
        const glm::quat turn = glm::angleAxis(0.9f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
        HandJoints moved;
        for (std::size_t joint = 0; joint < joints.size(); ++joint)
            moved[joint] = turn * joints[joint] + glm::vec3(0.3f, 1.2f, -0.4f);
        const HandPose moved_pose = handPoseFromJoints(moved);
        for (std::size_t joint = 0; joint < joints.size(); ++joint) {
            for (const glm::vec3 axis : {glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)}) {
                const glm::vec3 actual = moved_pose[joint].orientation * axis;
                const glm::vec3 expected = turn * (pose[joint].orientation * axis);
                for (int i = 0; i < 3; ++i)
                    EXPECT_NEAR(actual[i], expected[i], 1e-4f);
            }
        }
    }

    TEST(VrMath, HandModelGoesWhereTheJointsAre) {
        const auto model = handModelFromSkinnedMesh(handSkinnedMesh());
        ASSERT_TRUE(model.has_value());
        EXPECT_EQ(model->vertices.size(), std::size_t{3});
        EXPECT_EQ(model->indices.size(), std::size_t{3});
        // Bones are matched to the joints by name: the wrist is joint 1, the index tip joint 10.
        EXPECT_EQ(model->joints[0], std::size_t{1});
        EXPECT_EQ(model->joints[9], std::size_t{10});

        const HandJoints joints = openHand();
        const glm::vec3 wrist = joints[1], tip = joints[10];
        std::vector<HandVertex> mesh, scratch;
        // At rest the mesh is the model's own.
        appendSkinnedHand(mesh, *model, handPoseFromJoints(joints), scratch);
        ASSERT_TRUE(mesh.size() == 3);
        expectVecNear(mesh[0].position, wrist);
        expectVecNear(mesh[1].position, wrist + glm::vec3(0.01f, 0.0f, 0.0f));
        expectVecNear(mesh[2].position, tip);
        expectVecNear(mesh[0].normal, {0.0f, 1.0f, 0.0f});

        // A hand moved as a whole carries its mesh along, turned with it.
        const glm::quat turn = glm::angleAxis(0.9f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
        const glm::vec3 shift{0.3f, 1.2f, -0.4f};
        HandJoints moved;
        for (std::size_t joint = 0; joint < joints.size(); ++joint)
            moved[joint] = turn * joints[joint] + shift;
        mesh.clear();
        appendSkinnedHand(mesh, *model, handPoseFromJoints(moved), scratch);
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(mesh[1].position[i], (turn * (wrist + glm::vec3(0.01f, 0.0f, 0.0f)) + shift)[i], 1e-4f);
            EXPECT_NEAR(mesh[2].position[i], (turn * tip + shift)[i], 1e-4f);
            EXPECT_NEAR(mesh[0].normal[i], (turn * glm::vec3(0.0f, 1.0f, 0.0f))[i], 1e-4f);
        }

        // A finger that bends takes its part of the mesh and leaves the rest.
        HandJoints bent = joints;
        bent[10] += glm::vec3(0.0f, -0.03f, 0.02f);
        mesh.clear();
        appendSkinnedHand(mesh, *model, handPoseFromJoints(bent), scratch);
        expectVecNear(mesh[2].position, bent[10]);
        expectVecNear(mesh[1].position, wrist + glm::vec3(0.01f, 0.0f, 0.0f));
    }

    TEST(VrMath, HandModelRefusesMeshesThatAreNotHandModels) {
        // A skinned mesh whose bones are not the hand joints.
        EXPECT_FALSE(handModelFromSkinnedMesh(handSkinnedMesh(false)).has_value());
        // A hand that lacks a joint: the tip of the little finger.
        auto short_hand = handSkinnedMesh();
        short_hand.bones.pop_back();
        EXPECT_FALSE(handModelFromSkinnedMesh(short_hand).has_value());
        // A vertex moved by a bone the mesh does not have.
        auto stray = handSkinnedMesh();
        stray.vertices[2].bones[1] = 200;
        EXPECT_FALSE(handModelFromSkinnedMesh(stray).has_value());
        // Corners that make no triangle, or name no vertex.
        auto open_triangle = handSkinnedMesh();
        open_triangle.indices.pop_back();
        EXPECT_FALSE(handModelFromSkinnedMesh(open_triangle).has_value());
        auto outside = handSkinnedMesh();
        outside.indices[2] = 3;
        EXPECT_FALSE(handModelFromSkinnedMesh(outside).has_value());
        EXPECT_FALSE(handModelFromSkinnedMesh({}).has_value());
    }

    // The open hand with fingers folded toward the palm: by `knuckle` radians
    // where they leave it, then by `middle` and `last` more at the two joints
    // further out. `fingers` of them fold, from the index on.
    [[nodiscard]] HandJoints foldedHand(const float knuckle, const float middle, const float last,
                                        const std::size_t fingers = 4) {
        const HandJoints open = openHand();
        HandJoints folded = open;
        for (std::size_t finger = 0; finger < fingers; ++finger) {
            const std::size_t first = hand_detail::kFingers[finger] + 1;
            float turned = 0.0f;
            for (std::size_t joint = first; joint < first + 3; ++joint) {
                turned += joint == first ? knuckle : joint == first + 1 ? middle
                                                                        : last;
                // Fingers lie along -Z with the palm at -Y: they fold about X.
                folded[joint + 1] = folded[joint] + glm::angleAxis(-turned, glm::vec3(1, 0, 0)) * (open[joint + 1] - open[joint]);
            }
        }
        return folded;
    }

    [[nodiscard]] HandJoints fist() {
        return foldedHand(glm::radians(80.0f), glm::radians(100.0f), glm::radians(70.0f));
    }

    TEST(VrMath, FistClosureTellsAFistFromFingersBroughtTogether) {
        EXPECT_NEAR(fistClosure(openHand()), 0.0f, 1e-4f);
        // Every finger folded back onto the palm.
        EXPECT_TRUE(fistClosure(fist()) > 0.95f);
        // Fingers half bent do not make a fist.
        const float half = fistClosure(foldedHand(glm::radians(40.0f), glm::radians(60.0f), glm::radians(30.0f)));
        EXPECT_TRUE(half > 0.15f && half < 0.5f);
        // Nor does the index alone, bent to meet the thumb.
        EXPECT_TRUE(fistClosure(foldedHand(glm::radians(30.0f), glm::radians(50.0f), glm::radians(30.0f), 1)) < 0.1f);

        // The hand closes steadily into it.
        float before = 0.0f;
        for (int step = 1; step <= 10; ++step) {
            const float share = static_cast<float>(step) / 10.0f;
            const float closure = fistClosure(foldedHand(share * glm::radians(80.0f), share * glm::radians(100.0f),
                                                         share * glm::radians(70.0f)));
            EXPECT_TRUE(closure >= before);
            before = closure;
        }
        // Neither the size of the hand nor where it is matters.
        const HandJoints small = foldedHand(glm::radians(60.0f), glm::radians(80.0f), glm::radians(40.0f));
        HandJoints large = small;
        for (glm::vec3& joint : large)
            joint = joint * 1.3f + glm::vec3(0.2f, 1.1f, -0.3f);
        EXPECT_NEAR(fistClosure(large), fistClosure(small), 1e-4f);
        // Joints that all fell on one point are no fist.
        HandJoints collapsed;
        collapsed.fill(glm::vec3(0.1f, 1.0f, -0.2f));
        EXPECT_NEAR(fistClosure(collapsed), 0.0f, 1e-6f);
    }

    // The left hand that mirrors a right one.
    [[nodiscard]] HandJoints mirroredHand(HandJoints joints) {
        for (glm::vec3& joint : joints)
            joint.x = -joint.x;
        return joints;
    }

    TEST(VrMath, GrabHandFromJointsIsTurnedAsTheGripOfAController) {
        const HandJoints joints = openHand();
        const GrabHand hand = grabHandFromJoints(joints);
        expectVecNear(hand.position, joints[0]);
        // -Z from the little finger to the index along the knuckles, +X into
        // the palm of this right hand: out of its back.
        expectVecNear(hand.orientation * glm::vec3(0, 0, -1), glm::normalize(joints[7] - joints[22]));
        expectVecNear(hand.orientation * glm::vec3(1, 0, 0), {0.0f, 1.0f, 0.0f});
        // Closing the fist neither moves nor turns the hand that holds.
        const GrabHand closed = grabHandFromJoints(fist());
        expectVecNear(closed.position, hand.position);
        expectVecNear(closed.orientation * glm::vec3(1, 0, 0), hand.orientation * glm::vec3(1, 0, 0));
        expectVecNear(closed.orientation * glm::vec3(0, 1, 0), hand.orientation * glm::vec3(0, 1, 0));

        // The left hand held alike: +X comes out of its palm, and both hands
        // have their +Y the same way, mirrored.
        const GrabHand left = grabHandFromJoints(mirroredHand(joints));
        const glm::vec3 up = hand.orientation * glm::vec3(0, 1, 0);
        expectVecNear(left.orientation * glm::vec3(1, 0, 0), {0.0f, -1.0f, 0.0f});
        expectVecNear(left.orientation * glm::vec3(0, 1, 0), {-up.x, up.y, up.z});
        EXPECT_TRUE(glm::dot(left.orientation * glm::vec3(0, 1, 0), up) > 0.8f);
    }

    TEST(VrMath, TwoBareHandsRollTheSceneAsTwoGripsDo) {
        // Both hands held alike, 40 cm apart, the scene between them.
        const glm::vec3 midpoint{-0.2f, 1.3f, -0.4f};
        const glm::vec3 half{0.2f, 0.0f, 0.0f};
        const glm::vec3 held{0.3f, 0.8f, -1.0f};
        // A hand of this shape with its palm at `palm`, turned by `turn`.
        const auto handAt = [](const HandJoints& shape, const glm::vec3 palm, const glm::quat turn) {
            HandJoints joints;
            for (std::size_t joint = 0; joint < joints.size(); ++joint)
                joints[joint] = palm + turn * (shape[joint] - shape[0]);
            return grabHandFromJoints(joints);
        };
        const HandJoints right = openHand();
        const HandJoints left = mirroredHand(right);
        for (const float angle : {0.6f, -1.2f}) {
            auto rig = grabRig();
            const glm::vec3 world = worldOf(rig, held);
            const glm::quat roll = glm::angleAxis(angle, glm::vec3(1, 0, 0));
            GrabManipulation grab;
            grab.update(rig, handAt(left, midpoint - half, IDENTITY), handAt(right, midpoint + half, IDENTITY));
            // Twisting both wrists about the line between the hands rolls the
            // scene by as much.
            grab.update(rig, handAt(left, midpoint - half, roll), handAt(right, midpoint + half, roll));
            expectScenePointNear(worldOf(rig, midpoint + roll * (held - midpoint)), world);
        }
    }

    TEST(VrMath, EyeSeesWhatLiesInsideItsFieldOfView) {
        const EyeFov fov{glm::radians(-50.0f), glm::radians(40.0f), glm::radians(45.0f), glm::radians(-45.0f)};
        const glm::vec3 eye{0.1f, 1.6f, 0.2f};
        // A point at arm's length, at these angles in degrees to the right and up.
        const auto at = [&](const float across, const float upward) {
            return eye + glm::vec3(std::tan(glm::radians(across)), std::tan(glm::radians(upward)), -1.0f) * 0.4f;
        };
        EXPECT_TRUE(eyeSees(fov, IDENTITY, eye, at(0.0f, 0.0f)));
        EXPECT_TRUE(eyeSees(fov, IDENTITY, eye, at(-48.0f, 0.0f)));
        EXPECT_FALSE(eyeSees(fov, IDENTITY, eye, at(42.0f, 0.0f)));
        EXPECT_FALSE(eyeSees(fov, IDENTITY, eye, at(0.0f, 47.0f)));
        EXPECT_FALSE(eyeSees(fov, IDENTITY, eye, at(0.0f, -47.0f)));
        // Nothing behind the eye, whatever the angles.
        EXPECT_FALSE(eyeSees(fov, IDENTITY, eye, eye + glm::vec3(0.0f, 0.0f, 0.4f)));
        // A margin narrows the view, a negative one widens it.
        EXPECT_FALSE(eyeSees(fov, IDENTITY, eye, at(38.0f, 0.0f), glm::radians(4.0f)));
        EXPECT_TRUE(eyeSees(fov, IDENTITY, eye, at(42.0f, 0.0f), glm::radians(-4.0f)));
        // The view turns with the eye: a quarter turn to the left looks along -X.
        const glm::quat left = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0, 1, 0));
        EXPECT_TRUE(eyeSees(fov, left, eye, eye + glm::vec3(-0.4f, 0.0f, 0.0f)));
        EXPECT_FALSE(eyeSees(fov, left, eye, at(0.0f, 0.0f)));
    }
} // namespace
