/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/viewport.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <utility>

namespace {
    using Movement = lfs::vis::input::NavigationMovement;
    using Camera = decltype(std::declval<Viewport>().camera);

    Camera cameraAtOrigin() {
        Camera camera;
        camera.R = glm::mat3(1);
        camera.t = glm::vec3(0);
        camera.pivot = glm::vec3(0, 0, -5);
        return camera;
    }

    void expectVectorNear(glm::vec3 actual, glm::vec3 expected, float tolerance = 1e-5f) {
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(actual[i], expected[i], tolerance);
    }
} // namespace

TEST(AnalogNavigation, InvalidAndOversizedSamplesCannotProduceUnboundedMovement) {
    for (float invalid : {std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        for (int axis = 0; axis < 3; ++axis) {
            glm::vec3 axes(0.5f);
            axes[axis] = invalid;
            EXPECT_FALSE(Movement::fromAnalog(axes).active());
            EXPECT_FALSE(Movement::fromAnalog(axes).held());
        }
    }
    const auto input = Movement::fromAnalog(glm::vec3(std::numeric_limits<float>::max()));
    EXPECT_NEAR(glm::length(glm::vec3(input.right(), input.up(), input.forward())), 1.0f, 1e-6f);
}

TEST(AnalogNavigation, PerspectivePreservesPartialDeflectionAndPivotOffset) {
    auto full = cameraAtOrigin();
    auto half = full;
    const auto offset = full.pivot - full.t;
    for (int frame = 0; frame < 120; ++frame) {
        full.advanceWasd(1.0f / 60, Movement::fromAnalog({0, 0, 1}));
        half.advanceWasd(1.0f / 60, Movement::fromAnalog({0, 0, 0.5f}));
    }
    EXPECT_GT(glm::length(full.t), 0.0f);
    expectVectorNear(half.t, full.t * 0.5f);
    expectVectorNear(full.pivot - full.t, offset);
    expectVectorNear(half.pivot - half.t, offset);
    for (int frame = 0; frame < 300; ++frame) {
        full.advanceWasd(1.0f / 60, Movement{});
        half.advanceWasd(1.0f / 60, Movement{});
    }
    EXPECT_FALSE(full.hasWasdMomentum());
    EXPECT_FALSE(half.hasWasdMomentum());
}

TEST(AnalogNavigation, AnalogDiagonalsAreBoundedWhileKeyboardKeepsItsExistingSpeed) {
    auto straight = cameraAtOrigin();
    auto analog = straight;
    auto keyboard = straight;
    for (int frame = 0; frame < 60; ++frame) {
        straight.advanceWasd(1.0f / 60, Movement::fromAnalog({0, 0, 1}));
        analog.advanceWasd(1.0f / 60, Movement::fromAnalog({1, 0, 1}));
        keyboard.advanceWasd(1.0f / 60, true, false, false, true, false, false);
    }
    EXPECT_NEAR(glm::length(analog.t), glm::length(straight.t), 1e-5f);
    EXPECT_NEAR(glm::length(keyboard.t), glm::length(straight.t) * std::sqrt(2.0f), 1e-5f);
}

TEST(AnalogNavigation, OrthographicPanIsProportionalAndDollyPreservesPivot) {
    auto full = cameraAtOrigin();
    auto half = full;
    const auto pivot = full.pivot;
    for (int frame = 0; frame < 60; ++frame) {
        full.advanceOrthographicWasd(1.0f / 60, Movement::fromAnalog({1, 0, 0}), 100, 50, 5);
        half.advanceOrthographicWasd(1.0f / 60, Movement::fromAnalog({0.5f, 0, 0}), 100, 50, 5);
    }
    expectVectorNear(half.t, full.t * 0.5f);
    auto dolly_full = cameraAtOrigin();
    auto dolly_half = dolly_full;
    const float full_scale = dolly_full.advanceOrthographicWasd(
        0.1f, Movement::fromAnalog({0, 0, 1}), 100, 50, 5);
    const float half_scale = dolly_half.advanceOrthographicWasd(
        0.1f, Movement::fromAnalog({0, 0, 0.5f}), 100, 50, 5);
    EXPECT_GT(full_scale, half_scale);
    EXPECT_GT(half_scale, 100.0f);
    expectVectorNear(dolly_full.pivot, pivot);
    expectVectorNear(dolly_half.pivot, pivot);
}

TEST(AnalogNavigation, DroneTranslationIsProportionalAndStopsAfterRelease) {
    for (glm::vec3 axes : {glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)}) {
        auto full = cameraAtOrigin();
        auto half = full;
        full.enterDrone();
        half.enterDrone();
        for (int frame = 0; frame < 120; ++frame) {
            full.advanceDrone(1.0f / 60, Movement::fromAnalog(axes));
            half.advanceDrone(1.0f / 60, Movement::fromAnalog(axes * 0.5f));
        }
        EXPECT_GT(glm::length(full.t), 0.0f);
        expectVectorNear(half.t, full.t * 0.5f, 1e-4f);
        for (int frame = 0; frame < 600; ++frame) {
            full.advanceDrone(1.0f / 60, Movement{});
            half.advanceDrone(1.0f / 60, Movement{});
        }
        EXPECT_FALSE(full.hasDroneMotion());
        EXPECT_FALSE(half.hasDroneMotion());
    }
}

TEST(AnalogNavigation, WalkRemainsPlanarAtVerticalLookAndInBothNavigationFrames) {
    for (const auto axis : {lfs::vis::NavigationUpAxis::LegacyY, lfs::vis::NavigationUpAxis::DataZ}) {
        for (const float pitch : {0.0f, 0.7f, glm::half_pi<float>()}) {
            auto camera = cameraAtOrigin();
            camera.setNavigationUpAxis(axis);
            camera.R = lfs::vis::navigationFrame(axis) *
                       glm::mat3(glm::rotate(glm::mat4(1), pitch, glm::vec3(1, 0, 0)));
            const auto full = camera.walkVelocity(Movement::fromAnalog({0, 0, 1}), 2);
            const auto half = camera.walkVelocity(Movement::fromAnalog({0, 0, 0.5f}), 2);
            EXPECT_NEAR(glm::length(full), 2.0f, 1e-5f);
            EXPECT_NEAR(glm::dot(full, camera.navigationUp()), 0.0f, 1e-5f);
            expectVectorNear(half, full * 0.5f);
            EXPECT_NEAR(glm::length(camera.walkVelocity(
                            Movement::fromKeys(true, false, false, true, false, false), 2)),
                        2.0f, 1e-5f);
            expectVectorNear(camera.walkVelocity(Movement::fromAnalog({0, 1, 0}), 2), glm::vec3(0));
        }
    }
}

TEST(AnalogNavigation, OpposingKeysCancelWithoutLosingHeldState) {
    const auto opposed = Movement::fromKeys(true, true, true, true, true, true);
    EXPECT_FALSE(opposed.active());
    EXPECT_TRUE(opposed.held());
    auto camera = cameraAtOrigin();
    const auto initial = camera.t;
    camera.advanceWasd(0.1f, opposed);
    expectVectorNear(camera.t, initial);
    camera.enterDrone();
    camera.advanceDrone(0.1f, opposed);
    expectVectorNear(camera.t, initial);
}
