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

TEST(AnalogNavigation, LookAndOrbitUseAngularRatesWithoutChangingPointerState) {
    for (bool orbit : {false, true}) {
        for (bool trackball : {false, true}) {
            auto reference = cameraAtOrigin();
            reference.t = {0, 0, 5};
            reference.pivot = glm::vec3(0);
            reference.initScreenPos({123, 456});
            for (int frame = 0; frame < 60; ++frame) {
                if (orbit)
                    reference.advanceOrbit(1.0f / 60, {0.5f, 0}, trackball);
                else
                    reference.advanceLook(1.0f / 60, {0.5f, 0});
            }
            for (int fps : {30, 144}) {
                auto camera = cameraAtOrigin();
                camera.t = {0, 0, 5};
                camera.pivot = glm::vec3(0);
                camera.initScreenPos({123, 456});
                for (int frame = 0; frame < fps; ++frame) {
                    if (orbit)
                        camera.advanceOrbit(1.0f / fps, {0.5f, 0}, trackball);
                    else
                        camera.advanceLook(1.0f / fps, {0.5f, 0});
                }
                expectVectorNear(camera.t, reference.t, 1e-4f);
                expectVectorNear(camera.pivot, reference.pivot, 1e-4f);
                for (int col = 0; col < 3; ++col)
                    expectVectorNear(camera.R[col], reference.R[col], 1e-4f);
                EXPECT_EQ(camera.prePos, glm::vec2(123, 456));
                EXPECT_FALSE(camera.isOrbiting);
                EXPECT_FALSE(camera.hasOrbitMomentum());
            }
        }
    }
}

TEST(AnalogNavigation, UprightLookPreservesPositionAndLimitsPitchInBothFrames) {
    for (auto axis : {lfs::vis::NavigationUpAxis::LegacyY, lfs::vis::NavigationUpAxis::DataZ}) {
        auto camera = cameraAtOrigin();
        camera.setNavigationUpAxis(axis);
        camera.R = lfs::vis::navigationFrame(axis);
        camera.updatePivotFromCamera(5);
        const auto position = camera.t;
        for (int frame = 0; frame < 120; ++frame)
            camera.advanceLook(1.0f / 60, {0, 3});
        expectVectorNear(camera.t, position);
        EXPECT_NEAR(glm::length(camera.pivot - camera.t), 5.0f, 1e-4f);
        const auto forward = lfs::rendering::cameraForward(camera.R);
        EXPECT_NEAR(glm::dot(forward, camera.navigationUp()), std::sin(glm::radians(89.0f)), 1e-4f);
        EXPECT_NEAR(glm::determinant(camera.R), 1.0f, 1e-4f);
    }
}

TEST(AnalogNavigation, PanUsesSceneCappedDistanceAndTranslatesCameraAndPivotTogether) {
    auto camera = cameraAtOrigin();
    camera.setSceneExtent(2);
    camera.initScreenPos({12, 34});
    const auto pivot = camera.pivot;
    camera.advancePan(0.5f, {0.5f, 0.25f});
    expectVectorNear(camera.t, {0.5f, 0.25f, 0});
    expectVectorNear(camera.pivot - pivot, camera.t);
    EXPECT_EQ(camera.prePos, glm::vec2(12, 34));
    EXPECT_FALSE(camera.hasPanMomentum());
}

TEST(AnalogNavigation, RelativeDollyIsFrameRateIndependentAndKeepsPivot) {
    for (int fps : {30, 60, 144}) {
        auto camera = cameraAtOrigin();
        camera.t = {0, 0, 5};
        camera.pivot = glm::vec3(0);
        camera.initScreenPos({12, 34});
        float scale = 100;
        for (int frame = 0; frame < fps; ++frame) {
            camera.advanceDolly(1.0f / fps, 0.5f);
            scale = camera.advanceOrthographicDolly(1.0f / fps, 0.5f, scale);
        }
        EXPECT_NEAR(glm::length(camera.t), 5.0f * std::exp(-0.5f), 1e-4f);
        EXPECT_NEAR(scale, 100.0f * std::exp(0.5f), 0.002f);
        expectVectorNear(camera.pivot, glm::vec3(0));
        EXPECT_EQ(camera.prePos, glm::vec2(12, 34));
        camera.advanceDolly(1, std::numeric_limits<float>::max());
        EXPECT_NEAR(glm::length(camera.t), 0.1f, 1e-6f);
        camera.advanceDolly(1, -std::numeric_limits<float>::max());
        EXPECT_NEAR(glm::length(camera.t), lfs::rendering::DEFAULT_FAR_PLANE, 0.01f);
        EXPECT_FLOAT_EQ(camera.advanceOrthographicDolly(1, std::numeric_limits<float>::max(), 100), 10000);
        EXPECT_FLOAT_EQ(camera.advanceOrthographicDolly(1, -std::numeric_limits<float>::max(), 100), 1);
    }
}

TEST(AnalogNavigation, DroneLookUpdatesTheTargetAndSettlesThroughTheExistingIntegrator) {
    auto camera = cameraAtOrigin();
    camera.enterDrone();
    camera.initScreenPos({12, 34});
    const auto rotation = camera.R;
    const auto position = camera.t;
    camera.advanceDroneLook(0.1f, {1, 0.5f});
    EXPECT_EQ(camera.R, rotation);
    EXPECT_TRUE(camera.hasDroneMotion());
    EXPECT_EQ(camera.prePos, glm::vec2(12, 34));
    for (int frame = 0; frame < 600; ++frame)
        camera.advanceDrone(1.0f / 60, Movement{});
    EXPECT_FALSE(camera.hasDroneMotion());
    EXPECT_NE(camera.R, rotation);
    expectVectorNear(camera.t, position);
    EXPECT_NEAR(glm::determinant(camera.R), 1.0f, 1e-4f);
}

TEST(AnalogNavigation, DollyHandlesOffAxisPivotsAndDoesNotSnapExistingOutOfRangeViews) {
    auto camera = cameraAtOrigin();
    camera.t = {3, 0, 4};
    camera.pivot = glm::vec3(0);
    camera.advanceDolly(0.5f, 1);
    expectVectorNear(camera.t, glm::vec3(3, 0, 4) * std::exp(-0.5f));
    camera.t = {0, 0, lfs::rendering::DEFAULT_FAR_PLANE * 2};
    const auto distant = camera.t;
    camera.advanceDolly(1, -1);
    expectVectorNear(camera.t, distant);
    camera.advanceDolly(1, 0.1f);
    EXPECT_LT(glm::length(camera.t), glm::length(distant));
    EXPECT_FLOAT_EQ(camera.advanceOrthographicDolly(1, -1, 0.5f), 0.5f);
    EXPECT_FLOAT_EQ(camera.advanceOrthographicDolly(1, 1, 20000), 20000);
}

TEST(AnalogNavigation, InvalidContinuousRatesAndTimeStepsLeaveTheCameraUntouched) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (float dt : {-1.0f, 0.0f, nan, inf, 0.1f}) {
        for (float rate : {0.0f, nan, inf}) {
            auto camera = cameraAtOrigin();
            camera.initScreenPos({12, 34});
            const auto before = camera;
            camera.advanceLook(dt, {rate, 0});
            camera.advanceOrbit(dt, {rate, 0});
            camera.advancePan(dt, {rate, 0});
            camera.advanceDolly(dt, rate);
            camera.advanceDroneLook(dt, {rate, 0});
            EXPECT_EQ(camera.advanceOrthographicDolly(dt, rate, 100), 100);
            EXPECT_EQ(camera.t, before.t);
            EXPECT_EQ(camera.pivot, before.pivot);
            EXPECT_EQ(camera.R, before.R);
            EXPECT_EQ(camera.prePos, before.prePos);
            EXPECT_FALSE(camera.hasDroneMotion());
        }
    }
}
