/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "internal/viewport.hpp"
#include <gtest/gtest.h>
#include <utility>

namespace {
    struct NavigationCameraFixture {
        decltype(std::declval<Viewport>().camera) camera;
        NavigationCameraFixture(int, int) {}
    };
    void expectNavigationEquivalent(const NavigationCameraFixture& legacy, const NavigationCameraFixture& zup) {
        const auto basis = lfs::vis::navigationFrame(lfs::vis::NavigationUpAxis::DataZ);
        for (int row = 0; row < 3; ++row) {
            EXPECT_NEAR(zup.camera.t[row], (basis * legacy.camera.t)[row], 2e-4f);
            EXPECT_NEAR(zup.camera.pivot[row], (basis * legacy.camera.pivot)[row], 2e-4f);
            for (int col = 0; col < 3; ++col)
                EXPECT_NEAR(zup.camera.R[col][row], (basis * legacy.camera.R)[col][row], 2e-4f);
        }
        EXPECT_NEAR(glm::determinant(zup.camera.R), 1.0f, 2e-4f);
    }
    void initializeNavigationPair(NavigationCameraFixture& legacy, NavigationCameraFixture& zup) {
        legacy.camera.R = glm::mat3(1.0f);
        legacy.camera.t = glm::vec3(0, 0, 5);
        legacy.camera.pivot = glm::vec3(0);
        zup.camera.setNavigationUpAxis(lfs::vis::NavigationUpAxis::DataZ);
        const auto basis = lfs::vis::navigationFrame(lfs::vis::NavigationUpAxis::DataZ);
        zup.camera.R = basis * legacy.camera.R;
        zup.camera.t = basis * legacy.camera.t;
        zup.camera.pivot = basis * legacy.camera.pivot;
    }
} // namespace
TEST(ViewportNavigationUpTest, DataVerticalUsesTheExistingCoordinateBoundary) {
    const auto basis = lfs::vis::navigationFrame(lfs::vis::NavigationUpAxis::DataZ);
    EXPECT_EQ(basis[1], glm::vec3(0, 0, -1));
    EXPECT_EQ(basis[1], lfs::rendering::visualizerWorldPointFromDataWorld(glm::vec3(0, 0, 1)));
    EXPECT_FLOAT_EQ(glm::determinant(basis), 1.0f);
}
TEST(ViewportNavigationUpTest, SwitchingVerticalPreservesPosePivotAndHomeAndStopsMotion) {
    NavigationCameraFixture viewport(200, 200);
    const auto rotation = viewport.camera.R;
    const auto home = viewport.camera.home_R;
    viewport.camera.advanceWasd(0.1f, true, false, false, false, false, false);
    const auto moved = viewport.camera.t;
    const auto moved_pivot = viewport.camera.pivot;
    viewport.camera.setNavigationUpAxis(lfs::vis::NavigationUpAxis::DataZ);
    EXPECT_EQ(viewport.camera.R, rotation);
    EXPECT_EQ(viewport.camera.t, moved);
    EXPECT_EQ(viewport.camera.pivot, moved_pivot);
    EXPECT_EQ(viewport.camera.home_R, home);
    EXPECT_FALSE(viewport.camera.hasWasdMomentum());
    EXPECT_FALSE(viewport.camera.hasOrbitMomentum());
    EXPECT_FALSE(viewport.camera.hasDroneMotion());
}
TEST(ViewportNavigationUpTest, OrbitAndCoastAreEquivalentInTheSelectedFrame) {
    for (const bool trackball : {false, true}) {
        NavigationCameraFixture legacy(200, 200), zup(200, 200);
        initializeNavigationPair(legacy, zup);
        legacy.camera.startRotateAroundCenter({0, 0}, 0);
        zup.camera.startRotateAroundCenter({0, 0}, 0);
        for (int i = 1; i < 20; ++i) {
            const glm::vec2 pos(i * 7, i * 11);
            legacy.camera.orbitDrag(pos, trackball, i * 0.016f);
            zup.camera.orbitDrag(pos, trackball, i * 0.016f);
            expectNavigationEquivalent(legacy, zup);
        }
        legacy.camera.endRotateAroundCenter();
        zup.camera.endRotateAroundCenter();
        for (int i = 0; i < 100; ++i) {
            legacy.camera.updateOrbitCoast(0.016f);
            zup.camera.updateOrbitCoast(0.016f);
            expectNavigationEquivalent(legacy, zup);
        }
    }
}
TEST(ViewportNavigationUpTest, FpvAndLocalWasdAreEquivalentInTheSelectedFrame) {
    NavigationCameraFixture legacy(200, 200), zup(200, 200);
    initializeNavigationPair(legacy, zup);
    legacy.camera.initScreenPos({0, 0});
    zup.camera.initScreenPos({0, 0});
    for (int i = 1; i < 20; ++i) {
        const glm::vec2 pos(i * 7, i * 9);
        legacy.camera.rotateFpv(pos);
        zup.camera.rotateFpv(pos);
        legacy.camera.advanceWasd(0.016f, true, false, false, true, true, false);
        zup.camera.advanceWasd(0.016f, true, false, false, true, true, false);
        expectNavigationEquivalent(legacy, zup);
    }
}
TEST(ViewportNavigationUpTest, DroneFlightLookClimbAndBrakingAreEquivalent) {
    NavigationCameraFixture legacy(200, 200), zup(200, 200);
    initializeNavigationPair(legacy, zup);
    legacy.camera.enterDrone();
    zup.camera.enterDrone();
    legacy.camera.initScreenPos({0, 0});
    zup.camera.initScreenPos({0, 0});
    for (int i = 0; i < 300; ++i) {
        if (i < 40) {
            const glm::vec2 pos(i * 5, i * 7);
            legacy.camera.droneLook(pos);
            zup.camera.droneLook(pos);
        }
        legacy.camera.advanceDrone(0.016f, i < 80, false, false, i < 30, i < 20, false, 5);
        zup.camera.advanceDrone(0.016f, i < 80, false, false, i < 30, i < 20, false, 5);
        expectNavigationEquivalent(legacy, zup);
    }
    legacy.camera.finishDrone();
    zup.camera.finishDrone();
    expectNavigationEquivalent(legacy, zup);
}
TEST(ViewportNavigationUpTest, AxisViewsRemainUprightIncludingBothPoles) {
    NavigationCameraFixture viewport(200, 200);
    viewport.camera.setNavigationUpAxis(lfs::vis::NavigationUpAxis::DataZ);
    for (int axis = 0; axis < 3; ++axis) {
        for (bool negative : {false, true}) {
            viewport.camera.setAxisAlignedView(axis, negative);
            EXPECT_NEAR(glm::determinant(viewport.camera.R), 1.0f, 1e-5f);
            const auto forward = lfs::rendering::cameraForward(viewport.camera.R);
            EXPECT_NEAR(forward[axis], negative ? 1.0f : -1.0f, 1e-5f);
            if (axis != 2)
                EXPECT_GT(glm::dot(viewport.camera.R[1], viewport.camera.navigationUp()), 0.999f);
            viewport.camera.startRotateAroundCenter({0, 0}, 0);
            viewport.camera.updateRotateAroundCenter({0, 1}, 0.016f);
            EXPECT_NEAR(glm::determinant(viewport.camera.R), 1.0f, 1e-4f);
        }
    }
}

TEST(ViewportNavigationUpTest, PanZoomRollFocusAndSnappingPreserveTheSelectedFrame) {
    NavigationCameraFixture legacy(200, 200), zup(200, 200);
    initializeNavigationPair(legacy, zup);
    legacy.camera.initScreenPos({0, 0});
    zup.camera.initScreenPos({0, 0});
    for (int i = 1; i < 20; ++i) {
        legacy.camera.translate({i * 3, i * 2});
        zup.camera.translate({i * 3, i * 2});
        legacy.camera.zoom(0.2f, i % 2 == 0);
        zup.camera.zoom(0.2f, i % 2 == 0);
        legacy.camera.rotate_roll(0.5f);
        zup.camera.rotate_roll(0.5f);
        expectNavigationEquivalent(legacy, zup);
    }
    legacy.camera.focusOnBounds(glm::vec3(-1), glm::vec3(1));
    zup.camera.focusOnBounds(glm::vec3(-1), glm::vec3(1));
    expectNavigationEquivalent(legacy, zup);
    for (int axis = 0; axis < 3; ++axis) {
        for (bool negative : {false, true}) {
            zup.camera.setAxisAlignedView(axis, negative);
            const auto rotation = zup.camera.R;
            int snapped_axis = -1;
            bool snapped_negative = false;
            EXPECT_TRUE(zup.camera.snapToNearestAxisView(10.0f, &snapped_axis, &snapped_negative));
            EXPECT_EQ(snapped_axis, axis);
            EXPECT_EQ(snapped_negative, negative);
            EXPECT_EQ(zup.camera.R, rotation);
        }
    }
}
