/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "window/window_state_utils.hpp"

#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

    using lfs::vis::WindowRectangle;

    TEST(WindowStateContract, AcceptsWindowVisibleOnAnyConnectedDisplay) {
        const std::vector<WindowRectangle> displays{{0, 0, 1920, 1080}, {1920, -200, 2560, 1440}};
        EXPECT_TRUE(lfs::vis::windowRectangleVisible({2200, 100, 1280, 720}, displays));
    }

    TEST(WindowStateContract, RejectsWindowOutsideEveryConnectedDisplay) {
        const std::vector<WindowRectangle> displays{{0, 0, 1920, 1080}, {1920, 0, 1920, 1080}};
        EXPECT_FALSE(lfs::vis::windowRectangleVisible({5000, 100, 1280, 720}, displays));
    }

    TEST(WindowStateContract, IntersectionArithmeticDoesNotOverflow) {
        const std::vector<WindowRectangle> displays{{0, 0, 1920, 1080}};
        EXPECT_FALSE(lfs::vis::windowRectangleVisible(
            {std::numeric_limits<int>::max() - 10, 0, 1280, 720}, displays));
    }

    TEST(WindowStateContract, RecoveryClampsAndCentersOversizedWindow) {
        const auto recovered = lfs::vis::centerWindowOnDisplay(
            {-4000, -3000, 8000, 6000}, {1920, -200, 1600, 900});
        EXPECT_EQ(recovered.x, 1920);
        EXPECT_EQ(recovered.y, -200);
        EXPECT_EQ(recovered.width, 1600);
        EXPECT_EQ(recovered.height, 900);
    }

    TEST(WindowStateContract, FirstLaunchDefaultsCenterOnPrimaryUsableBounds) {
        const auto centered = lfs::vis::centerWindowOnDisplay(
            {0, 0, 1280, 720}, {0, 0, 1920, 1040});
        EXPECT_EQ(centered, (WindowRectangle{320, 160, 1280, 720}));
    }

    TEST(WindowStateContract, RecoverySupportsSmallDisplays) {
        const auto recovered = lfs::vis::centerWindowOnDisplay(
            {0, 0, 1, 1}, {100, 200, 320, 240});
        EXPECT_EQ(recovered, (WindowRectangle{100, 200, 320, 240}));
    }

    TEST(WindowStateContract, OversizedVisibleWindowIsClampedToSmallerDisplay) {
        const std::vector<WindowRectangle> displays{{0, 0, 1280, 680}};
        const auto recovered = lfs::vis::recoverWindowRectangle(
            {20, 20, 2400, 1400}, displays, displays.front());
        EXPECT_EQ(recovered, (WindowRectangle{0, 0, 1280, 680}));
    }

    TEST(WindowStateContract, OversizedWindowUsesDisplayWithLargestIntersection) {
        const std::vector<WindowRectangle> displays{{0, 0, 1280, 720}, {1280, 0, 1920, 1040}};
        const auto recovered = lfs::vis::recoverWindowRectangle(
            {1400, 20, 2560, 1440}, displays, displays.front());
        EXPECT_EQ(recovered, (WindowRectangle{1280, 0, 1920, 1040}));
    }

    TEST(WindowStateContract, VisibleWindowThatFitsItsDisplayIsPreserved) {
        const std::vector<WindowRectangle> displays{{0, 0, 1280, 720}, {1280, 0, 1920, 1080}};
        const WindowRectangle saved{1500, 100, 1200, 700};
        EXPECT_EQ(lfs::vis::recoverWindowRectangle(saved, displays, displays.front()), saved);
    }

    TEST(WindowStateContract, SavedGeometryUnderTopPanelIsPushedIntoWorkArea) {
        const std::vector<WindowRectangle> displays{{67, 32, 1853, 1168}};
        const auto recovered = lfs::vis::recoverWindowRectangle(
            {0, 0, 1280, 720}, displays, displays.front());
        EXPECT_EQ(recovered, (WindowRectangle{67, 32, 1280, 720}));
    }

    TEST(WindowStateContract, FullDisplaySaveIsClampedToWorkArea) {
        const std::vector<WindowRectangle> displays{{67, 32, 1853, 1168}};
        const auto recovered = lfs::vis::recoverWindowRectangle(
            {0, 0, 1920, 1200}, displays, displays.front());
        EXPECT_EQ(recovered, (WindowRectangle{67, 32, 1853, 1168}));
    }

    TEST(WindowStateContract, WindowAboveWorkAreaTopIsPulledDown) {
        const std::vector<WindowRectangle> displays{{0, 32, 1920, 1168}};
        const auto recovered = lfs::vis::recoverWindowRectangle(
            {600, -40, 1280, 720}, displays, displays.front());
        EXPECT_EQ(recovered, (WindowRectangle{600, 32, 1280, 720}));
    }

    TEST(WindowStateContract, StraddlingTwoDisplaysIsPreserved) {
        const std::vector<WindowRectangle> displays{{0, 32, 1920, 1168}, {1920, 32, 1920, 1168}};
        const WindowRectangle saved{1400, 100, 1200, 700};
        EXPECT_EQ(lfs::vis::recoverWindowRectangle(saved, displays, displays.front()), saved);
    }

    TEST(WindowStateContract, SafeModeDisablesAutomaticPersistence) {
        const char* previous_raw = std::getenv("LFS_SAFE_MODE");
        const std::optional<std::string> previous = previous_raw
                                                        ? std::optional<std::string>(previous_raw)
                                                        : std::nullopt;
#ifdef _WIN32
        (void)_putenv_s("LFS_SAFE_MODE", "1");
#else
        (void)setenv("LFS_SAFE_MODE", "1", 1);
#endif
        EXPECT_FALSE(lfs::vis::automaticWindowStatePersistenceEnabled());
#ifdef _WIN32
        (void)_putenv_s("LFS_SAFE_MODE", previous ? previous->c_str() : "");
#else
        if (previous)
            (void)setenv("LFS_SAFE_MODE", previous->c_str(), 1);
        else
            (void)unsetenv("LFS_SAFE_MODE");
#endif
    }

} // namespace
