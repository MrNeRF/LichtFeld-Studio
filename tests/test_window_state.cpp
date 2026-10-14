/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/persisted_settings.hpp"
#include "licht_test_support.hpp"
#include "window/window_state_utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

    using lfs::vis::WindowRectangle;

    TEST(WindowStateContract, ExternalPanelsRequireOptInAndHonorBothSafeModePaths) {
        EXPECT_FALSE(lfs::vis::externalFloatingWindowsEnabled(true, false, false, false));
        EXPECT_TRUE(lfs::vis::externalFloatingWindowsEnabled(true, false, false, true));
        EXPECT_FALSE(lfs::vis::externalFloatingWindowsEnabled(true, true, false, true));
        EXPECT_FALSE(lfs::vis::externalFloatingWindowsEnabled(true, false, true, true));
        EXPECT_FALSE(lfs::vis::externalFloatingWindowsEnabled(false, false, false, true));
    }

    TEST(WindowStateContract, FutureWindowFieldsSurviveGeometrySave) {
        const auto root = std::filesystem::temp_directory_path() / "lfs_window_future_settings";
        std::filesystem::create_directories(root);
        const auto path = root / "window.json";
        const nlohmann::json original = {
            {"schema_version", 999},
            {"version", 1001},
            {"x", 4},
            {"y", 8},
            {"width", 1000},
            {"height", 600},
            {"maximized", false},
            {"monitor_uuid", "future-display"},
            {"workspace_layout", {{"new_monitor_behavior", 1}, {"panels", nlohmann::json::array({"scene", "future"})}}},
        };
        std::ofstream(path) << original.dump(2) << '\n';
        const auto merged = lfs::core::persisted_settings::mergeForWrite(
            path,
            {{"x", 40}, {"y", 80}, {"width", 1200}, {"height", 700}, {"maximized", true}},
            "schema_version", 1);
        std::ofstream(path, std::ios::trunc) << merged.dump(2) << '\n';
        const auto saved = lfs::core::persisted_settings::readObject(path);
        auto expected = original;
        expected.update({{"x", 40}, {"y", 80}, {"width", 1200}, {"height", 700}, {"maximized", true}});
        EXPECT_EQ(saved, expected);
        std::filesystem::remove_all(root);
    }

    TEST(WindowStateContract, DamagedWindowIsBackedUpBeforeReplacement) {
        const auto root = std::filesystem::temp_directory_path() / "lfs_window_damaged_settings";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        const auto path = root / "window.json";
        const std::string original = R"({"x":440,"y":260,"width":)";
        std::ofstream(path) << original;
        const auto merged = lfs::core::persisted_settings::mergeForWrite(
            path, {{"x", 320}, {"y", 180}, {"width", 1280}}, "schema_version", 1,
            nullptr, {}, root / "backups");
        EXPECT_EQ(merged.at("width"), 1280);
        std::ifstream backup(lfs::test::licht::find_backup(root / "backups", path));
        ASSERT_TRUE(backup);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(backup), {}), original);
        backup.close();
        std::filesystem::remove_all(root);
    }

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

    TEST(WindowStateContract, DetachedSmallPanelRecoversAfterSecondaryMonitorDisappears) {
        const WindowRectangle primary{0, 0, 1920, 1080};
        const WindowRectangle panel{-1200, 100, 480, 300};
        const auto recovered = lfs::vis::recoverWindowRectangle(panel, {primary}, primary, 96, 64, 320, 180);
        EXPECT_EQ(recovered.width, 480);
        EXPECT_EQ(recovered.height, 300);
        EXPECT_GE(recovered.x, primary.x);
        EXPECT_GE(recovered.y, primary.y);
        EXPECT_LE(recovered.x + recovered.width, primary.width);
        EXPECT_LE(recovered.y + recovered.height, primary.height);
    }

    TEST(WindowStateContract, DetachedPanelRemainsOnConnectedMonitorWithNegativeCoordinates) {
        const WindowRectangle primary{0, 0, 1920, 1080};
        const WindowRectangle secondary{-1920, -200, 1920, 1080};
        const WindowRectangle panel{-1400, 0, 640, 480};
        EXPECT_EQ(lfs::vis::recoverWindowRectangle(panel, {primary, secondary}, primary, 96, 64, 320, 180), panel);
    }

    TEST(WindowStateContract, DetachedLargePanelKeepsItsSizeAfterMonitorRemoval) {
        const WindowRectangle remaining{100, 32, 1920, 1048};
        const WindowRectangle panel{-1800, 100, 1200, 700};
        EXPECT_EQ(lfs::vis::recoverWindowRectangle(panel, {remaining}, remaining, 96, 64, 320, 180),
                  (WindowRectangle{460, 206, 1200, 700}));
    }

    TEST(WindowStateContract, DetachedPanelFitsSmallerRemainingMonitorIncludingTitleBar) {
        const WindowRectangle remaining{0, 40, 800, 560};
        const WindowRectangle panel{3000, -700, 1600, 900};
        EXPECT_EQ(lfs::vis::recoverWindowRectangle(panel, {remaining}, remaining, 96, 64, 320, 180), remaining);
    }

    TEST(WindowStateContract, DetachedPanelWithOnlyLowerEdgeVisibleRecoversItsTitleBar) {
        const WindowRectangle remaining{0, 40, 1280, 680};
        const WindowRectangle panel{200, -400, 640, 600};
        const auto recovered = lfs::vis::recoverWindowRectangle(panel, {remaining}, remaining, 96, 64, 320, 180);
        EXPECT_EQ(recovered, (WindowRectangle{200, 40, 640, 600}));
    }

    TEST(WindowStateContract, DetachedPanelStraddlingOffsetMonitorsCannotLoseItsTitleBar) {
        const std::vector<WindowRectangle> displays{{0, 100, 1280, 800}, {1280, 200, 1280, 800}};
        const WindowRectangle panel{1000, -100, 600, 500};
        EXPECT_EQ(lfs::vis::recoverFloatingWindowRectangle(panel, displays, displays.front()),
                  (WindowRectangle{680, 100, 600, 500}));
    }

    TEST(WindowStateContract, FloatingPanelWithoutSavedPositionOpensInsideApplication) {
        const WindowRectangle app{500, 300, 1000, 700};
        const WindowRectangle display{0, 0, 1920, 1080};
        EXPECT_EQ(lfs::vis::defaultFloatingWindowRectangle({0, 0, 780, 440}, app, display),
                  (WindowRectangle{610, 430, 780, 440}));
    }

    TEST(WindowStateContract, FloatingPanelInitialSizeFitsSmallerApplicationWindow) {
        const WindowRectangle app{200, 100, 640, 480};
        const WindowRectangle display{0, 0, 1920, 1080};
        EXPECT_EQ(lfs::vis::defaultFloatingWindowRectangle({0, 0, 1200, 900}, app, display), app);
    }

    TEST(WindowStateContract, DetachedPanelStraddlingMonitorsWithAccessibleTitleIsPreserved) {
        const std::vector<WindowRectangle> displays{{0, 100, 1280, 800}, {1280, 200, 1280, 800}};
        const WindowRectangle panel{1000, 220, 600, 500};
        EXPECT_EQ(lfs::vis::recoverFloatingWindowRectangle(panel, displays, displays.front()), panel);
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
