/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/startup_overlay_geometry.hpp"

#include <gtest/gtest.h>

namespace lfs::vis::gui {

    TEST(StartupOverlayGeometry, BlocksCardAndOpenDropdownOnly) {
        const StartupOverlayRect card{200.0f, 80.0f, 440.0f, 280.0f};
        const StartupOverlayRect dropdown{270.0f, 230.0f, 410.0f, 350.0f};

        EXPECT_TRUE(startupOverlayBlocksPointer(card, std::nullopt, 200.0f, 80.0f));
        EXPECT_TRUE(startupOverlayBlocksPointer(card, dropdown, 320.0f, 300.0f));
        EXPECT_FALSE(startupOverlayBlocksPointer(card, dropdown, 100.0f, 180.0f));
        EXPECT_FALSE(startupOverlayBlocksPointer(card, dropdown, 440.0f, 280.0f));
    }

    TEST(StartupOverlayGeometry, ReducesDpRatioForCenteredAndAsymmetricExtents) {
        EXPECT_FLOAT_EQ(startupOverlayFitDpRatio(
                            2.0f, 2.0f, 640.0f, 360.0f,
                            StartupOverlayRect{-80.0f, -120.0f, 720.0f, 480.0f}),
                        1.2f);
        EXPECT_FLOAT_EQ(startupOverlayFitDpRatio(
                            2.0f, 2.0f, 640.0f, 360.0f,
                            StartupOverlayRect{100.0f, 30.0f, 540.0f, 430.0f}),
                        1.44f);
    }

    TEST(StartupOverlayGeometry, NeverExceedsGlobalDpRatio) {
        EXPECT_FLOAT_EQ(startupOverlayFitDpRatio(
                            1.5f, 1.5f, 1280.0f, 720.0f,
                            StartupOverlayRect{440.0f, 260.0f, 840.0f, 460.0f}),
                        1.5f);
    }

} // namespace lfs::vis::gui
