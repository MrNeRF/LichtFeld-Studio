/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/status_bar_pacman.hpp"
#include "gui/status_bar_progress.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <string_view>

namespace lfs::vis::gui::pacman {
    namespace {
        size_t count(std::string_view text, std::string_view needle) {
            size_t result = 0;
            for (size_t pos = 0; (pos = text.find(needle, pos)) != std::string_view::npos; pos += needle.size())
                ++result;
            return result;
        }
    } // namespace

    TEST(StatusBarPacman, PowerPelletActivatesBothChasesAndExpires) {
        for (const float step : {0.20f, 0.65f}) {
            const auto before = buildFrame(step - 0.001f, 360, 200);
            const auto powered = buildFrame(step, 360, 200);
            EXPECT_FALSE(before.powered);
            EXPECT_TRUE(powered.powered);
            EXPECT_EQ(count(before.scenery_rml, "pacman-power-pellet"),
                      count(powered.scenery_rml, "pacman-power-pellet") + 1);
            EXPECT_NE(powered.sprite.find("power-open.png"), std::string::npos);
            EXPECT_EQ(count(powered.scenery_rml, "/frightened.png"), 1);
            EXPECT_FALSE(buildFrame(step + 0.231f, 360, 200).powered);
        }
    }

    TEST(StatusBarPacman, EatenGhostRetreatsAsEyesThenDisappears) {
        const auto before = buildFrame(0.349f, 360, 200);
        const auto eaten = buildFrame(0.351f, 360, 200);
        const auto after = buildFrame(0.389f, 360, 200);
        EXPECT_EQ(count(before.scenery_rml, "pacman-ghost"), 1);
        EXPECT_EQ(count(eaten.scenery_rml, "pacman-ghost"), 0);
        EXPECT_EQ(count(eaten.scenery_rml, "pacman-eyes"), 1);
        EXPECT_EQ(count(after.scenery_rml, "pacman-eyes"), 0);
    }

    TEST(StatusBarPacman, SceneryDoesNotInventCheckpoints) {
        for (float progress : {0.0f, 0.23f, 0.48f, 0.65f, 0.94f, 1.0f})
            EXPECT_EQ(count(buildFrame(progress, 360, 0).scenery_rml, "cherries"), 0);
    }

    TEST(StatusBarPacman, SaveStepsBecomeFruitAtTheirActualPositions) {
        const auto markers = buildProgressMarkersRml({690, 2040, 3000}, 3000, 660, {}, ProgressMarkerStyle::Pacman);
        EXPECT_EQ(count(markers, "../icon/pacman/cherries.png"), 3);
        EXPECT_NE(markers.find("left:23.000%"), std::string::npos);
        EXPECT_NE(markers.find("left:68.000%"), std::string::npos);
        EXPECT_NE(markers.find("left:99.500%"), std::string::npos);
        EXPECT_EQ(count(markers, "is-past"), 0);
        EXPECT_TRUE(buildProgressMarkersRml({}, 3000, 660, {}, ProgressMarkerStyle::Pacman).empty());
    }

    TEST(StatusBarPacman, CollectedFruitKeepsItsHoverTargetForEditing) {
        const auto markers = buildProgressMarkersRml({690, 2040}, 3000, 1020, {.hover_step = 690}, ProgressMarkerStyle::Pacman);
        EXPECT_EQ(count(markers, "is-past"), 1);
        EXPECT_NE(markers.find("progress-marker is-past is-hovered"), std::string::npos);
        EXPECT_EQ(count(markers, "cherries.png"), 2);
    }

    TEST(StatusBarPacman, DraggingMovesTheFruitAndAddingKeepsExistingFruit) {
        const ProgressMarkerRenderState moving{.dragging = true, .original_step = 2040, .preview_step = 2400};
        const auto moved = buildProgressMarkersRml({690, 2040}, 3000, 660, moving, ProgressMarkerStyle::Pacman);
        EXPECT_EQ(count(moved, "cherries.png"), 2);
        EXPECT_EQ(moved.find("left:68.000%"), std::string::npos);
        EXPECT_NE(moved.find("left:80.000%"), std::string::npos);
        EXPECT_NE(moved.find("is-hovered is-preview"), std::string::npos);
        auto adding = moving;
        adding.adding = true;
        const auto added = buildProgressMarkersRml({690, 2040}, 3000, 660, adding, ProgressMarkerStyle::Pacman);
        EXPECT_EQ(count(added, "cherries.png"), 3);
        EXPECT_NE(added.find("left:68.000%"), std::string::npos);
    }

    TEST(StatusBarPacman, OtherStylesKeepTheirCheckpointAppearance) {
        const auto classic = buildProgressMarkersRml({690, 2040}, 3000, 1020, {}, ProgressMarkerStyle::Classic);
        EXPECT_EQ(count(classic, "<div"), 2);
        const auto miner = buildProgressMarkersRml({690, 2040}, 3000, 1020, {}, ProgressMarkerStyle::Miner);
        EXPECT_EQ(count(miner, "gem-collected.png"), 1);
        EXPECT_EQ(count(miner, "gem.png"), 1);
        EXPECT_EQ(count(classic + miner, "cherries.png"), 0);
    }

    TEST(StatusBarPacman, EncountersStaySparseAtEveryWidth) {
        for (const float width : {140.0f, 220.0f, 360.0f}) {
            for (int step = 0; step <= 1000; ++step) {
                const auto frame = buildFrame(step / 1000.0f, width, 200);
                const auto ghosts = count(frame.scenery_rml, "pacman-ghost");
                const auto cherries = count(frame.scenery_rml, "pacman-cherries");
                EXPECT_LE(ghosts, 1);
                EXPECT_EQ(cherries, 0);
                EXPECT_LE(count(frame.scenery_rml, "pacman-power-pellet"), 1);
                EXPECT_LE(ghosts + cherries, 1);
            }
        }
        const auto complete = buildFrame(1.0f, 140, 0);
        EXPECT_FLOAT_EQ(complete.left_dp, 126.0f);
        EXPECT_TRUE(complete.scenery_rml.empty());
        EXPECT_FALSE(complete.powered);
    }

    TEST(StatusBarPacman, InvalidGeometryAndProgressCannotEscapeTheTrack) {
        for (const float width : {-10.0f, 0.0f, 14.0f, std::numeric_limits<float>::infinity()}) {
            const auto frame = buildFrame(1.0f, width, 0);
            EXPECT_FLOAT_EQ(frame.left_dp, 0.0f);
            EXPECT_TRUE(frame.scenery_rml.empty());
        }
        EXPECT_FLOAT_EQ(buildFrame(-1.0f, 360, 0).left_dp, 0.0f);
        EXPECT_FLOAT_EQ(buildFrame(2.0f, 360, 0).left_dp, 346.0f);
        EXPECT_FLOAT_EQ(buildFrame(std::numeric_limits<float>::quiet_NaN(), 360, 0).left_dp, 0.0f);
    }
} // namespace lfs::vis::gui::pacman
