/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gui/status_bar_mining.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {

    int countNeedle(std::string_view haystack, std::string_view needle) {
        int count = 0;
        for (size_t pos = 0; (pos = haystack.find(needle, pos)) != std::string_view::npos;
             pos += needle.size()) {
            ++count;
        }
        return count;
    }

} // namespace

namespace lfs::vis::gui::mining {

    TEST(StatusBarMiningLayoutTest, RightAlignedSixteenDpBlocks) {
        const auto wide = miningLayout(360.0f);
        EXPECT_EQ(wide.block_count, 22);
        EXPECT_FLOAT_EQ(wide.offset_dp, 8.0f);

        const auto mid = miningLayout(220.0f);
        EXPECT_EQ(mid.block_count, 13);
        EXPECT_FLOAT_EQ(mid.offset_dp, 12.0f);

        const auto narrow = miningLayout(140.0f);
        EXPECT_EQ(narrow.block_count, 8);
        EXPECT_FLOAT_EQ(narrow.offset_dp, 12.0f);
    }

    TEST(StatusBarMiningBlockTypeTest, LastBlockIsDiamondOre) {
        for (const int count : {1, 8, 13, 22, 40}) {
            EXPECT_STREQ(miningBlockType(count - 1, count), "diamond-ore");
            for (int index = 0; index < count - 1; ++index)
                EXPECT_STRNE(miningBlockType(index, count), "diamond-ore");
        }
    }

    TEST(StatusBarMiningCrackStageTest, StepsAcrossABlockAndClamps) {
        constexpr int current_block = 5;
        constexpr float block_start = 80.0f;
        EXPECT_EQ(miningCrackStage(block_start, current_block), 0);
        EXPECT_EQ(miningCrackStage(block_start + 4.0f, current_block), 1);
        EXPECT_EQ(miningCrackStage(block_start + 8.0f, current_block), 2);
        EXPECT_EQ(miningCrackStage(block_start + 12.0f, current_block), 3);
        EXPECT_EQ(miningCrackStage(block_start + 15.9f, current_block), 3);
        EXPECT_EQ(miningCrackStage(block_start - 8.0f, current_block), 0);
        EXPECT_EQ(miningCrackStage(block_start + 32.0f, current_block), 3);
    }

    TEST(StatusBarMiningWallRmlTest, RemainingBlocksPlusOptionalCrack) {
        const MiningLayout layout = miningLayout(360.0f);
        constexpr int current_block = 5;
        const auto plain = buildMiningWallRml(layout, current_block, 0);
        EXPECT_EQ(countNeedle(plain, "class=\"wall-block\""), layout.block_count - current_block);
        EXPECT_EQ(countNeedle(plain, "crack-"), 0);
        EXPECT_EQ(countNeedle(plain, "../icon/mining/"), layout.block_count - current_block);
        EXPECT_EQ(countNeedle(plain, "../icon/block-"), 0);

        const auto cracked = buildMiningWallRml(layout, current_block, 2);
        EXPECT_EQ(countNeedle(cracked, "class=\"wall-block\""),
                  layout.block_count - current_block + 1);
        EXPECT_EQ(countNeedle(cracked, "../icon/mining/crack-2.png"), 1);
        EXPECT_EQ(countNeedle(cracked, "../icon/mining/"), layout.block_count - current_block + 1);
    }

    TEST(StatusBarMiningDebrisRmlTest, BreakFramesAndEmptySentinel) {
        const MiningLayout layout = miningLayout(220.0f);
        constexpr int block = 4;
        for (int frame = 0; frame <= 2; ++frame) {
            const auto rml = buildMiningDebrisRml(layout, block, frame);
            EXPECT_NE(rml.find("../icon/mining/break-" + std::to_string(frame + 1) + ".png"),
                      std::string::npos);
            EXPECT_EQ(countNeedle(rml, "../icon/mining/"), 1);
        }
        EXPECT_TRUE(buildMiningDebrisRml(layout, block, -1).empty());
    }

} // namespace lfs::vis::gui::mining
