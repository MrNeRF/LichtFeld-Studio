/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gui/status_bar_mining.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

namespace {

    int countNeedle(std::string_view haystack, std::string_view needle) {
        int count = 0;
        for (size_t pos = 0; (pos = haystack.find(needle, pos)) != std::string_view::npos;
             pos += needle.size()) {
            ++count;
        }
        return count;
    }

    int alphaInRml(const std::string& rml) {
        const size_t hash = rml.find('#');
        return std::stoi(rml.substr(hash + 7, 2), nullptr, 16);
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

    TEST(StatusBarMiningLayoutTest, ProgressTextMovesAheadAndClampsAtTheRightEdge) {
        EXPECT_FLOAT_EQ(miningProgressTextLeftDp(0.0f, 360.0f, 28.0f), 48.0f);
        EXPECT_FLOAT_EQ(miningProgressTextLeftDp(340.0f, 360.0f, 28.0f), 330.0f);
        EXPECT_FLOAT_EQ(miningProgressTextLeftDp(10.0f, 20.0f, 28.0f), 0.0f);
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

    TEST(StatusBarMiningParticlesTest, BreakAndStrikeSpawnCounts) {
        const MiningLayout layout = miningLayout(220.0f);
        std::vector<MiningParticle> particles;
        spawnBreakParticles(particles, layout, 4, "stone");
        EXPECT_EQ(particles.size(), 16);

        spawnStrikeChips(particles, 80.0f, 6.0f, "stone", 23);
        EXPECT_EQ(particles.size(), 20);
        for (size_t i = 16; i < particles.size(); ++i)
            EXPECT_EQ(particles[i].size, 1);
    }

    TEST(StatusBarMiningParticlesTest, SpawningIsDeterministicForTheSameSeed) {
        const MiningLayout layout = miningLayout(220.0f);
        std::vector<MiningParticle> first;
        std::vector<MiningParticle> second;
        spawnBreakParticles(first, layout, 4, "iron");
        spawnBreakParticles(second, layout, 4, "iron");
        ASSERT_EQ(first.size(), second.size());
        for (size_t i = 0; i < first.size(); ++i) {
            EXPECT_FLOAT_EQ(first[i].x, second[i].x);
            EXPECT_FLOAT_EQ(first[i].y, second[i].y);
            EXPECT_FLOAT_EQ(first[i].vx, second[i].vx);
            EXPECT_FLOAT_EQ(first[i].vy, second[i].vy);
            EXPECT_EQ(first[i].size, second[i].size);
            EXPECT_EQ(first[i].rgb, second[i].rgb);
            EXPECT_FLOAT_EQ(first[i].life, second[i].life);
        }
    }

    TEST(StatusBarMiningParticlesTest, GravityMovesParticlesAndLandingStopsAtFloor) {
        std::vector<MiningParticle> particles{{.x = 0.0f,
                                               .y = 0.0f,
                                               .vx = 0.0f,
                                               .vy = 0.0f,
                                               .size = 2,
                                               .rgb = 0x123456,
                                               .life = 1.0f}};
        stepMiningParticles(particles, 0.1f);
        ASSERT_EQ(particles.size(), 1);
        EXPECT_GT(particles[0].y, 0.0f);
        particles[0].y = 14.0f;
        particles[0].vy = 10.0f;
        stepMiningParticles(particles, 0.1f);
        ASSERT_EQ(particles.size(), 1);
        EXPECT_TRUE(particles[0].landed);
        EXPECT_FLOAT_EQ(particles[0].y, 13.0f);
    }

    TEST(StatusBarMiningParticlesTest, ParticlesExpireAfterTheirLife) {
        std::vector<MiningParticle> particles{{.x = 0.0f,
                                               .y = 0.0f,
                                               .vx = 0.0f,
                                               .vy = 0.0f,
                                               .size = 1,
                                               .rgb = 0xffffff,
                                               .life = 0.2f}};
        stepMiningParticles(particles, 0.2f);
        EXPECT_TRUE(particles.empty());
    }

    TEST(StatusBarMiningParticlesTest, ParticleRmlContainsOneDivPerLiveParticle) {
        const std::vector<MiningParticle> particles{{.x = 1.234f,
                                                     .y = 5.678f,
                                                     .size = 2,
                                                     .rgb = 0x123abc,
                                                     .life = 1.0f},
                                                    {.x = 9.0f,
                                                     .y = 3.0f,
                                                     .size = 1,
                                                     .rgb = 0xff0000,
                                                     .life = 1.0f}};
        const auto rml = buildMiningParticlesRml(particles);
        EXPECT_EQ(countNeedle(rml, "class=\"mining-particle\""), 2);
        EXPECT_NE(rml.find("left:1.23dp;top:5.68dp;width:2dp;height:2dp;background-color:#123abcff"),
                  std::string::npos);
        EXPECT_TRUE(buildMiningParticlesRml({}).empty());
    }

    TEST(StatusBarMiningSmokeTest, SpriteIndexFollowsPausePhases) {
        EXPECT_EQ(miningSmokeSpriteIndex(0), 0);
        EXPECT_EQ(miningSmokeSpriteIndex(499), 0);
        EXPECT_EQ(miningSmokeSpriteIndex(500), 1);
        EXPECT_EQ(miningSmokeSpriteIndex(749), 1);
        EXPECT_EQ(miningSmokeSpriteIndex(750), 0);
        EXPECT_EQ(miningSmokeSpriteIndex(1000), 1);
        EXPECT_EQ(miningSmokeSpriteIndex(1249), 1);
        EXPECT_EQ(miningSmokeSpriteIndex(1250), 0);
        EXPECT_EQ(miningSmokeSpriteIndex(5000), 0);
        EXPECT_EQ(miningSmokeSpriteIndex(5279), 0);
        EXPECT_EQ(miningSmokeSpriteIndex(5280), 1);
        EXPECT_EQ(miningSmokeSpriteIndex(6680), 0);
    }

    TEST(StatusBarMiningSmokeTest, LettersSpawnOneReversedGlyphPerExhale) {
        std::vector<MiningParticle> particles;
        spawnSmokeLetters(particles, 10.0f, 1.0f, -1, 499);
        EXPECT_TRUE(particles.empty());

        constexpr size_t expected_sizes[] = {14, 11, 15, 15, 13, 14, 11, 8, 13};
        int previous_pause_ms = -1;
        for (size_t index = 0; index < std::size(expected_sizes); ++index) {
            const int pause_ms = 500 + static_cast<int>(index) * 500;
            const size_t before = particles.size();
            spawnSmokeLetters(particles, 10.0f, 1.0f, previous_pause_ms, pause_ms);
            EXPECT_EQ(particles.size() - before, expected_sizes[index]);
            previous_pause_ms = pause_ms;
        }

        ASSERT_EQ(particles.size(), 114);
        int shadow_count = 0;
        int lit_count = 0;
        for (const auto& particle : particles) {
            EXPECT_FALSE(particle.gravity);
            EXPECT_TRUE(particle.fade);
            EXPECT_EQ(particle.size, 1);
            EXPECT_FLOAT_EQ(particle.vx, 8.0f);
            EXPECT_FLOAT_EQ(particle.vy, -0.25f);
            EXPECT_FLOAT_EQ(particle.life, 6.0f);
            if (particle.rgb == 0x222228)
                ++shadow_count;
            if (particle.rgb == 0xdedee6)
                ++lit_count;
        }
        EXPECT_EQ(shadow_count, 6 + 5 + 6 + 7 + 5 + 6 + 5 + 4 + 6);
        EXPECT_EQ(lit_count, 8 + 6 + 9 + 8 + 8 + 8 + 6 + 4 + 7);

        std::vector<MiningParticle> full_pause;
        spawnSmokeLetters(full_pause, 10.0f, 1.0f, -1, 5000);
        EXPECT_EQ(full_pause.size(), 114);
    }

    TEST(StatusBarMiningSmokeTest, SmokeParticlesFadeAndDoNotLand) {
        MiningParticle particle{
            .x = 0.0f,
            .y = 1.0f,
            .vx = 8.0f,
            .vy = -0.25f,
            .size = 1,
            .rgb = 0xdedee6,
            .life = 6.0f,
            .gravity = false,
            .fade = true,
            .alpha = 240,
        };
        std::vector<MiningParticle> particles{particle};
        const int alpha_at_start = alphaInRml(buildMiningParticlesRml(particles));
        particle.age = 2.0f;
        particles[0] = particle;
        const int alpha_in_middle = alphaInRml(buildMiningParticlesRml(particles));
        particle.age = 5.0f;
        particles[0] = particle;
        const int alpha_near_end = alphaInRml(buildMiningParticlesRml(particles));
        EXPECT_GT(alpha_at_start, alpha_in_middle);
        EXPECT_GT(alpha_in_middle, alpha_near_end);

        particle.age = 0.0f;
        particles[0] = particle;
        stepMiningParticles(particles, 1.0f);
        ASSERT_EQ(particles.size(), 1);
        EXPECT_FLOAT_EQ(particles[0].x, 8.0f);
        EXPECT_FLOAT_EQ(particles[0].y, 0.75f);
        EXPECT_FALSE(particles[0].landed);
        stepMiningParticles(particles, 5.0f);
        EXPECT_TRUE(particles.empty());
    }

} // namespace lfs::vis::gui::mining
