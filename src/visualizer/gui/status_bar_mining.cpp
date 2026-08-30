/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/status_bar_mining.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <random>
#include <string_view>

namespace lfs::vis::gui::mining {
    namespace {
        constexpr int kBlockDp = 16;
        constexpr float kGroundDp = 15.0f;
        constexpr float kGravity = 110.0f;
        constexpr int kSmokeLettersFromMs = 500;
        constexpr int kSmokeLetterEveryMs = 500;
        constexpr int kSmokeFrameMs = 280;
        constexpr int kSmokeWordLength = 9;
        constexpr char kSmokeWord[] = "LichtFeld";
        constexpr std::array<std::array<std::string_view, 5>, 9> kSmokeGlyphs = {{
            {"100", "100", "100", "100", "111"},
            {"010", "000", "010", "010", "010"},
            {"000", "011", "100", "100", "011"},
            {"100", "100", "110", "101", "101"},
            {"010", "111", "010", "010", "011"},
            {"111", "100", "110", "100", "100"},
            {"010", "101", "111", "100", "011"},
            {"010", "010", "010", "010", "011"},
            {"001", "001", "011", "101", "011"},
        }};
        constexpr std::array<char, 9> kSmokeGlyphKeys = {'L', 'i', 'c', 'h', 't', 'F', 'e', 'l', 'd'};
        constexpr uint32_t kSmokeLetterRgb = 0xdedee6;
        constexpr uint32_t kSmokeLetterShadowRgb = 0x222228;
        constexpr uint8_t kSmokeLetterAlpha = 240;
        constexpr uint8_t kSmokeLetterShadowAlpha = 150;

        struct Rgb {
            int r;
            int g;
            int b;
        };

        constexpr Rgb kStone[] = {{125, 125, 125}, {110, 110, 110}, {96, 96, 96}};
        constexpr Rgb kDirt[] = {{134, 96, 67}, {115, 81, 53}, {97, 67, 42}};

        struct Ore {
            Rgb blob[2];
        };

        constexpr Ore kCoal{{{38, 38, 38}, {58, 58, 58}}};
        constexpr Ore kIron{{{216, 175, 147}, {178, 140, 116}}};
        constexpr Ore kGold{{{250, 220, 96}, {202, 172, 60}}};
        constexpr Ore kDiamond{{{102, 219, 214}, {58, 178, 190}}};

        uint32_t packRgb(const Rgb color) {
            return (static_cast<uint32_t>(color.r) << 16) |
                   (static_cast<uint32_t>(color.g) << 8) | static_cast<uint32_t>(color.b);
        }

        Rgb sampleBlockColor(const char* type, std::mt19937& rng) {
            const Rgb* base = std::strcmp(type, "dirt") == 0 ? kDirt : kStone;
            int base_count = 3;
            const Ore* ore = nullptr;
            if (std::strcmp(type, "coal") == 0)
                ore = &kCoal;
            else if (std::strcmp(type, "iron") == 0)
                ore = &kIron;
            else if (std::strcmp(type, "gold") == 0)
                ore = &kGold;
            else if (std::strcmp(type, "diamond-ore") == 0)
                ore = &kDiamond;

            const int palette_size = base_count + (ore ? 2 : 0);
            const int selected = std::uniform_int_distribution<int>(0, palette_size - 1)(rng);
            return selected < base_count ? base[selected] : ore->blob[selected - base_count];
        }

        float randomFloat(std::mt19937& rng, const float min, const float max) {
            return std::uniform_real_distribution<float>(min, max)(rng);
        }

        const std::array<std::string_view, 5>& smokeGlyph(const char key) {
            const auto it = std::find(kSmokeGlyphKeys.begin(), kSmokeGlyphKeys.end(), key);
            return kSmokeGlyphs[static_cast<size_t>(std::distance(kSmokeGlyphKeys.begin(), it))];
        }

        bool smokeGlyphPixel(const std::array<std::string_view, 5>& glyph, const int x,
                             const int y) {
            return y >= 0 && y < 5 && x >= 0 && x < 3 && glyph[static_cast<size_t>(y)][x] == '1';
        }

        uint8_t miningParticleAlpha(const MiningParticle& particle) {
            if (!particle.fade || particle.life <= 0.0f)
                return particle.alpha;
            const float remaining = std::max(0.0f, 1.0f - particle.age / particle.life);
            return static_cast<uint8_t>(static_cast<float>(particle.alpha) *
                                        std::pow(remaining, 0.7f));
        }
    } // namespace

    MiningLayout miningLayout(const float bar_dp) {
        const int block_count = std::max(1, static_cast<int>(bar_dp / static_cast<float>(kBlockDp)));
        const float offset_dp = bar_dp - static_cast<float>(block_count * kBlockDp);
        return {block_count, offset_dp};
    }

    float miningProgressTextLeftDp(const float fill_edge_dp, const float bar_dp,
                                   const float text_width_dp) {
        const float max_left_dp = std::max(0.0f, bar_dp - text_width_dp - 2.0f);
        return std::clamp(fill_edge_dp + 48.0f, 0.0f, max_left_dp);
    }

    const char* miningBlockType(const int index, const int block_count) {
        if (index == block_count - 1)
            return "diamond-ore";
        uint32_t h = static_cast<uint32_t>(index) * 2654435761u;
        h ^= h >> 13;
        h *= 2246822519u;
        h ^= h >> 16;
        switch (h % 20) {
        case 0:
        case 1:
            return "coal";
        case 2:
        case 3:
            return "iron";
        case 4:
            return "gold";
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
            return "dirt";
        default:
            return "stone";
        }
    }

    int miningCrackStage(const float fill_dp, const int current_block) {
        return std::clamp(
            static_cast<int>(4.0f * (fill_dp - static_cast<float>(current_block * kBlockDp)) /
                             static_cast<float>(kBlockDp)),
            0, 3);
    }

    std::string buildMiningWallRml(const MiningLayout& layout, const int current_block,
                                   const int crack_stage) {
        std::string wall_rml;
        for (int i = current_block; i < layout.block_count; ++i) {
            const int left = static_cast<int>(std::round(
                layout.offset_dp + static_cast<float>(i * kBlockDp)));
            wall_rml += std::format(
                "<img class=\"wall-block\" style=\"left:{}dp\" src=\"../icon/mining/{}.png\"/>",
                left, miningBlockType(i, layout.block_count));
            if (i == current_block && crack_stage > 0) {
                wall_rml += std::format(
                    "<img class=\"wall-block\" style=\"left:{}dp\" src=\"../icon/mining/crack-{}.png\"/>",
                    left, crack_stage);
            }
        }
        return wall_rml;
    }

    void spawnBreakParticles(std::vector<MiningParticle>& particles, const MiningLayout& layout,
                             const int block, const char* type) {
        if (block < 0 || !type)
            return;

        std::mt19937 rng(static_cast<uint32_t>(block));
        const float left = layout.offset_dp + static_cast<float>(block * kBlockDp);
        const int sizes[] = {1, 1, 2, 2, 2};
        for (int i = 0; i < 16; ++i) {
            const int px = std::uniform_int_distribution<int>(0, 15)(rng);
            const int py = std::uniform_int_distribution<int>(0, 15)(rng);
            const int size = sizes[std::uniform_int_distribution<int>(0, 4)(rng)];
            particles.push_back({
                .x = left + static_cast<float>(px),
                .y = static_cast<float>(py),
                .vx = randomFloat(rng, -26.0f, 30.0f),
                .vy = randomFloat(rng, -42.0f, 4.0f),
                .size = size,
                .rgb = packRgb(sampleBlockColor(type, rng)),
                .life = randomFloat(rng, 0.45f, 0.8f),
            });
        }
    }

    void spawnStrikeChips(std::vector<MiningParticle>& particles, const float hit_x,
                          const float hit_y, const char* type, const uint32_t seed) {
        if (!type)
            return;

        std::mt19937 rng(seed);
        for (int i = 0; i < 4; ++i) {
            particles.push_back({
                .x = hit_x + randomFloat(rng, -1.0f, 1.0f),
                .y = hit_y + randomFloat(rng, -1.5f, 1.5f),
                .vx = randomFloat(rng, -30.0f, -4.0f),
                .vy = randomFloat(rng, -30.0f, -6.0f),
                .size = 1,
                .rgb = packRgb(sampleBlockColor(type, rng)),
                .life = randomFloat(rng, 0.3f, 0.45f),
            });
        }
    }

    void stepMiningParticles(std::vector<MiningParticle>& particles, const float dt_s) {
        if (dt_s <= 0.0f)
            return;

        std::erase_if(particles, [dt_s](MiningParticle& particle) {
            particle.age += dt_s;
            if (particle.age >= particle.life)
                return true;
            if (!particle.gravity) {
                particle.x += particle.vx * dt_s;
                particle.y += particle.vy * dt_s;
                return false;
            }
            if (particle.landed) {
                particle.x += particle.vx * dt_s;
                return false;
            }

            particle.vy += kGravity * dt_s;
            particle.x += particle.vx * dt_s;
            particle.y += particle.vy * dt_s;
            if (particle.y + static_cast<float>(particle.size) >= kGroundDp) {
                particle.y = kGroundDp - static_cast<float>(particle.size);
                particle.landed = true;
                particle.vx *= 0.3f;
            }
            return false;
        });
    }

    std::string buildMiningParticlesRml(const std::vector<MiningParticle>& particles) {
        std::string rml;
        for (const auto& particle : particles) {
            const int x = static_cast<int>(std::round(particle.x));
            const int y = static_cast<int>(std::round(particle.y));
            const uint32_t rgba = (particle.rgb << 8) | miningParticleAlpha(particle);
            rml += std::format(
                "<div class=\"mining-particle\" style=\"left:{}dp;top:{}dp;width:{}dp;height:{}dp;background-color:#{:08x}\"></div>",
                x, y, particle.size, particle.size, rgba);
        }
        return rml;
    }

    int miningSmokeSpriteIndex(const int pause_ms) {
        constexpr int word_end_ms = kSmokeLettersFromMs + kSmokeLetterEveryMs * kSmokeWordLength;
        if (pause_ms < kSmokeLettersFromMs)
            return 0;
        if (pause_ms < word_end_ms)
            return (pause_ms - kSmokeLettersFromMs) % kSmokeLetterEveryMs < 250 ? 1 : 0;
        return ((pause_ms - word_end_ms) / kSmokeFrameMs) % 6;
    }

    void spawnSmokeLetters(std::vector<MiningParticle>& particles, const float x, const float y,
                           const int prev_pause_ms, const int pause_ms) {
        for (int index = 0; index < kSmokeWordLength; ++index) {
            const int at = kSmokeLettersFromMs + index * kSmokeLetterEveryMs;
            if (prev_pause_ms >= at || at > pause_ms)
                continue;

            const auto& glyph = smokeGlyph(kSmokeWord[kSmokeWordLength - 1 - index]);
            for (int gy = 0; gy < 5; ++gy) {
                for (int gx = 0; gx < 3; ++gx) {
                    if (!smokeGlyphPixel(glyph, gx, gy))
                        continue;
                    if (!smokeGlyphPixel(glyph, gx + 1, gy + 1)) {
                        particles.push_back({
                            .x = x + static_cast<float>(gx + 1),
                            .y = y + static_cast<float>(gy + 1),
                            .vx = 8.0f,
                            .vy = -0.25f,
                            .size = 1,
                            .rgb = kSmokeLetterShadowRgb,
                            .life = 6.0f,
                            .gravity = false,
                            .fade = true,
                            .alpha = kSmokeLetterShadowAlpha,
                        });
                    }
                }
            }
            for (int gy = 0; gy < 5; ++gy) {
                for (int gx = 0; gx < 3; ++gx) {
                    if (!smokeGlyphPixel(glyph, gx, gy))
                        continue;
                    particles.push_back({
                        .x = x + static_cast<float>(gx),
                        .y = y + static_cast<float>(gy),
                        .vx = 8.0f,
                        .vy = -0.25f,
                        .size = 1,
                        .rgb = kSmokeLetterRgb,
                        .life = 6.0f,
                        .gravity = false,
                        .fade = true,
                        .alpha = kSmokeLetterAlpha,
                    });
                }
            }
        }
    }

} // namespace lfs::vis::gui::mining
