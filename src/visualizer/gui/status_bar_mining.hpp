/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lfs::vis::gui::mining {

    struct MiningLayout {
        int block_count = 0;
        float offset_dp = 0.0f;
    };

    struct LFS_VIS_API MiningParticle {
        float x = 0.0f;
        float y = 0.0f;
        float vx = 0.0f;
        float vy = 0.0f;
        int size = 1;
        uint32_t rgb = 0;
        float life = 0.0f;
        float age = 0.0f;
        bool landed = false;
        bool gravity = true;
        bool fade = false;
        uint8_t alpha = 255;
    };

    [[nodiscard]] LFS_VIS_API MiningLayout miningLayout(float bar_dp);
    [[nodiscard]] LFS_VIS_API float miningProgressTextLeftDp(float fill_edge_dp, float bar_dp,
                                                             float text_width_dp);
    [[nodiscard]] LFS_VIS_API const char* miningBlockType(int index, int block_count);
    [[nodiscard]] LFS_VIS_API int miningCrackStage(float fill_dp, int current_block);
    [[nodiscard]] LFS_VIS_API std::string buildMiningWallRml(const MiningLayout& layout,
                                                             int current_block,
                                                             int crack_stage);
    LFS_VIS_API void spawnBreakParticles(std::vector<MiningParticle>& particles,
                                         const MiningLayout& layout, int block,
                                         const char* type);
    LFS_VIS_API void spawnStrikeChips(std::vector<MiningParticle>& particles, float hit_x,
                                      float hit_y, const char* type, uint32_t seed);
    LFS_VIS_API void stepMiningParticles(std::vector<MiningParticle>& particles, float dt_s);
    [[nodiscard]] LFS_VIS_API std::string buildMiningParticlesRml(
        const std::vector<MiningParticle>& particles);
    [[nodiscard]] LFS_VIS_API int miningSmokeSpriteIndex(int pause_ms);
    LFS_VIS_API void spawnSmokeLetters(std::vector<MiningParticle>& particles, float x, float y,
                                       int prev_pause_ms, int pause_ms);

} // namespace lfs::vis::gui::mining
