/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

#include <string>

namespace lfs::vis::gui::mining {

    struct MiningLayout {
        int block_count = 0;
        float offset_dp = 0.0f;
    };

    [[nodiscard]] LFS_VIS_API MiningLayout miningLayout(float bar_dp);
    [[nodiscard]] LFS_VIS_API const char* miningBlockType(int index, int block_count);
    [[nodiscard]] LFS_VIS_API int miningCrackStage(float fill_dp, int current_block);
    [[nodiscard]] LFS_VIS_API std::string buildMiningWallRml(const MiningLayout& layout,
                                                             int current_block,
                                                             int crack_stage);
    [[nodiscard]] LFS_VIS_API std::string buildMiningDebrisRml(const MiningLayout& layout,
                                                               int block,
                                                               int break_frame);

} // namespace lfs::vis::gui::mining
