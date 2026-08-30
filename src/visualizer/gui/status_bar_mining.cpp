/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/status_bar_mining.hpp"

#include <algorithm>
#include <cstdint>
#include <format>

namespace lfs::vis::gui::mining {
    namespace {
        constexpr int kBlockDp = 16;
    } // namespace

    MiningLayout miningLayout(const float bar_dp) {
        const int block_count = std::max(1, static_cast<int>(bar_dp / static_cast<float>(kBlockDp)));
        const float offset_dp = bar_dp - static_cast<float>(block_count * kBlockDp);
        return {block_count, offset_dp};
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
            const float left = layout.offset_dp + static_cast<float>(i * kBlockDp);
            wall_rml += std::format(
                "<img class=\"wall-block\" style=\"left: {:.2f}dp\" src=\"../icon/mining/{}.png\"/>",
                left, miningBlockType(i, layout.block_count));
            if (i == current_block && crack_stage > 0) {
                wall_rml += std::format(
                    "<img class=\"wall-block\" style=\"left: {:.2f}dp\" src=\"../icon/mining/crack-{}.png\"/>",
                    left, crack_stage);
            }
        }
        return wall_rml;
    }

    std::string buildMiningDebrisRml(const MiningLayout& layout, const int block,
                                     const int break_frame) {
        if (block < 0 || break_frame < 0)
            return {};
        const float left = layout.offset_dp + static_cast<float>(block * kBlockDp);
        return std::format(
            "<img class=\"wall-block\" style=\"left: {:.2f}dp\" src=\"../icon/mining/break-{}.png\"/>",
            left, break_frame + 1);
    }

} // namespace lfs::vis::gui::mining
