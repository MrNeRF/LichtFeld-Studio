/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cstddef>

namespace lfs::vis::gui {

    [[nodiscard]] constexpr int insertionLineTop(const std::size_t boundary,
                                                 const std::size_t row_count) noexcept {
        constexpr int row_height_dp = 20;
        constexpr int line_height_dp = 2;
        constexpr int edge_inset_dp = 1;
        const int content_height_dp = static_cast<int>(row_count) * row_height_dp;
        const int max_top_dp = std::max(edge_inset_dp, content_height_dp - line_height_dp);
        return std::clamp(static_cast<int>(boundary) * row_height_dp,
                          edge_inset_dp,
                          max_top_dp);
    }

    struct DropTargetComputation {
        // These are flat-row coordinates. The caller replaces parent with the
        // hovered node's actual parent id after looking up the row snapshot.
        int parent = -1;
        int index = -1;
        bool into_group = false;
        bool show_line = false;
        int line_top_dp = 0;
        int line_left_dp = 4;
    };

    [[nodiscard]] constexpr DropTargetComputation computeDropTarget(
        const std::size_t hovered_fidx,
        const int depth,
        const float rel,
        const bool is_group,
        const std::size_t row_count) noexcept {
        if (hovered_fidx >= row_count) {
            return DropTargetComputation{
                .parent = -1,
                .index = static_cast<int>(row_count),
                .into_group = false,
                .show_line = true,
                .line_top_dp = insertionLineTop(row_count, row_count),
                .line_left_dp = 4,
            };
        }

        if (is_group && rel > 0.2f && rel < 0.8f) {
            return DropTargetComputation{
                .parent = static_cast<int>(hovered_fidx),
                .index = -1,
                .into_group = true,
                .show_line = false,
                .line_top_dp = 0,
                .line_left_dp = 4,
            };
        }

        const bool after = rel >= 0.5f;
        const int index = static_cast<int>(hovered_fidx) + (after ? 1 : 0);
        return DropTargetComputation{
            .parent = -1,
            .index = index,
            .into_group = false,
            .show_line = true,
            .line_top_dp = insertionLineTop(static_cast<std::size_t>(index), row_count),
            .line_left_dp = 21 + std::max(depth, 0) * 16,
        };
    }

} // namespace lfs::vis::gui
