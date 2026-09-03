/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    [[nodiscard]] inline std::optional<std::pair<std::size_t, std::size_t>>
    cameraThumbnailVisibleRowRange(const std::size_t a,
                                   const std::size_t b,
                                   const std::size_t row_count) noexcept {
        const auto [lower, upper] = std::minmax(a, b);
        const std::size_t start = std::min(lower, row_count);
        const std::size_t end = std::min(upper, row_count);
        if (a > b || start >= end)
            return std::nullopt;
        return std::pair{start, end};
    }

    // The viewport overlay supplies all cameras under effectively visible
    // camera groups. Selected cameras and overscanned scene-graph rows only
    // affect priority; they never remove a drawable camera from the result.
    [[nodiscard]] inline std::vector<int> cameraThumbnailRequestOrder(
        const std::span<const int> all_camera_uids,
        const std::span<const int> visible_row_uids,
        const std::span<const int> selected_uids) {
        std::unordered_set<int> available;
        available.reserve(all_camera_uids.size());
        for (const int uid : all_camera_uids) {
            if (uid >= 0)
                available.insert(uid);
        }

        std::vector<int> result;
        result.reserve(available.size());
        std::unordered_set<int> added;
        added.reserve(available.size());
        const auto append = [&](const std::span<const int> uids) {
            for (const int uid : uids) {
                if (available.contains(uid) && added.insert(uid).second)
                    result.push_back(uid);
            }
        };
        append(selected_uids);
        append(visible_row_uids);
        append(all_camera_uids);
        return result;
    }

} // namespace lfs::vis::gui
