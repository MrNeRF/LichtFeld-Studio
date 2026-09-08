/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace lfs::io {
    // Same 21-bit per-axis normalization and stable ties as the CUDA SOG
    // sorter. Work directly on host level rows, without a CUDA round trip per leaf.
    inline void sort_streamed_sog_leaf(const float* positions, std::span<int> rows) {
        if (rows.empty())
            return;
        std::array<float, 3> low, high, multiplier;
        low.fill(std::numeric_limits<float>::infinity());
        high.fill(-std::numeric_limits<float>::infinity());
        for (int row : rows)
            for (int axis = 0; axis < 3; ++axis) {
                low[axis] = std::min(low[axis], positions[size_t(row) * 3 + axis]);
                high[axis] = std::max(high[axis], positions[size_t(row) * 3 + axis]);
            }
        for (int axis = 0; axis < 3; ++axis) {
            const float extent = high[axis] - low[axis];
            multiplier[axis] = extent == 0 ? 0 : float(1u << 21) / extent;
        }
        const auto spread = [](uint64_t x) {
            x &= 0x1fffffULL;
            x = (x | (x << 32)) & 0x1f00000000ffffULL;
            x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
            x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
            x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
            return (x | (x << 2)) & 0x1249249249249249ULL;
        };
        struct KeyRow {
            uint64_t key;
            int row;
        };
        std::vector<KeyRow> keys;
        keys.reserve(rows.size());
        for (int row : rows) {
            uint64_t key = 0;
            for (int axis = 0; axis < 3; ++axis) {
                const float normalized = (positions[size_t(row) * 3 + axis] - low[axis]) * multiplier[axis];
                const auto coordinate = std::min((1u << 21) - 1, static_cast<uint32_t>(normalized));
                key |= spread(coordinate) << axis;
            }
            keys.push_back({key, row});
        }
        std::stable_sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        for (size_t i = 0; i < rows.size(); ++i)
            rows[i] = keys[i].row;
    }
} // namespace lfs::io
