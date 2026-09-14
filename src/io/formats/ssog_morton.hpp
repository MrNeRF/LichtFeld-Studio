/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../cuda/morton_encoding.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace lfs::io {
    inline void sort_ssog_leaf(const float* positions, std::span<int> rows) {
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
            multiplier[axis] = morton_multiplier(extent);
        }
        struct KeyRow {
            uint64_t key;
            int row;
        };
        std::vector<KeyRow> keys;
        keys.reserve(rows.size());
        for (int row : rows) {
            uint64_t key = 0;
            for (int axis = 0; axis < 3; ++axis) {
                key |= morton_spread(morton_coordinate(positions[size_t(row) * 3 + axis], low[axis], multiplier[axis])) << axis;
            }
            keys.push_back({key, row});
        }
        std::stable_sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        for (size_t i = 0; i < rows.size(); ++i)
            rows[i] = keys[i].row;
    }
} // namespace lfs::io
