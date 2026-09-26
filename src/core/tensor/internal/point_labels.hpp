/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "point_math.hpp"
#include <cstdint>

namespace lfs::core::internal {
    struct LabelUpdateParams {
        uint8_t* output;
        const uint8_t *selected, *existing, *locked, *allowed;
        const int32_t *indices, *categories;
        uint32_t count, output_count, allowed_count, label, mode;
    };

    template <uint32_t Phase>
    LFS_POINT_HD inline void updateLabel(const LabelUpdateParams& p, uint32_t row) {
        const int32_t id = p.indices ? p.indices[row] : int32_t(row);
        if (id < 0 || uint32_t(id) >= p.output_count)
            return;
        const uint8_t old = p.existing ? p.existing[id] : 0;
        const bool eligible = !p.categories || (p.categories[row] >= 0 && uint32_t(p.categories[row]) < p.allowed_count && p.allowed[p.categories[row]]);
        if (!eligible) {
            if constexpr (Phase == 0)
                p.output[id] = old;
            return;
        }
        const bool selected = p.selected[row] != 0;
        if constexpr (Phase == 1) {
            if (!selected && old == p.label)
                p.output[id] = 0;
            return;
        }
        if constexpr (Phase == 2) {
            if (!selected)
                return;
        }
        uint8_t value = old;
        if (selected) {
            if (p.mode == 1)
                value = old == p.label ? 0 : old;
            else if (!(old != 0 && old != p.label && p.locked && p.locked[old]))
                value = uint8_t(p.label);
        } else if (p.mode == 2 && old == p.label) {
            value = 0;
        }
        p.output[id] = value;
    }
} // namespace lfs::core::internal
