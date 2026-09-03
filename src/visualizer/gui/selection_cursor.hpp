/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/rendering_types.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace lfs::vis::gui {

    enum class SelectionCursorOperation : uint8_t {
        Replace,
        Add,
        Remove,
        Intersect,
    };

    struct SelectionCursorColor {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        uint8_t a = 0;

        bool operator==(const SelectionCursorColor&) const = default;
    };

    struct SelectionCursorImage {
        int size = 0;
        int hotspot = 0;
        int badge_x = -1;
        int badge_y = -1;
        std::vector<uint8_t> rgba;

        [[nodiscard]] bool valid() const {
            return size > 0 && hotspot >= 0 && hotspot < size &&
                   rgba.size() == static_cast<size_t>(size) * size * 4;
        }
    };

    [[nodiscard]] constexpr int selectionCursorPadding() {
        return 8;
    }

    [[nodiscard]] constexpr int selectionCursorMaxSize() {
        return 256;
    }

    [[nodiscard]] bool useHardwareSelectionRing(bool preview_active,
                                                SelectionPreviewMode mode,
                                                int radius_px);

    [[nodiscard]] SelectionCursorImage makeSelectionCursorImage(
        int radius_px,
        SelectionCursorColor color,
        std::span<const uint8_t> badge_rgba = {},
        int badge_width = 0,
        int badge_height = 0);

} // namespace lfs::vis::gui
