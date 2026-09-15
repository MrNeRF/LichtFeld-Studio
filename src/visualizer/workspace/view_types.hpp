/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/render_constants.hpp"

#include <cstddef>
#include <cstdint>

namespace lfs::vis {

    // Ratios are strictly between zero and one. The layout solver applies
    // minimum pane sizes using the containing subtree's pixel extent.
    inline constexpr float kMinSplitRatio = 0.0f;
    inline constexpr float kMaxSplitRatio = 1.0f;
    inline constexpr int kDefaultMinPanePixels = 64;
    inline constexpr int kDefaultDividerPixels = 4;

    enum class SplitAxis : std::uint8_t {
        Horizontal = 0,
        Vertical = 1,
    };

    enum class LayoutNodeKind : std::uint8_t {
        Leaf = 0,
        Split = 1,
    };

    enum class LayoutPreset : std::uint8_t {
        Single = 0,
        DualHorizontal = 1,
        DualVertical = 2,
        Quad = 3,
    };

    // Axis-aligned integer rectangle with half-open bounds.
    struct ViewRect {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;

        [[nodiscard]] constexpr bool empty() const noexcept { return width <= 0 || height <= 0; }
        [[nodiscard]] constexpr int right() const noexcept { return x + width; }
        [[nodiscard]] constexpr int bottom() const noexcept { return y + height; }
        [[nodiscard]] constexpr int area() const noexcept { return empty() ? 0 : width * height; }
        [[nodiscard]] constexpr bool contains(const int px, const int py) const noexcept {
            return !empty() && px >= x && px < right() && py >= y && py < bottom();
        }
        [[nodiscard]] friend constexpr bool operator==(const ViewRect&, const ViewRect&) noexcept = default;
    };

    struct ViewProjectionState {
        float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        bool orthographic = false;
        float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
        float near_plane = lfs::rendering::DEFAULT_NEAR_PLANE;
        float far_plane = lfs::rendering::DEFAULT_FAR_PLANE;
        bool equirectangular = false;

        [[nodiscard]] friend bool operator==(const ViewProjectionState&, const ViewProjectionState&) = default;
    };

    [[nodiscard]] constexpr std::size_t presetLeafCount(const LayoutPreset preset) noexcept {
        switch (preset) {
        case LayoutPreset::Single:
            return 1;
        case LayoutPreset::DualHorizontal:
        case LayoutPreset::DualVertical:
            return 2;
        case LayoutPreset::Quad:
            return 4;
        }
        return 1;
    }

} // namespace lfs::vis
