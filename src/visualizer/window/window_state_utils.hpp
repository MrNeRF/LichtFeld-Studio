/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/environment.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace lfs::vis {

    [[nodiscard]] inline bool automaticWindowStatePersistenceEnabled() {
        return !lfs::core::environment::flag("LFS_SAFE_MODE", false);
    }

    struct WindowRectangle {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;

        bool operator==(const WindowRectangle&) const = default;
    };

    [[nodiscard]] inline bool windowRectangleVisible(
        const WindowRectangle& window, const std::vector<WindowRectangle>& displays,
        const int minimum_visible_width = 96, const int minimum_visible_height = 64) {
        if (window.width <= 0 || window.height <= 0)
            return false;

        for (const auto& display : displays) {
            if (display.width <= 0 || display.height <= 0)
                continue;
            const std::int64_t left = std::max<std::int64_t>(window.x, display.x);
            const std::int64_t top = std::max<std::int64_t>(window.y, display.y);
            const std::int64_t right = std::min<std::int64_t>(
                static_cast<std::int64_t>(window.x) + window.width,
                static_cast<std::int64_t>(display.x) + display.width);
            const std::int64_t bottom = std::min<std::int64_t>(
                static_cast<std::int64_t>(window.y) + window.height,
                static_cast<std::int64_t>(display.y) + display.height);
            if (right - left >= minimum_visible_width &&
                bottom - top >= minimum_visible_height)
                return true;
        }
        return false;
    }

    [[nodiscard]] inline WindowRectangle centerWindowOnDisplay(
        WindowRectangle window, const WindowRectangle& display,
        const int minimum_width = 640, const int minimum_height = 360) {
        if (display.width <= 0 || display.height <= 0)
            return window;
        window.width = std::clamp(window.width, std::min(minimum_width, display.width), display.width);
        window.height = std::clamp(window.height, std::min(minimum_height, display.height), display.height);
        window.x = display.x + (display.width - window.width) / 2;
        window.y = display.y + (display.height - window.height) / 2;
        return window;
    }

} // namespace lfs::vis
