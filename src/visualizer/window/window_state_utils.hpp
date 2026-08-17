/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/environment.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
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

    [[nodiscard]] inline std::optional<WindowRectangle> displayWithLargestIntersection(
        const WindowRectangle& window, const std::vector<WindowRectangle>& displays) {
        std::optional<WindowRectangle> best;
        std::int64_t best_area = 0;
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
            const std::int64_t area = std::max<std::int64_t>(0, right - left) *
                                      std::max<std::int64_t>(0, bottom - top);
            if (area > best_area) {
                best_area = area;
                best = display;
            }
        }
        return best;
    }

    [[nodiscard]] inline WindowRectangle recoverWindowRectangle(
        const WindowRectangle& window, const std::vector<WindowRectangle>& displays,
        const WindowRectangle& fallback_display, const int minimum_visible_width = 96,
        const int minimum_visible_height = 64, const int minimum_width = 640,
        const int minimum_height = 360) {
        const auto intersected_display = displayWithLargestIntersection(window, displays);
        if (!windowRectangleVisible(window, displays, minimum_visible_width,
                                    minimum_visible_height)) {
            return centerWindowOnDisplay(window, fallback_display, minimum_width, minimum_height);
        }

        if (intersected_display &&
            (window.width > intersected_display->width ||
             window.height > intersected_display->height)) {
            return centerWindowOnDisplay(window, *intersected_display, minimum_width, minimum_height);
        }
        return window;
    }

} // namespace lfs::vis
