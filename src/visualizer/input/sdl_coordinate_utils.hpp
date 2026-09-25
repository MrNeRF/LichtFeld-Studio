/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <glm/vec2.hpp>

namespace lfs::vis::input {

    // UI layout and Vulkan drawing use framebuffer pixels. SDL window geometry
    // and pointer events use logical coordinates on high-density displays.
    inline glm::vec2 windowPixelScale(SDL_Window* window) {
        int width = 0, height = 0, pixel_width = 0, pixel_height = 0;
        if (!window || !SDL_GetWindowSize(window, &width, &height) ||
            !SDL_GetWindowSizeInPixels(window, &pixel_width, &pixel_height) ||
            width <= 0 || height <= 0 || pixel_width <= 0 || pixel_height <= 0)
            return {1.0f, 1.0f};
        return {static_cast<float>(pixel_width) / width,
                static_cast<float>(pixel_height) / height};
    }

    inline SDL_MouseButtonFlags mouseStateInPixels(SDL_Window* window, float* x, float* y) {
        const auto buttons = SDL_GetMouseState(x, y);
        const auto scale = windowPixelScale(window);
        if (x)
            *x *= scale.x;
        if (y)
            *y *= scale.y;
        return buttons;
    }

    // Pointer position for wheel and gesture events. macOS scrolls the window
    // under the cursor even while it is inactive, but SDL only tracks motion in
    // the key window, so its own position can be stale there.
    inline glm::vec2 wheelPointerInPixels(SDL_Window* window) {
        float x = 0.0f, y = 0.0f;
#ifdef __APPLE__
        int window_x = 0, window_y = 0;
        if (window && SDL_GetWindowPosition(window, &window_x, &window_y)) {
            SDL_GetGlobalMouseState(&x, &y);
            return glm::vec2(x - static_cast<float>(window_x), y - static_cast<float>(window_y)) *
                   windowPixelScale(window);
        }
#endif
        mouseStateInPixels(window, &x, &y);
        return {x, y};
    }

} // namespace lfs::vis::input
