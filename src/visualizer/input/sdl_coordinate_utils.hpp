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

} // namespace lfs::vis::input
