/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/injected_pointer.hpp"

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
        if (const auto injected = injectedPointer()) {
            if (x)
                *x = injected->x;
            if (y)
                *y = injected->y;
            return injected->buttons;
        }
        const auto buttons = SDL_GetMouseState(x, y);
        const auto scale = windowPixelScale(window);
        if (x)
            *x *= scale.x;
        if (y)
            *y *= scale.y;
        return buttons;
    }

    inline SDL_MouseButtonFlags mouseButtons() {
        if (const auto injected = injectedPointer())
            return injected->buttons;
        return SDL_GetMouseState(nullptr, nullptr);
    }

    inline SDL_MouseButtonFlags mouseStateInWindowCoordinates(SDL_Window* window, float* x,
                                                              float* y) {
        if (const auto injected = injectedPointer()) {
            const auto scale = windowPixelScale(window);
            if (x)
                *x = injected->x / scale.x;
            if (y)
                *y = injected->y / scale.y;
            return injected->buttons;
        }
        return SDL_GetMouseState(x, y);
    }

    inline SDL_MouseButtonFlags globalMouseState(SDL_Window* window, float* x, float* y) {
        if (const auto injected = injectedPointer()) {
            int window_x = 0;
            int window_y = 0;
            SDL_GetWindowPosition(window, &window_x, &window_y);
            const auto scale = windowPixelScale(window);
            if (x)
                *x = static_cast<float>(window_x) + injected->x / scale.x;
            if (y)
                *y = static_cast<float>(window_y) + injected->y / scale.y;
            return injected->buttons;
        }
        return SDL_GetGlobalMouseState(x, y);
    }

    // Pointer position for wheel and gesture events. macOS scrolls the window
    // under the cursor even while it is inactive, but SDL only tracks motion in
    // the key window, so its own position can be stale there.
    inline glm::vec2 wheelPointerInPixels(SDL_Window* window) {
        if (const auto injected = injectedPointer())
            return {injected->x, injected->y};
        float x = 0.0f, y = 0.0f;
#ifdef __APPLE__
        int window_x = 0, window_y = 0;
        if (window && SDL_GetWindowPosition(window, &window_x, &window_y)) {
            globalMouseState(window, &x, &y);
            return glm::vec2(x - static_cast<float>(window_x), y - static_cast<float>(window_y)) *
                   windowPixelScale(window);
        }
#endif
        mouseStateInPixels(window, &x, &y);
        return {x, y};
    }

} // namespace lfs::vis::input
