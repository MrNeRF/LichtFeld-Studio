/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <glm/vec2.hpp>

namespace lfs::vis::input {

    // UI layout and Vulkan drawing use framebuffer pixels. SDL window geometry
    // and pointer events use logical coordinates on high-density displays.
    // SDL_GetMouseFocus() can return null outside the window; no window scale
    // is available then, so callers retain the unscaled SDL coordinates.
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

    // Immediate GUI dispatch must use the same pixels as frame-time polling.
    // Keep the original event in logical coordinates for native window handling.
    inline SDL_Event pointerEventInPixels(const SDL_Event& native_event, SDL_Window* window) {
        SDL_Event event = native_event;
        const auto scale = windowPixelScale(window);
        switch (event.type) {
        case SDL_EVENT_MOUSE_MOTION:
            event.motion.x *= scale.x;
            event.motion.y *= scale.y;
            event.motion.xrel *= scale.x;
            event.motion.yrel *= scale.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            event.button.x *= scale.x;
            event.button.y *= scale.y;
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            event.wheel.mouse_x *= scale.x;
            event.wheel.mouse_y *= scale.y;
            break;
        default:
            break;
        }
        return event;
    }

} // namespace lfs::vis::input
