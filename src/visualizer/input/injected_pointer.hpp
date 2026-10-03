/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_mouse.h>
#include <chrono>
#include <optional>

namespace lfs::vis::input {

    struct InjectedPointerState {
        float x = 0.0f;
        float y = 0.0f;
        SDL_MouseButtonFlags buttons = 0;
    };

    LFS_VIS_API void injectPointerMove(float x, float y);
    LFS_VIS_API void retainInjectedPointerFor(std::chrono::milliseconds duration);
    LFS_VIS_API void injectPointerButton(int sdl_button, bool down);
    LFS_VIS_API void injectPointerWheel();
    LFS_VIS_API void injectPointerModifiers(SDL_Keymod modifiers);
    LFS_VIS_API void pushInjectedPointerEvent(SDL_Event& event);
    [[nodiscard]] LFS_VIS_API bool prepareInjectedPointerEvent(const SDL_Event& event);
    LFS_VIS_API void finishInjectedPointerFrame();
    [[nodiscard]] LFS_VIS_API std::optional<InjectedPointerState> injectedPointer();
    [[nodiscard]] LFS_VIS_API std::optional<InjectedPointerState> lastInjectedPointer();

} // namespace lfs::vis::input
