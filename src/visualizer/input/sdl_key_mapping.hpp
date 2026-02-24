/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>

namespace lfs::vis::input {

    int sdlKeycodeToAppKey(SDL_Keycode sdl_key);
    int sdlModsToAppMods(SDL_Keymod sdl_mods);
    int sdlMouseButtonToApp(uint8_t sdl_button);
    SDL_Keycode appKeyToSdlKeycode(int app_key);

} // namespace lfs::vis::input
