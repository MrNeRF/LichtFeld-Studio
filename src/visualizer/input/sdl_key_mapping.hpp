/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_scancode.h>

namespace lfs::vis::input {

    int sdlScancodeToAppKey(SDL_Scancode scancode);
    int sdlModsToAppMods(SDL_Keymod sdl_mods);
    int sdlMouseButtonToApp(uint8_t sdl_button);
    SDL_Scancode appKeyToSdlScancode(int app_key);

} // namespace lfs::vis::input
