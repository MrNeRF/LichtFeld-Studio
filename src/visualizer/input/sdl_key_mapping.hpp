/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_scancode.h>

namespace lfs::vis::input {

    LFS_VIS_API int sdlScancodeToAppKey(SDL_Scancode scancode);
    LFS_VIS_API int sdlModsToAppMods(SDL_Keymod sdl_mods);
    LFS_VIS_API int sdlMouseButtonToApp(uint8_t sdl_button);
    LFS_VIS_API SDL_Scancode appKeyToSdlScancode(int app_key);

} // namespace lfs::vis::input
