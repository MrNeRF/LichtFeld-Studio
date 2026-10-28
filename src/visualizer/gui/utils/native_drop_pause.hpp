/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <SDL3/SDL.h>

namespace lfs::vis::gui {
    // SDL's Linux backends update XdndAware / Wayland's offer acceptance when
    // both file and text reception are disabled. Our separate source stays live.
    class NativeDropPause {
        bool files_ = SDL_EventEnabled(SDL_EVENT_DROP_FILE);
        bool text_ = SDL_EventEnabled(SDL_EVENT_DROP_TEXT);

    public:
        NativeDropPause() {
            SDL_SetEventEnabled(SDL_EVENT_DROP_FILE, false);
            SDL_SetEventEnabled(SDL_EVENT_DROP_TEXT, false);
        }
        ~NativeDropPause() {
            SDL_FlushEvents(SDL_EVENT_DROP_FILE, SDL_EVENT_DROP_POSITION);
            SDL_SetEventEnabled(SDL_EVENT_DROP_FILE, files_);
            SDL_SetEventEnabled(SDL_EVENT_DROP_TEXT, text_);
        }
        NativeDropPause(const NativeDropPause&) = delete;
        NativeDropPause& operator=(const NativeDropPause&) = delete;
    };
} // namespace lfs::vis::gui
