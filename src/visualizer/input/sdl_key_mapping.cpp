/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/sdl_key_mapping.hpp"
#include "input/key_codes.hpp"

namespace lfs::vis::input {

    int sdlScancodeToAppKey(SDL_Scancode scancode) {
        if (scancode >= SDL_SCANCODE_A && scancode <= SDL_SCANCODE_Z)
            return KEY_A + (scancode - SDL_SCANCODE_A);

        // SDL scancodes: 1-9 (30-38), 0 (39)
        if (scancode >= SDL_SCANCODE_1 && scancode <= SDL_SCANCODE_0) {
            const int digit = (scancode - SDL_SCANCODE_1 + 1) % 10;
            return KEY_0 + digit;
        }

        if (scancode >= SDL_SCANCODE_KP_1 && scancode <= SDL_SCANCODE_KP_9) {
            return KEY_KP_1 + (scancode - SDL_SCANCODE_KP_1);
        }
        if (scancode == SDL_SCANCODE_KP_0) {
            return KEY_KP_0;
        }

        switch (scancode) {
        case SDL_SCANCODE_SPACE: return KEY_SPACE;
        case SDL_SCANCODE_APOSTROPHE: return KEY_APOSTROPHE;
        case SDL_SCANCODE_COMMA: return KEY_COMMA;
        case SDL_SCANCODE_MINUS: return KEY_MINUS;
        case SDL_SCANCODE_PERIOD: return KEY_PERIOD;
        case SDL_SCANCODE_SLASH: return KEY_SLASH;
        case SDL_SCANCODE_SEMICOLON: return KEY_SEMICOLON;
        case SDL_SCANCODE_EQUALS: return KEY_EQUAL;
        case SDL_SCANCODE_LEFTBRACKET: return KEY_LEFT_BRACKET;
        case SDL_SCANCODE_BACKSLASH: return KEY_BACKSLASH;
        case SDL_SCANCODE_RIGHTBRACKET: return KEY_RIGHT_BRACKET;
        case SDL_SCANCODE_GRAVE: return KEY_GRAVE_ACCENT;
        case SDL_SCANCODE_ESCAPE: return KEY_ESCAPE;
        case SDL_SCANCODE_RETURN: return KEY_ENTER;
        case SDL_SCANCODE_TAB: return KEY_TAB;
        case SDL_SCANCODE_BACKSPACE: return KEY_BACKSPACE;
        case SDL_SCANCODE_INSERT: return KEY_INSERT;
        case SDL_SCANCODE_DELETE: return KEY_DELETE;
        case SDL_SCANCODE_RIGHT: return KEY_RIGHT;
        case SDL_SCANCODE_LEFT: return KEY_LEFT;
        case SDL_SCANCODE_DOWN: return KEY_DOWN;
        case SDL_SCANCODE_UP: return KEY_UP;
        case SDL_SCANCODE_PAGEUP: return KEY_PAGE_UP;
        case SDL_SCANCODE_PAGEDOWN: return KEY_PAGE_DOWN;
        case SDL_SCANCODE_HOME: return KEY_HOME;
        case SDL_SCANCODE_END: return KEY_END;
        case SDL_SCANCODE_CAPSLOCK: return KEY_CAPS_LOCK;
        case SDL_SCANCODE_SCROLLLOCK: return KEY_SCROLL_LOCK;
        case SDL_SCANCODE_NUMLOCKCLEAR: return KEY_NUM_LOCK;
        case SDL_SCANCODE_PRINTSCREEN: return KEY_PRINT_SCREEN;
        case SDL_SCANCODE_PAUSE: return KEY_PAUSE;
        case SDL_SCANCODE_F1: return KEY_F1;
        case SDL_SCANCODE_F2: return KEY_F2;
        case SDL_SCANCODE_F3: return KEY_F3;
        case SDL_SCANCODE_F4: return KEY_F4;
        case SDL_SCANCODE_F5: return KEY_F5;
        case SDL_SCANCODE_F6: return KEY_F6;
        case SDL_SCANCODE_F7: return KEY_F7;
        case SDL_SCANCODE_F8: return KEY_F8;
        case SDL_SCANCODE_F9: return KEY_F9;
        case SDL_SCANCODE_F10: return KEY_F10;
        case SDL_SCANCODE_F11: return KEY_F11;
        case SDL_SCANCODE_F12: return KEY_F12;
        case SDL_SCANCODE_KP_PERIOD: return KEY_KP_DECIMAL;
        case SDL_SCANCODE_KP_DIVIDE: return KEY_KP_DIVIDE;
        case SDL_SCANCODE_KP_MULTIPLY: return KEY_KP_MULTIPLY;
        case SDL_SCANCODE_KP_MINUS: return KEY_KP_SUBTRACT;
        case SDL_SCANCODE_KP_PLUS: return KEY_KP_ADD;
        case SDL_SCANCODE_KP_ENTER: return KEY_KP_ENTER;
        case SDL_SCANCODE_KP_EQUALS: return KEY_KP_EQUAL;
        case SDL_SCANCODE_LSHIFT: return KEY_LEFT_SHIFT;
        case SDL_SCANCODE_LCTRL: return KEY_LEFT_CONTROL;
        case SDL_SCANCODE_LALT: return KEY_LEFT_ALT;
        case SDL_SCANCODE_LGUI: return KEY_LEFT_SUPER;
        case SDL_SCANCODE_RSHIFT: return KEY_RIGHT_SHIFT;
        case SDL_SCANCODE_RCTRL: return KEY_RIGHT_CONTROL;
        case SDL_SCANCODE_RALT: return KEY_RIGHT_ALT;
        case SDL_SCANCODE_RGUI: return KEY_RIGHT_SUPER;
        case SDL_SCANCODE_APPLICATION: return KEY_MENU;
        default: return KEY_UNKNOWN;
        }
    }

    int sdlModsToAppMods(SDL_Keymod sdl_mods) {
        int mods = KEYMOD_NONE;
        if (sdl_mods & SDL_KMOD_SHIFT)
            mods |= KEYMOD_SHIFT;
        if (sdl_mods & SDL_KMOD_CTRL)
            mods |= KEYMOD_CTRL;
        if (sdl_mods & SDL_KMOD_ALT)
            mods |= KEYMOD_ALT;
        if (sdl_mods & SDL_KMOD_GUI)
            mods |= KEYMOD_SUPER;
        return mods;
    }

    int sdlMouseButtonToApp(uint8_t sdl_button) {
        switch (sdl_button) {
        case SDL_BUTTON_LEFT: return static_cast<int>(AppMouseButton::LEFT);
        case SDL_BUTTON_RIGHT: return static_cast<int>(AppMouseButton::RIGHT);
        case SDL_BUTTON_MIDDLE: return static_cast<int>(AppMouseButton::MIDDLE);
        default: return sdl_button - 1;
        }
    }

    SDL_Scancode appKeyToSdlScancode(int app_key) {
        if (app_key >= KEY_A && app_key <= KEY_Z)
            return static_cast<SDL_Scancode>(SDL_SCANCODE_A + (app_key - KEY_A));

        if (app_key >= KEY_0 && app_key <= KEY_9) {
            const int digit = app_key - KEY_0;
            return digit == 0 ? SDL_SCANCODE_0
                              : static_cast<SDL_Scancode>(SDL_SCANCODE_1 + digit - 1);
        }

        if (app_key >= KEY_KP_0 && app_key <= KEY_KP_9) {
            const int digit = app_key - KEY_KP_0;
            return digit == 0 ? SDL_SCANCODE_KP_0
                              : static_cast<SDL_Scancode>(SDL_SCANCODE_KP_1 + digit - 1);
        }

        switch (app_key) {
        case KEY_SPACE: return SDL_SCANCODE_SPACE;
        case KEY_APOSTROPHE: return SDL_SCANCODE_APOSTROPHE;
        case KEY_COMMA: return SDL_SCANCODE_COMMA;
        case KEY_MINUS: return SDL_SCANCODE_MINUS;
        case KEY_PERIOD: return SDL_SCANCODE_PERIOD;
        case KEY_SLASH: return SDL_SCANCODE_SLASH;
        case KEY_SEMICOLON: return SDL_SCANCODE_SEMICOLON;
        case KEY_EQUAL: return SDL_SCANCODE_EQUALS;
        case KEY_LEFT_BRACKET: return SDL_SCANCODE_LEFTBRACKET;
        case KEY_BACKSLASH: return SDL_SCANCODE_BACKSLASH;
        case KEY_RIGHT_BRACKET: return SDL_SCANCODE_RIGHTBRACKET;
        case KEY_GRAVE_ACCENT: return SDL_SCANCODE_GRAVE;
        case KEY_ESCAPE: return SDL_SCANCODE_ESCAPE;
        case KEY_ENTER: return SDL_SCANCODE_RETURN;
        case KEY_TAB: return SDL_SCANCODE_TAB;
        case KEY_BACKSPACE: return SDL_SCANCODE_BACKSPACE;
        case KEY_INSERT: return SDL_SCANCODE_INSERT;
        case KEY_DELETE: return SDL_SCANCODE_DELETE;
        case KEY_RIGHT: return SDL_SCANCODE_RIGHT;
        case KEY_LEFT: return SDL_SCANCODE_LEFT;
        case KEY_DOWN: return SDL_SCANCODE_DOWN;
        case KEY_UP: return SDL_SCANCODE_UP;
        case KEY_PAGE_UP: return SDL_SCANCODE_PAGEUP;
        case KEY_PAGE_DOWN: return SDL_SCANCODE_PAGEDOWN;
        case KEY_HOME: return SDL_SCANCODE_HOME;
        case KEY_END: return SDL_SCANCODE_END;
        case KEY_CAPS_LOCK: return SDL_SCANCODE_CAPSLOCK;
        case KEY_SCROLL_LOCK: return SDL_SCANCODE_SCROLLLOCK;
        case KEY_NUM_LOCK: return SDL_SCANCODE_NUMLOCKCLEAR;
        case KEY_PRINT_SCREEN: return SDL_SCANCODE_PRINTSCREEN;
        case KEY_PAUSE: return SDL_SCANCODE_PAUSE;
        case KEY_F1: return SDL_SCANCODE_F1;
        case KEY_F2: return SDL_SCANCODE_F2;
        case KEY_F3: return SDL_SCANCODE_F3;
        case KEY_F4: return SDL_SCANCODE_F4;
        case KEY_F5: return SDL_SCANCODE_F5;
        case KEY_F6: return SDL_SCANCODE_F6;
        case KEY_F7: return SDL_SCANCODE_F7;
        case KEY_F8: return SDL_SCANCODE_F8;
        case KEY_F9: return SDL_SCANCODE_F9;
        case KEY_F10: return SDL_SCANCODE_F10;
        case KEY_F11: return SDL_SCANCODE_F11;
        case KEY_F12: return SDL_SCANCODE_F12;
        case KEY_KP_DECIMAL: return SDL_SCANCODE_KP_PERIOD;
        case KEY_KP_DIVIDE: return SDL_SCANCODE_KP_DIVIDE;
        case KEY_KP_MULTIPLY: return SDL_SCANCODE_KP_MULTIPLY;
        case KEY_KP_SUBTRACT: return SDL_SCANCODE_KP_MINUS;
        case KEY_KP_ADD: return SDL_SCANCODE_KP_PLUS;
        case KEY_KP_ENTER: return SDL_SCANCODE_KP_ENTER;
        case KEY_KP_EQUAL: return SDL_SCANCODE_KP_EQUALS;
        case KEY_LEFT_SHIFT: return SDL_SCANCODE_LSHIFT;
        case KEY_LEFT_CONTROL: return SDL_SCANCODE_LCTRL;
        case KEY_LEFT_ALT: return SDL_SCANCODE_LALT;
        case KEY_LEFT_SUPER: return SDL_SCANCODE_LGUI;
        case KEY_RIGHT_SHIFT: return SDL_SCANCODE_RSHIFT;
        case KEY_RIGHT_CONTROL: return SDL_SCANCODE_RCTRL;
        case KEY_RIGHT_ALT: return SDL_SCANCODE_RALT;
        case KEY_RIGHT_SUPER: return SDL_SCANCODE_RGUI;
        case KEY_MENU: return SDL_SCANCODE_APPLICATION;
        default: return SDL_SCANCODE_UNKNOWN;
        }
    }

} // namespace lfs::vis::input
