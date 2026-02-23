/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/sdl_key_mapping.hpp"
#include "input/key_codes.hpp"

namespace lfs::vis::input {

    int sdlKeycodeToAppKey(SDL_Keycode sdl_key) {
        // ASCII-range keys map 1:1 (SDL uses lowercase, GLFW/app uses uppercase)
        if (sdl_key >= SDLK_A && sdl_key <= SDLK_Z)
            return KEY_A + (sdl_key - SDLK_A);
        if (sdl_key >= SDLK_0 && sdl_key <= SDLK_9)
            return KEY_0 + (sdl_key - SDLK_0);

        switch (sdl_key) {
        case SDLK_SPACE: return KEY_SPACE;
        case SDLK_APOSTROPHE: return KEY_APOSTROPHE;
        case SDLK_COMMA: return KEY_COMMA;
        case SDLK_MINUS: return KEY_MINUS;
        case SDLK_PERIOD: return KEY_PERIOD;
        case SDLK_SLASH: return KEY_SLASH;
        case SDLK_SEMICOLON: return KEY_SEMICOLON;
        case SDLK_EQUALS: return KEY_EQUAL;
        case SDLK_LEFTBRACKET: return KEY_LEFT_BRACKET;
        case SDLK_BACKSLASH: return KEY_BACKSLASH;
        case SDLK_RIGHTBRACKET: return KEY_RIGHT_BRACKET;
        case SDLK_GRAVE: return KEY_GRAVE_ACCENT;
        case SDLK_ESCAPE: return KEY_ESCAPE;
        case SDLK_RETURN: return KEY_ENTER;
        case SDLK_TAB: return KEY_TAB;
        case SDLK_BACKSPACE: return KEY_BACKSPACE;
        case SDLK_INSERT: return KEY_INSERT;
        case SDLK_DELETE: return KEY_DELETE;
        case SDLK_RIGHT: return KEY_RIGHT;
        case SDLK_LEFT: return KEY_LEFT;
        case SDLK_DOWN: return KEY_DOWN;
        case SDLK_UP: return KEY_UP;
        case SDLK_PAGEUP: return KEY_PAGE_UP;
        case SDLK_PAGEDOWN: return KEY_PAGE_DOWN;
        case SDLK_HOME: return KEY_HOME;
        case SDLK_END: return KEY_END;
        case SDLK_CAPSLOCK: return KEY_CAPS_LOCK;
        case SDLK_SCROLLLOCK: return KEY_SCROLL_LOCK;
        case SDLK_NUMLOCKCLEAR: return KEY_NUM_LOCK;
        case SDLK_PRINTSCREEN: return KEY_PRINT_SCREEN;
        case SDLK_PAUSE: return KEY_PAUSE;
        case SDLK_F1: return KEY_F1;
        case SDLK_F2: return KEY_F2;
        case SDLK_F3: return KEY_F3;
        case SDLK_F4: return KEY_F4;
        case SDLK_F5: return KEY_F5;
        case SDLK_F6: return KEY_F6;
        case SDLK_F7: return KEY_F7;
        case SDLK_F8: return KEY_F8;
        case SDLK_F9: return KEY_F9;
        case SDLK_F10: return KEY_F10;
        case SDLK_F11: return KEY_F11;
        case SDLK_F12: return KEY_F12;
        case SDLK_KP_0: return KEY_KP_0;
        case SDLK_KP_1: return KEY_KP_1;
        case SDLK_KP_2: return KEY_KP_2;
        case SDLK_KP_3: return KEY_KP_3;
        case SDLK_KP_4: return KEY_KP_4;
        case SDLK_KP_5: return KEY_KP_5;
        case SDLK_KP_6: return KEY_KP_6;
        case SDLK_KP_7: return KEY_KP_7;
        case SDLK_KP_8: return KEY_KP_8;
        case SDLK_KP_9: return KEY_KP_9;
        case SDLK_KP_DECIMAL: return KEY_KP_DECIMAL;
        case SDLK_KP_DIVIDE: return KEY_KP_DIVIDE;
        case SDLK_KP_MULTIPLY: return KEY_KP_MULTIPLY;
        case SDLK_KP_MINUS: return KEY_KP_SUBTRACT;
        case SDLK_KP_PLUS: return KEY_KP_ADD;
        case SDLK_KP_ENTER: return KEY_KP_ENTER;
        case SDLK_KP_EQUALS: return KEY_KP_EQUAL;
        case SDLK_LSHIFT: return KEY_LEFT_SHIFT;
        case SDLK_LCTRL: return KEY_LEFT_CONTROL;
        case SDLK_LALT: return KEY_LEFT_ALT;
        case SDLK_LGUI: return KEY_LEFT_SUPER;
        case SDLK_RSHIFT: return KEY_RIGHT_SHIFT;
        case SDLK_RCTRL: return KEY_RIGHT_CONTROL;
        case SDLK_RALT: return KEY_RIGHT_ALT;
        case SDLK_RGUI: return KEY_RIGHT_SUPER;
        case SDLK_APPLICATION: return KEY_MENU;
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

    SDL_Keycode appKeyToSdlKeycode(int app_key) {
        if (app_key >= KEY_A && app_key <= KEY_Z)
            return static_cast<SDL_Keycode>(SDLK_A + (app_key - KEY_A));
        if (app_key >= KEY_0 && app_key <= KEY_9)
            return static_cast<SDL_Keycode>(SDLK_0 + (app_key - KEY_0));

        switch (app_key) {
        case KEY_SPACE: return SDLK_SPACE;
        case KEY_APOSTROPHE: return SDLK_APOSTROPHE;
        case KEY_COMMA: return SDLK_COMMA;
        case KEY_MINUS: return SDLK_MINUS;
        case KEY_PERIOD: return SDLK_PERIOD;
        case KEY_SLASH: return SDLK_SLASH;
        case KEY_SEMICOLON: return SDLK_SEMICOLON;
        case KEY_EQUAL: return SDLK_EQUALS;
        case KEY_LEFT_BRACKET: return SDLK_LEFTBRACKET;
        case KEY_BACKSLASH: return SDLK_BACKSLASH;
        case KEY_RIGHT_BRACKET: return SDLK_RIGHTBRACKET;
        case KEY_GRAVE_ACCENT: return SDLK_GRAVE;
        case KEY_ESCAPE: return SDLK_ESCAPE;
        case KEY_ENTER: return SDLK_RETURN;
        case KEY_TAB: return SDLK_TAB;
        case KEY_BACKSPACE: return SDLK_BACKSPACE;
        case KEY_INSERT: return SDLK_INSERT;
        case KEY_DELETE: return SDLK_DELETE;
        case KEY_RIGHT: return SDLK_RIGHT;
        case KEY_LEFT: return SDLK_LEFT;
        case KEY_DOWN: return SDLK_DOWN;
        case KEY_UP: return SDLK_UP;
        case KEY_PAGE_UP: return SDLK_PAGEUP;
        case KEY_PAGE_DOWN: return SDLK_PAGEDOWN;
        case KEY_HOME: return SDLK_HOME;
        case KEY_END: return SDLK_END;
        case KEY_CAPS_LOCK: return SDLK_CAPSLOCK;
        case KEY_SCROLL_LOCK: return SDLK_SCROLLLOCK;
        case KEY_NUM_LOCK: return SDLK_NUMLOCKCLEAR;
        case KEY_PRINT_SCREEN: return SDLK_PRINTSCREEN;
        case KEY_PAUSE: return SDLK_PAUSE;
        case KEY_F1: return SDLK_F1;
        case KEY_F2: return SDLK_F2;
        case KEY_F3: return SDLK_F3;
        case KEY_F4: return SDLK_F4;
        case KEY_F5: return SDLK_F5;
        case KEY_F6: return SDLK_F6;
        case KEY_F7: return SDLK_F7;
        case KEY_F8: return SDLK_F8;
        case KEY_F9: return SDLK_F9;
        case KEY_F10: return SDLK_F10;
        case KEY_F11: return SDLK_F11;
        case KEY_F12: return SDLK_F12;
        case KEY_KP_0: return SDLK_KP_0;
        case KEY_KP_1: return SDLK_KP_1;
        case KEY_KP_2: return SDLK_KP_2;
        case KEY_KP_3: return SDLK_KP_3;
        case KEY_KP_4: return SDLK_KP_4;
        case KEY_KP_5: return SDLK_KP_5;
        case KEY_KP_6: return SDLK_KP_6;
        case KEY_KP_7: return SDLK_KP_7;
        case KEY_KP_8: return SDLK_KP_8;
        case KEY_KP_9: return SDLK_KP_9;
        case KEY_KP_DECIMAL: return SDLK_KP_DECIMAL;
        case KEY_KP_DIVIDE: return SDLK_KP_DIVIDE;
        case KEY_KP_MULTIPLY: return SDLK_KP_MULTIPLY;
        case KEY_KP_SUBTRACT: return SDLK_KP_MINUS;
        case KEY_KP_ADD: return SDLK_KP_PLUS;
        case KEY_KP_ENTER: return SDLK_KP_ENTER;
        case KEY_KP_EQUAL: return SDLK_KP_EQUALS;
        case KEY_LEFT_SHIFT: return SDLK_LSHIFT;
        case KEY_LEFT_CONTROL: return SDLK_LCTRL;
        case KEY_LEFT_ALT: return SDLK_LALT;
        case KEY_LEFT_SUPER: return SDLK_LGUI;
        case KEY_RIGHT_SHIFT: return SDLK_RSHIFT;
        case KEY_RIGHT_CONTROL: return SDLK_RCTRL;
        case KEY_RIGHT_ALT: return SDLK_RALT;
        case KEY_RIGHT_SUPER: return SDLK_RGUI;
        case KEY_MENU: return SDLK_APPLICATION;
        default: return SDLK_UNKNOWN;
        }
    }

} // namespace lfs::vis::input
