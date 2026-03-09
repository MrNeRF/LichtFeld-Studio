/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_scancode.h>
#include <vector>

namespace lfs::vis::gui {

    inline Rml::Input::KeyIdentifier sdlKeycodeToRml(SDL_Keycode sdlkey) {
        // clang-format off
        switch (sdlkey) {
        case SDLK_UNKNOWN:      return Rml::Input::KI_UNKNOWN;
        case SDLK_ESCAPE:       return Rml::Input::KI_ESCAPE;
        case SDLK_SPACE:        return Rml::Input::KI_SPACE;
        case SDLK_0:            return Rml::Input::KI_0;
        case SDLK_1:            return Rml::Input::KI_1;
        case SDLK_2:            return Rml::Input::KI_2;
        case SDLK_3:            return Rml::Input::KI_3;
        case SDLK_4:            return Rml::Input::KI_4;
        case SDLK_5:            return Rml::Input::KI_5;
        case SDLK_6:            return Rml::Input::KI_6;
        case SDLK_7:            return Rml::Input::KI_7;
        case SDLK_8:            return Rml::Input::KI_8;
        case SDLK_9:            return Rml::Input::KI_9;
        case SDLK_A:            return Rml::Input::KI_A;
        case SDLK_B:            return Rml::Input::KI_B;
        case SDLK_C:            return Rml::Input::KI_C;
        case SDLK_D:            return Rml::Input::KI_D;
        case SDLK_E:            return Rml::Input::KI_E;
        case SDLK_F:            return Rml::Input::KI_F;
        case SDLK_G:            return Rml::Input::KI_G;
        case SDLK_H:            return Rml::Input::KI_H;
        case SDLK_I:            return Rml::Input::KI_I;
        case SDLK_J:            return Rml::Input::KI_J;
        case SDLK_K:            return Rml::Input::KI_K;
        case SDLK_L:            return Rml::Input::KI_L;
        case SDLK_M:            return Rml::Input::KI_M;
        case SDLK_N:            return Rml::Input::KI_N;
        case SDLK_O:            return Rml::Input::KI_O;
        case SDLK_P:            return Rml::Input::KI_P;
        case SDLK_Q:            return Rml::Input::KI_Q;
        case SDLK_R:            return Rml::Input::KI_R;
        case SDLK_S:            return Rml::Input::KI_S;
        case SDLK_T:            return Rml::Input::KI_T;
        case SDLK_U:            return Rml::Input::KI_U;
        case SDLK_V:            return Rml::Input::KI_V;
        case SDLK_W:            return Rml::Input::KI_W;
        case SDLK_X:            return Rml::Input::KI_X;
        case SDLK_Y:            return Rml::Input::KI_Y;
        case SDLK_Z:            return Rml::Input::KI_Z;
        case SDLK_SEMICOLON:    return Rml::Input::KI_OEM_1;
        case SDLK_EQUALS:       return Rml::Input::KI_OEM_PLUS;
        case SDLK_PLUS:         return Rml::Input::KI_OEM_PLUS;
        case SDLK_COMMA:        return Rml::Input::KI_OEM_COMMA;
        case SDLK_MINUS:        return Rml::Input::KI_OEM_MINUS;
        case SDLK_PERIOD:       return Rml::Input::KI_OEM_PERIOD;
        case SDLK_SLASH:        return Rml::Input::KI_OEM_2;
        case SDLK_GRAVE:        return Rml::Input::KI_OEM_3;
        case SDLK_LEFTBRACKET:  return Rml::Input::KI_OEM_4;
        case SDLK_BACKSLASH:    return Rml::Input::KI_OEM_5;
        case SDLK_RIGHTBRACKET: return Rml::Input::KI_OEM_6;
        case SDLK_DBLAPOSTROPHE:return Rml::Input::KI_OEM_7;
        case SDLK_KP_0:         return Rml::Input::KI_NUMPAD0;
        case SDLK_KP_1:         return Rml::Input::KI_NUMPAD1;
        case SDLK_KP_2:         return Rml::Input::KI_NUMPAD2;
        case SDLK_KP_3:         return Rml::Input::KI_NUMPAD3;
        case SDLK_KP_4:         return Rml::Input::KI_NUMPAD4;
        case SDLK_KP_5:         return Rml::Input::KI_NUMPAD5;
        case SDLK_KP_6:         return Rml::Input::KI_NUMPAD6;
        case SDLK_KP_7:         return Rml::Input::KI_NUMPAD7;
        case SDLK_KP_8:         return Rml::Input::KI_NUMPAD8;
        case SDLK_KP_9:         return Rml::Input::KI_NUMPAD9;
        case SDLK_KP_ENTER:     return Rml::Input::KI_NUMPADENTER;
        case SDLK_KP_MULTIPLY:  return Rml::Input::KI_MULTIPLY;
        case SDLK_KP_PLUS:      return Rml::Input::KI_ADD;
        case SDLK_KP_MINUS:     return Rml::Input::KI_SUBTRACT;
        case SDLK_KP_PERIOD:    return Rml::Input::KI_DECIMAL;
        case SDLK_KP_DIVIDE:    return Rml::Input::KI_DIVIDE;
        case SDLK_KP_EQUALS:    return Rml::Input::KI_OEM_NEC_EQUAL;
        case SDLK_BACKSPACE:    return Rml::Input::KI_BACK;
        case SDLK_TAB:          return Rml::Input::KI_TAB;
        case SDLK_CLEAR:        return Rml::Input::KI_CLEAR;
        case SDLK_RETURN:       return Rml::Input::KI_RETURN;
        case SDLK_PAUSE:        return Rml::Input::KI_PAUSE;
        case SDLK_CAPSLOCK:     return Rml::Input::KI_CAPITAL;
        case SDLK_PAGEUP:       return Rml::Input::KI_PRIOR;
        case SDLK_PAGEDOWN:     return Rml::Input::KI_NEXT;
        case SDLK_END:          return Rml::Input::KI_END;
        case SDLK_HOME:         return Rml::Input::KI_HOME;
        case SDLK_LEFT:         return Rml::Input::KI_LEFT;
        case SDLK_UP:           return Rml::Input::KI_UP;
        case SDLK_RIGHT:        return Rml::Input::KI_RIGHT;
        case SDLK_DOWN:         return Rml::Input::KI_DOWN;
        case SDLK_INSERT:       return Rml::Input::KI_INSERT;
        case SDLK_DELETE:       return Rml::Input::KI_DELETE;
        case SDLK_HELP:         return Rml::Input::KI_HELP;
        case SDLK_F1:           return Rml::Input::KI_F1;
        case SDLK_F2:           return Rml::Input::KI_F2;
        case SDLK_F3:           return Rml::Input::KI_F3;
        case SDLK_F4:           return Rml::Input::KI_F4;
        case SDLK_F5:           return Rml::Input::KI_F5;
        case SDLK_F6:           return Rml::Input::KI_F6;
        case SDLK_F7:           return Rml::Input::KI_F7;
        case SDLK_F8:           return Rml::Input::KI_F8;
        case SDLK_F9:           return Rml::Input::KI_F9;
        case SDLK_F10:          return Rml::Input::KI_F10;
        case SDLK_F11:          return Rml::Input::KI_F11;
        case SDLK_F12:          return Rml::Input::KI_F12;
        case SDLK_F13:          return Rml::Input::KI_F13;
        case SDLK_F14:          return Rml::Input::KI_F14;
        case SDLK_F15:          return Rml::Input::KI_F15;
        case SDLK_NUMLOCKCLEAR: return Rml::Input::KI_NUMLOCK;
        case SDLK_SCROLLLOCK:   return Rml::Input::KI_SCROLL;
        case SDLK_LSHIFT:       return Rml::Input::KI_LSHIFT;
        case SDLK_RSHIFT:       return Rml::Input::KI_RSHIFT;
        case SDLK_LCTRL:        return Rml::Input::KI_LCONTROL;
        case SDLK_RCTRL:        return Rml::Input::KI_RCONTROL;
        case SDLK_LALT:         return Rml::Input::KI_LMENU;
        case SDLK_RALT:         return Rml::Input::KI_RMENU;
        case SDLK_LGUI:         return Rml::Input::KI_LMETA;
        case SDLK_RGUI:         return Rml::Input::KI_RMETA;
        default:                return Rml::Input::KI_UNKNOWN;
        }
        // clang-format on
    }

    inline Rml::Input::KeyIdentifier sdlScancodeToRml(SDL_Scancode sc) {
        // Shortcut and navigation keys should be layout-independent. Text entry
        // still comes from SDL text events through text_codepoints.
        // clang-format off
        switch (sc) {
        case SDL_SCANCODE_SPACE:          return Rml::Input::KI_SPACE;
        case SDL_SCANCODE_APOSTROPHE:     return Rml::Input::KI_OEM_7;
        case SDL_SCANCODE_COMMA:          return Rml::Input::KI_OEM_COMMA;
        case SDL_SCANCODE_MINUS:          return Rml::Input::KI_OEM_MINUS;
        case SDL_SCANCODE_PERIOD:         return Rml::Input::KI_OEM_PERIOD;
        case SDL_SCANCODE_SLASH:          return Rml::Input::KI_OEM_2;
        case SDL_SCANCODE_SEMICOLON:      return Rml::Input::KI_OEM_1;
        case SDL_SCANCODE_EQUALS:         return Rml::Input::KI_OEM_PLUS;
        case SDL_SCANCODE_LEFTBRACKET:    return Rml::Input::KI_OEM_4;
        case SDL_SCANCODE_BACKSLASH:      return Rml::Input::KI_OEM_5;
        case SDL_SCANCODE_NONUSBACKSLASH: return Rml::Input::KI_OEM_102;
        case SDL_SCANCODE_RIGHTBRACKET:   return Rml::Input::KI_OEM_6;
        case SDL_SCANCODE_GRAVE:          return Rml::Input::KI_OEM_3;
        case SDL_SCANCODE_KP_0:           return Rml::Input::KI_NUMPAD0;
        case SDL_SCANCODE_KP_1:           return Rml::Input::KI_NUMPAD1;
        case SDL_SCANCODE_KP_2:           return Rml::Input::KI_NUMPAD2;
        case SDL_SCANCODE_KP_3:           return Rml::Input::KI_NUMPAD3;
        case SDL_SCANCODE_KP_4:           return Rml::Input::KI_NUMPAD4;
        case SDL_SCANCODE_KP_5:           return Rml::Input::KI_NUMPAD5;
        case SDL_SCANCODE_KP_6:           return Rml::Input::KI_NUMPAD6;
        case SDL_SCANCODE_KP_7:           return Rml::Input::KI_NUMPAD7;
        case SDL_SCANCODE_KP_8:           return Rml::Input::KI_NUMPAD8;
        case SDL_SCANCODE_KP_9:           return Rml::Input::KI_NUMPAD9;
        case SDL_SCANCODE_KP_ENTER:       return Rml::Input::KI_NUMPADENTER;
        case SDL_SCANCODE_KP_MULTIPLY:    return Rml::Input::KI_MULTIPLY;
        case SDL_SCANCODE_KP_PLUS:        return Rml::Input::KI_ADD;
        case SDL_SCANCODE_KP_MINUS:       return Rml::Input::KI_SUBTRACT;
        case SDL_SCANCODE_KP_PERIOD:      return Rml::Input::KI_DECIMAL;
        case SDL_SCANCODE_KP_DIVIDE:      return Rml::Input::KI_DIVIDE;
        case SDL_SCANCODE_KP_EQUALS:      return Rml::Input::KI_OEM_NEC_EQUAL;
        case SDL_SCANCODE_BACKSPACE:      return Rml::Input::KI_BACK;
        case SDL_SCANCODE_TAB:            return Rml::Input::KI_TAB;
        case SDL_SCANCODE_CLEAR:          return Rml::Input::KI_CLEAR;
        case SDL_SCANCODE_RETURN:         return Rml::Input::KI_RETURN;
        case SDL_SCANCODE_PAUSE:          return Rml::Input::KI_PAUSE;
        case SDL_SCANCODE_CAPSLOCK:       return Rml::Input::KI_CAPITAL;
        case SDL_SCANCODE_PAGEUP:         return Rml::Input::KI_PRIOR;
        case SDL_SCANCODE_PAGEDOWN:       return Rml::Input::KI_NEXT;
        case SDL_SCANCODE_END:            return Rml::Input::KI_END;
        case SDL_SCANCODE_HOME:           return Rml::Input::KI_HOME;
        case SDL_SCANCODE_LEFT:           return Rml::Input::KI_LEFT;
        case SDL_SCANCODE_UP:             return Rml::Input::KI_UP;
        case SDL_SCANCODE_RIGHT:          return Rml::Input::KI_RIGHT;
        case SDL_SCANCODE_DOWN:           return Rml::Input::KI_DOWN;
        case SDL_SCANCODE_INSERT:         return Rml::Input::KI_INSERT;
        case SDL_SCANCODE_DELETE:         return Rml::Input::KI_DELETE;
        case SDL_SCANCODE_HELP:           return Rml::Input::KI_HELP;
        case SDL_SCANCODE_F1:             return Rml::Input::KI_F1;
        case SDL_SCANCODE_F2:             return Rml::Input::KI_F2;
        case SDL_SCANCODE_F3:             return Rml::Input::KI_F3;
        case SDL_SCANCODE_F4:             return Rml::Input::KI_F4;
        case SDL_SCANCODE_F5:             return Rml::Input::KI_F5;
        case SDL_SCANCODE_F6:             return Rml::Input::KI_F6;
        case SDL_SCANCODE_F7:             return Rml::Input::KI_F7;
        case SDL_SCANCODE_F8:             return Rml::Input::KI_F8;
        case SDL_SCANCODE_F9:             return Rml::Input::KI_F9;
        case SDL_SCANCODE_F10:            return Rml::Input::KI_F10;
        case SDL_SCANCODE_F11:            return Rml::Input::KI_F11;
        case SDL_SCANCODE_F12:            return Rml::Input::KI_F12;
        case SDL_SCANCODE_F13:            return Rml::Input::KI_F13;
        case SDL_SCANCODE_F14:            return Rml::Input::KI_F14;
        case SDL_SCANCODE_F15:            return Rml::Input::KI_F15;
        case SDL_SCANCODE_NUMLOCKCLEAR:   return Rml::Input::KI_NUMLOCK;
        case SDL_SCANCODE_SCROLLLOCK:     return Rml::Input::KI_SCROLL;
        case SDL_SCANCODE_LSHIFT:         return Rml::Input::KI_LSHIFT;
        case SDL_SCANCODE_RSHIFT:         return Rml::Input::KI_RSHIFT;
        case SDL_SCANCODE_LCTRL:          return Rml::Input::KI_LCONTROL;
        case SDL_SCANCODE_RCTRL:          return Rml::Input::KI_RCONTROL;
        case SDL_SCANCODE_LALT:           return Rml::Input::KI_LMENU;
        case SDL_SCANCODE_RALT:           return Rml::Input::KI_RMENU;
        case SDL_SCANCODE_LGUI:           return Rml::Input::KI_LMETA;
        case SDL_SCANCODE_RGUI:           return Rml::Input::KI_RMETA;
        case SDL_SCANCODE_APPLICATION:    return Rml::Input::KI_APPS;
        default:                          break;
        }
        // clang-format on

        if (sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z)
            return static_cast<Rml::Input::KeyIdentifier>(
                Rml::Input::KI_A + (sc - SDL_SCANCODE_A));

        if (sc == SDL_SCANCODE_0)
            return Rml::Input::KI_0;
        if (sc >= SDL_SCANCODE_1 && sc <= SDL_SCANCODE_9)
            return static_cast<Rml::Input::KeyIdentifier>(
                Rml::Input::KI_1 + (sc - SDL_SCANCODE_1));

        return Rml::Input::KI_UNKNOWN;
    }

    inline int sdlModsToRml(const SDL_Keymod sdl_mods) {
        int retval = 0;
        if (sdl_mods & SDL_KMOD_CTRL)
            retval |= Rml::Input::KM_CTRL;
        if (sdl_mods & SDL_KMOD_SHIFT)
            retval |= Rml::Input::KM_SHIFT;
        if (sdl_mods & SDL_KMOD_ALT)
            retval |= Rml::Input::KM_ALT;
        if (sdl_mods & SDL_KMOD_GUI)
            retval |= Rml::Input::KM_META;
        if (sdl_mods & SDL_KMOD_NUM)
            retval |= Rml::Input::KM_NUMLOCK;
        if (sdl_mods & SDL_KMOD_CAPS)
            retval |= Rml::Input::KM_CAPSLOCK;
        return retval;
    }

    inline int sdlModsToRml() {
        return sdlModsToRml(SDL_GetModState());
    }

    inline int sdlModsToRml(const bool ctrl, const bool shift,
                            const bool alt, const bool super) {
        int retval = 0;
        if (ctrl)
            retval |= Rml::Input::KM_CTRL;
        if (shift)
            retval |= Rml::Input::KM_SHIFT;
        if (alt)
            retval |= Rml::Input::KM_ALT;
        if (super)
            retval |= Rml::Input::KM_META;
        return retval;
    }

    inline bool hasKey(const std::vector<int>& keys, SDL_Scancode target) {
        const int t = static_cast<int>(target);
        for (int k : keys)
            if (k == t)
                return true;
        return false;
    }

} // namespace lfs::vis::gui
