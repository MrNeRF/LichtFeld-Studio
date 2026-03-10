/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <cassert>
#include <vector>

namespace lfs::vis {

    struct FrameInputBuffer {
        float mouse_x = 0;
        float mouse_y = 0;
        bool mouse_down[3] = {};
        bool mouse_clicked[3] = {};
        bool mouse_released[3] = {};
        float mouse_wheel = 0;
        std::vector<SDL_Scancode> keys_pressed;
        std::vector<SDL_Scancode> keys_released;
        std::vector<uint32_t> text_codepoints;
        SDL_Keymod key_mods = SDL_KMOD_NONE;
        int window_w = 0;
        int window_h = 0;

        void beginFrame() {
            mouse_clicked[0] = mouse_clicked[1] = mouse_clicked[2] = false;
            mouse_released[0] = mouse_released[1] = mouse_released[2] = false;
            mouse_wheel = 0;
            keys_pressed.clear();
            keys_released.clear();
            text_codepoints.clear();
        }

        void processEvent(const SDL_Event& event, const SDL_WindowID target_window_id = 0) {
            if (!matchesWindow(event, target_window_id))
                return;

            switch (event.type) {
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                const int idx = buttonIndex(event.button.button);
                if (idx >= 0) {
                    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
                        mouse_clicked[idx] = true;
                    else
                        mouse_released[idx] = true;
                }
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL:
                mouse_wheel += event.wheel.y;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (!event.key.repeat)
                    keys_pressed.push_back(event.key.scancode);
                break;
            case SDL_EVENT_KEY_UP:
                keys_released.push_back(event.key.scancode);
                break;
            case SDL_EVENT_TEXT_INPUT:
                decodeUtf8(event.text.text, text_codepoints);
                break;
            default:
                break;
            }
        }

        void finalize(SDL_Window* window) {
            assert(window);
            const SDL_MouseButtonFlags buttons = SDL_GetMouseState(&mouse_x, &mouse_y);
            mouse_down[0] = (buttons & SDL_BUTTON_LMASK) != 0;
            mouse_down[1] = (buttons & SDL_BUTTON_RMASK) != 0;
            mouse_down[2] = (buttons & SDL_BUTTON_MMASK) != 0;
            key_mods = SDL_GetModState();
            int w = 0, h = 0;
            SDL_GetWindowSize(window, &w, &h);
            window_w = w;
            window_h = h;
        }

    private:
        static bool matchesWindow(const SDL_Event& event, const SDL_WindowID target_window_id) {
            if (target_window_id == 0)
                return true;

            switch (event.type) {
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            case SDL_EVENT_WINDOW_FOCUS_LOST:
            case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
                return event.window.windowID == target_window_id;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP:
                return event.button.windowID == target_window_id;
            case SDL_EVENT_MOUSE_WHEEL:
                return event.wheel.windowID == target_window_id;
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP:
                return event.key.windowID == target_window_id;
            case SDL_EVENT_TEXT_INPUT:
                return event.text.windowID == target_window_id;
            case SDL_EVENT_DROP_FILE:
            case SDL_EVENT_DROP_COMPLETE:
                return event.drop.windowID == target_window_id;
            default:
                return true;
            }
        }

        static bool isContinuationByte(const unsigned char c) {
            return (c & 0xC0u) == 0x80u;
        }

        static void decodeUtf8(const char* text, std::vector<uint32_t>& out) {
            if (!text)
                return;
            for (size_t i = 0; text[i] != '\0';) {
                uint32_t cp = 0;
                const auto c = static_cast<unsigned char>(text[i]);
                if (c < 0x80) {
                    cp = c;
                    i += 1;
                } else if ((c & 0xE0u) == 0xC0u) {
                    if (!text[i + 1] ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 1]))) {
                        i += 1;
                        continue;
                    }
                    cp = (c & 0x1F) << 6;
                    cp |= static_cast<unsigned char>(text[i + 1]) & 0x3F;
                    if (cp < 0x80) {
                        i += 1;
                        continue;
                    }
                    i += 2;
                } else if ((c & 0xF0u) == 0xE0u) {
                    if (!text[i + 1] || !text[i + 2] ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 1])) ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 2]))) {
                        i += 1;
                        continue;
                    }
                    cp = (c & 0x0F) << 12;
                    cp |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6;
                    cp |= static_cast<unsigned char>(text[i + 2]) & 0x3F;
                    if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) {
                        i += 1;
                        continue;
                    }
                    i += 3;
                } else if ((c & 0xF8u) == 0xF0u) {
                    if (!text[i + 1] || !text[i + 2] || !text[i + 3] ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 1])) ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 2])) ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 3]))) {
                        i += 1;
                        continue;
                    }
                    cp = (c & 0x07) << 18;
                    cp |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12;
                    cp |= (static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6;
                    cp |= static_cast<unsigned char>(text[i + 3]) & 0x3F;
                    if (cp < 0x10000 || cp > 0x10FFFF) {
                        i += 1;
                        continue;
                    }
                    i += 4;
                } else {
                    i += 1;
                    continue;
                }
                out.push_back(cp);
            }
        }

        static int buttonIndex(int sdl_button) {
            switch (sdl_button) {
            case SDL_BUTTON_LEFT: return 0;
            case SDL_BUTTON_RIGHT: return 1;
            case SDL_BUTTON_MIDDLE: return 2;
            default: return -1;
            }
        }
    };

} // namespace lfs::vis
