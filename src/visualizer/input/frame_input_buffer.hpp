/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis {

    enum class FrameInputEventKind { KeyDown,
                                     KeyUp,
                                     Text,
                                     TextEditing,
                                     MouseButton };

    struct InputEventDispatch {
        bool consumed = false;
    };

    struct FrameInputEvent {
        FrameInputEventKind kind = FrameInputEventKind::KeyDown;
        SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;
        SDL_Keymod modifiers = SDL_KMOD_NONE;
        bool repeat = false;
        std::string text{};
        int editing_start = -1;
        int editing_length = -1;
        size_t mouse_button_index = 0;
        // Shared by transient panel adapters during one global dispatch.
        InputEventDispatch* dispatch = nullptr;
    };

    struct FrameMouseButtonEvent {
        uint8_t button = 0;
        bool down = false;
        float x = 0.0f;
        float y = 0.0f;
        uint64_t timestamp = 0;
        uint8_t clicks = 0;
    };

    struct FrameInputBuffer {
        uint64_t serial = 0;
        float mouse_x = 0;
        float mouse_y = 0;
        bool mouse_down[3] = {};
        bool mouse_clicked[3] = {};
        bool mouse_released[3] = {};
        float mouse_wheel = 0;
        float mouse_wheel_x = 0;
        std::vector<FrameMouseButtonEvent> mouse_button_events;
        std::vector<FrameInputEvent> input_events;
        // Aggregate key state for shortcuts; text consumers replay input_events only.
        std::vector<SDL_Scancode> keys_pressed;

        bool had_event = false;
        bool mouse_moved = false;
        bool window_event = false;
        bool user_event = false;
        SDL_Keymod key_mods = SDL_KMOD_NONE;
        int window_w = 0;
        int window_h = 0;
        std::chrono::steady_clock::time_point poll_time{};

        [[nodiscard]] bool hasUserInput() const {
            return mouse_moved || window_event || mouse_wheel != 0.0f || mouse_wheel_x != 0.0f ||
                   !mouse_button_events.empty() || !input_events.empty() || !keys_pressed.empty();
        }

        void beginFrame() {
            ++serial;
            mouse_clicked[0] = mouse_clicked[1] = mouse_clicked[2] = false;
            mouse_released[0] = mouse_released[1] = mouse_released[2] = false;
            mouse_wheel = 0;
            mouse_wheel_x = 0;
            mouse_button_events.clear();
            input_events.clear();
            keys_pressed.clear();

            had_event = false;
            mouse_moved = false;
            window_event = false;
            user_event = false;
        }

        void processEvent(const SDL_Event& event, const SDL_WindowID target_window_id = 0) {
            if (!matchesWindow(event, target_window_id))
                return;

            had_event = true;
            if (event.type >= SDL_EVENT_WINDOW_FIRST && event.type <= SDL_EVENT_WINDOW_LAST)
                window_event = true;
            else if (event.type == SDL_EVENT_USER)
                user_event = true;

            switch (event.type) {
            case SDL_EVENT_MOUSE_MOTION:
                mouse_moved = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                const int idx = buttonIndex(event.button.button);
                if (idx >= 0) {
                    mouse_button_events.push_back({
                        .button = static_cast<uint8_t>(idx),
                        .down = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN,
                        .x = event.button.x,
                        .y = event.button.y,
                        .timestamp = event.button.timestamp,
                        .clicks = event.button.clicks,
                    });
                    input_events.push_back({.kind = FrameInputEventKind::MouseButton,
                                            .mouse_button_index = mouse_button_events.size() - 1});
                    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
                        mouse_clicked[idx] = true;
                    else
                        mouse_released[idx] = true;
                }
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL:
                mouse_wheel += event.wheel.y;
                mouse_wheel_x += event.wheel.x;
                break;
            case SDL_EVENT_KEY_DOWN:
                input_events.push_back({.kind = FrameInputEventKind::KeyDown, .scancode = event.key.scancode, .modifiers = event.key.mod, .repeat = event.key.repeat});
                if (!event.key.repeat)
                    keys_pressed.push_back(event.key.scancode);
                break;
            case SDL_EVENT_KEY_UP:
                input_events.push_back({.kind = FrameInputEventKind::KeyUp, .scancode = event.key.scancode, .modifiers = event.key.mod});

                break;
            case SDL_EVENT_TEXT_INPUT:
                if (event.text.text)
                    input_events.push_back({.kind = FrameInputEventKind::Text, .text = event.text.text});
                break;
            case SDL_EVENT_TEXT_EDITING:
                input_events.push_back({.kind = FrameInputEventKind::TextEditing,
                                        .text = event.edit.text ? event.edit.text : "",
                                        .editing_start = event.edit.start,
                                        .editing_length = event.edit.length});
                break;
            default:
                break;
            }
        }

        void finalize(SDL_Window* window) {
            assert(window);
            poll_time = std::chrono::steady_clock::now();
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

            if (event.type >= SDL_EVENT_WINDOW_FIRST && event.type <= SDL_EVENT_WINDOW_LAST)
                return event.window.windowID == target_window_id;

            switch (event.type) {
            case SDL_EVENT_MOUSE_MOTION:
                return event.motion.windowID == target_window_id;
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
            case SDL_EVENT_TEXT_EDITING:
                return event.edit.windowID == target_window_id;
            case SDL_EVENT_DROP_FILE:
            case SDL_EVENT_DROP_COMPLETE:
                return event.drop.windowID == target_window_id;
            default:
                return true;
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
